import torch
import torch.nn as nn

from resume import ResUME
from modules import Encoder_x2, Decoder_x2, ChannelEncoder, ChannelDecoder
from mobilenet import MobileNetV2Encoder, MobileNetV2Decoder
from mobilevit import MobileViTEncoder, MobileViTDecoder


class PrefixDepthFusion(nn.Module):
    """Residual conditioning on the receiver-known normalized valid-prefix depth map."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )
        # Start as an exact no-op; the decoder can learn to use the side information gradually.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        return z + self.net(depth_map.to(dtype=z.dtype))


class _DigitalBackboneBase(nn.Module):
    """Shared theory-first digital pipeline supporting stage- and group-wise packets."""

    def __init__(
        self,
        num_stages,
        vq_bitrate_per_stage,
        embedding_dim,
        batch_size,
        device,
        use_prefix_map=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_prefix_map = bool(use_prefix_map)
        self.noise_fusion_enc = ChannelEncoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        self.noise_fusion_dec = ChannelDecoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        self.resume = ResUME(
            num_stages=num_stages,
            vq_bitrate_per_stage=vq_bitrate_per_stage,
            data_dim=embedding_dim,
            batch_size=batch_size,
            device=device,
        )
        self.prefix_depth_fusion = PrefixDepthFusion(embedding_dim)

    def _encode_image(self, x):
        raise NotImplementedError

    def _decode_feature_map(self, z):
        raise NotImplementedError

    def encode_latent(self, x: torch.Tensor, snr: torch.Tensor):
        """Return fused latent [B,N,D], original spatial shape [B,Hl,Wl,D], and SNR gate."""
        z = self._encode_image(x)
        z = z.permute(0, 2, 3, 1).contiguous()
        origin_shape = z.shape
        B, Hl, Wl, D = origin_shape
        z_flat = z.reshape(B * Hl * Wl, D)
        # Existing ChannelEncoder is scalar-SNR conditioned, so a batch uses one sampled SNR.
        snr_fusion = torch.clamp(snr.reshape(-1)[0], 0.0, 15.0)
        z_e = self.noise_fusion_enc(z_flat, snr_fusion)
        return z_e.reshape(B, Hl * Wl, D), origin_shape, snr_fusion

    def decode_latent(self, z: torch.Tensor, origin_shape, snr_fusion):
        B, Hl, Wl, D = origin_shape
        z_flat = z.reshape(B * Hl * Wl, D)
        z_flat = self.noise_fusion_dec(z_flat, snr_fusion)
        z_map = z_flat.reshape(B, Hl, Wl, D).permute(0, 3, 1, 2).contiguous()
        return self._decode_feature_map(z_map)

    def forward(
        self,
        x,
        active_stages,
        snr,
        mod_orders,
        apply_fading=False,
        equalizer="zf",
        rvq_activate=True,
        use_crc_prefix=True,
        packet_mode="stage",
        group_hw=(4, 4),
        use_prefix_map=None,
        return_details=True,
        **_ignored_legacy_kwargs,
    ):
        z_e, origin_shape, snr_fusion = self.encode_latent(x, snr)
        B, Hl, Wl, _ = origin_shape
        if use_prefix_map is None:
            use_prefix_map = self.use_prefix_map

        if rvq_activate:
            resume_out = self.resume(
                z_e,
                active_stages=active_stages,
                snr_db=snr,
                mod_orders=mod_orders,
                apply_fading=apply_fading,
                equalizer=equalizer,
                use_crc_prefix=use_crc_prefix,
                packet_mode=packet_mode,
                latent_hw=(Hl, Wl),
                group_hw=group_hw,
            )
            z_q = resume_out["z_out"]
            if use_prefix_map:
                z_q = self.prefix_depth_fusion(z_q, resume_out["depth_map"])
        else:
            z_q = z_e
            depth_map = torch.ones((B, Hl * Wl, 1), dtype=z_e.dtype, device=z_e.device)
            if use_prefix_map:
                z_q = self.prefix_depth_fusion(z_q, depth_map)
            resume_out = {
                "z_out": z_q,
                "z_hard": z_q,
                "depth_map": depth_map,
                "codebook_loss": z_e.new_zeros(()),
                "commitment_loss": z_e.new_zeros(()),
                "perplexities": [],
                "codebooks_used": None,
                "prefix_len": torch.full((B, 1), int(active_stages), dtype=torch.long, device=x.device),
                "symbol_counts": [],
                "packet_mode": packet_mode,
                "num_groups": 1,
                "group_size": Hl * Wl,
            }

        x_recon = self.decode_latent(z_q, origin_shape, snr_fusion)
        if return_details:
            return x_recon, resume_out, z_e
        return x_recon

    @torch.no_grad()
    def extract_rvq_indices(self, x, snr, active_stages=None):
        z_e, origin_shape, _ = self.encode_latent(x, snr)
        idx, q = self.resume.encode_indices(z_e, active_stages=active_stages)
        return idx, q, origin_shape


class Digital_SemCom(_DigitalBackboneBase):
    def __init__(
        self,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        num_stages,
        vq_bitrate_per_stage,
        embedding_dim,
        batch_size,
        device,
        use_prefix_map=True,
    ):
        super().__init__(
            num_stages,
            vq_bitrate_per_stage,
            embedding_dim,
            batch_size,
            device,
            use_prefix_map=use_prefix_map,
        )
        self._encoder = Encoder_x2(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=4, stride=2, padding=1)
        self._decoder = Decoder_x2(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def _encode_image(self, x):
        return self._pre_vq_conv(self._encoder(x))

    def _decode_feature_map(self, z):
        return self._decoder(z)


class MobileNet_SemCom(_DigitalBackboneBase):
    def __init__(
        self,
        num_stages,
        vq_bitrate_per_stage,
        embedding_dim,
        batch_size,
        device,
        use_prefix_map=True,
    ):
        super().__init__(
            num_stages,
            vq_bitrate_per_stage,
            embedding_dim,
            batch_size,
            device,
            use_prefix_map=use_prefix_map,
        )
        self._encoder = MobileNetV2Encoder(ch_in=3, width_mult=1.0, use_bn=True, latent_ch=embedding_dim)
        self._decoder = MobileNetV2Decoder(in_ch=embedding_dim, out_ch=3, width_mult=1.0, use_bn=True)

    def _encode_image(self, x):
        return self._encoder(x)

    def _decode_feature_map(self, z):
        return self._decoder(z)


class MobileViT_SemCom(_DigitalBackboneBase):
    def __init__(
        self,
        num_stages,
        vq_bitrate_per_stage,
        embedding_dim,
        batch_size,
        device,
        variant="xs",
        use_prefix_map=True,
    ):
        super().__init__(
            num_stages,
            vq_bitrate_per_stage,
            embedding_dim,
            batch_size,
            device,
            use_prefix_map=use_prefix_map,
        )
        assert variant in ["xxs", "xs", "s"]
        if variant == "xxs":
            cfg = dict(c0=16, c1=16, c2=24, c3=48, c4=64, c5=80, cb=160,
                       d1=64, d2=80, d3=96, L1=2, L2=4, L3=3)
            expansion = 2
        elif variant == "xs":
            cfg = dict(c0=16, c1=32, c2=48, c3=64, c4=80, c5=128, cb=192,
                       d1=96, d2=120, d3=144, L1=2, L2=4, L3=3)
            expansion = 4
        else:
            cfg = dict(c0=16, c1=32, c2=64, c3=96, c4=128, c5=128, cb=320,
                       d1=144, d2=192, d3=240, L1=2, L2=4, L3=3)
            expansion = 4
        self._encoder = MobileViTEncoder(
            image_size=(128, 128), dims=(cfg["d1"], cfg["d2"], cfg["d3"]),
            channels=(cfg["c0"], cfg["c1"], cfg["c2"], cfg["c3"], cfg["c4"], cfg["c5"]),
            expansion=expansion, kernel_size=3, patch_size=(2, 2),
            depths=(cfg["L1"], cfg["L2"], cfg["L3"]), latent_ch=embedding_dim,
        )
        self._decoder = MobileViTDecoder(
            image_size=(128, 128), cfg=cfg, kernel_size=3, patch_size=(2, 2)
        )

    def _encode_image(self, x):
        return self._encoder(x)

    def _decode_feature_map(self, z):
        return self._decoder(z)


class Analog_SemCom(nn.Module):
    """Analog reference kept separate from the theory-first digital ResUME path."""

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, target_CBR):
        super().__init__()
        self._encoder = Encoder_x2(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.noise_fusion_enc = ChannelEncoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        self.noise_fusion_dec = ChannelDecoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        bottleneck_dim = max(1, int(target_CBR * 3 * 128 * 128 // 64))
        self.bottleneck_enc = nn.Linear(embedding_dim, bottleneck_dim)
        self.bottleneck_dec = nn.Linear(bottleneck_dim, embedding_dim)
        self._decoder = Decoder_x2(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def analog_channel(self, x, snr_db, apply_fading=False):
        B = x.shape[0]
        snr = snr_db.reshape(-1)[0]
        snr_linear = 10 ** (snr / 10)
        power = x.square().mean()
        noise_std = torch.sqrt(power / snr_linear + 1e-12)
        noise = torch.randn_like(x) * noise_std
        if not apply_fading:
            return x + noise
        h = torch.randn(B, 1, device=x.device)
        return (x * h + noise) / (h + 1e-8)

    def forward(self, x, active_stages, snr, mod_orders, apply_fading=False, **kwargs):
        z = self._pre_vq_conv(self._encoder(x)).permute(0, 2, 3, 1).contiguous()
        origin = z.shape
        z = z.reshape(-1, z.shape[-1])
        snr_fusion = torch.clamp(snr.reshape(-1)[0], 0.0, 15.0)
        z = self.noise_fusion_enc(z, snr_fusion)
        z = self.bottleneck_enc(z)
        z = self.analog_channel(z, snr, apply_fading)
        z = self.bottleneck_dec(z)
        z = self.noise_fusion_dec(z, snr_fusion)
        z = z.reshape(origin).permute(0, 3, 1, 2).contiguous()
        return self._decoder(z), {"codebook_loss": x.new_zeros(()), "commitment_loss": x.new_zeros(())}, None
