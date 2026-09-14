import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from packet_utils import (
    append_crc16,
    bits_to_symbols,
    indices_to_payload_bits,
    packet_group_indices,
    payload_bits_to_indices,
    split_and_check_crc16,
    symbols_to_bits,
)


def generate_constellation(M: int, power: float = 1.0) -> torch.Tensor:
    """Natural-labelled BPSK / square-QAM constellation, normalized to average symbol power."""
    if M == 2:
        c = torch.tensor([-1.0, 1.0], dtype=torch.float32).to(torch.complex64)
    else:
        m = int(round(math.sqrt(M)))
        if m * m != M or (int(math.log2(M)) % 2 != 0):
            raise ValueError(f"Only BPSK and square power-of-two QAM are supported, got M={M}")
        axis = torch.arange(-(m - 1), m, 2, dtype=torch.float32)
        re = axis.repeat_interleave(m)
        im = axis.repeat(m)
        c = torch.complex(re, im)
    c = c / torch.sqrt((c.abs().square().mean() / power).to(torch.float32))
    return c


class ResUME(nn.Module):
    """
    Theory-first ResUME supporting two packetization modes.

    stage:
        One CRC-protected packet per RVQ stage. A failure at stage l terminates the
        refinement prefix for the whole image. This corresponds to the stage-level theorem.

    group:
        Each stage is split into G non-overlapping spatial packets. Prefix validity is
        tracked independently for each group. This implements the group-wise MI lower-bound
        construction; stage mode is exactly the G=1 special case.

    In both modes, deployment uses hard RVQ + hard QAM detection + CRC prefix decoding.
    During training, the forward value is exactly the hard reconstructed latent while the
    encoder receives an identity straight-through gradient.
    """

    SUPPORTED_MOD_ORDERS = (2, 4, 16, 64, 256)

    def __init__(
        self,
        num_stages: int,
        vq_bitrate_per_stage: int,
        data_dim: int,
        batch_size: int | None = None,
        device: torch.device | str = "cuda",
        beta: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_stages = int(num_stages)
        self.bits_per_index = int(vq_bitrate_per_stage)
        self.K = 1 << self.bits_per_index
        self.D = int(data_dim)
        self.beta = float(beta)

        init = torch.empty(self.num_stages, self.K, self.D).uniform_(-1.0, 1.0)
        self.codebooks = nn.Parameter(init)
        self.register_buffer("codebooks_used", torch.zeros(self.num_stages, self.K, dtype=torch.long))
        self.register_buffer("stage_batches", torch.zeros(self.num_stages, dtype=torch.long))

        self._constellation_names: Dict[int, str] = {}
        for M in self.SUPPORTED_MOD_ORDERS:
            name = f"constellation_{M}"
            self.register_buffer(name, generate_constellation(M), persistent=False)
            self._constellation_names[M] = name

    def constellation(self, M: int) -> torch.Tensor:
        if M not in self._constellation_names:
            raise ValueError(f"Unsupported modulation order {M}; choose from {self.SUPPORTED_MOD_ORDERS}")
        return getattr(self, self._constellation_names[M])

    # ------------------------------------------------------------------ RVQ
    def hard_vq(self, x: torch.Tensor, codebook: torch.Tensor):
        """x [BN,D] -> q [BN,D], residual [BN,D], idx [BN]."""
        dists = (
            x.square().sum(dim=1, keepdim=True)
            - 2.0 * x @ codebook.t()
            + codebook.square().sum(dim=1).unsqueeze(0)
        )
        idx = dists.argmin(dim=1)
        q = codebook[idx]
        return q, x - q, idx

    def rvq_encode(self, z: torch.Tensor, active_stages: int, update_usage: bool = True) -> Dict[str, object]:
        if z.ndim != 3:
            raise ValueError(f"ResUME expects z=[B,N,D], got {tuple(z.shape)}")
        if not 1 <= active_stages <= self.num_stages:
            raise ValueError("active_stages must lie in [1,num_stages]")

        B, N, D = z.shape
        if D != self.D:
            raise ValueError(f"latent D={D}, expected {self.D}")

        residual = z
        q_list: List[torch.Tensor] = []
        idx_list: List[torch.Tensor] = []
        perplexities: List[float] = []
        codebook_loss = z.new_zeros(())
        commitment_loss = z.new_zeros(())

        for l in range(active_stages):
            flat = residual.reshape(B * N, D)
            q_flat, r_flat, idx_flat = self.hard_vq(flat, self.codebooks[l])
            q = q_flat.reshape(B, N, D)
            residual_next = r_flat.reshape(B, N, D)
            idx = idx_flat.reshape(B, N)

            codebook_loss = codebook_loss + F.mse_loss(q, residual.detach())
            commitment_loss = commitment_loss + self.beta * F.mse_loss(residual, q.detach())

            q_list.append(q)
            idx_list.append(idx)
            perplexities.append(self.calculate_perplexity(idx_flat))
            residual = residual_next

            if update_usage:
                with torch.no_grad():
                    self.codebooks_used[l] += torch.bincount(idx_flat, minlength=self.K)
                    self.stage_batches[l] += 1

        return {
            "q_list": q_list,
            "idx_list": idx_list,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexities": perplexities,
            "final_residual": residual,
        }

    # --------------------------------------------------------------- channel
    def _normalize_snr(self, snr_db: torch.Tensor | float, B: int, device) -> torch.Tensor:
        if not torch.is_tensor(snr_db):
            return torch.full((B,), float(snr_db), device=device)
        snr_db = snr_db.to(device=device, dtype=torch.float32).reshape(-1)
        if snr_db.numel() == 1:
            return snr_db.expand(B)
        if snr_db.numel() != B:
            raise ValueError(f"snr_db must have 1 or B={B} elements")
        return snr_db

    def transmit_symbols(
        self,
        tx_symbol_idx: torch.Tensor,
        M: int,
        snr_db: torch.Tensor | float,
        apply_fading: bool = False,
        equalizer: str = "zf",
    ) -> torch.Tensor:
        """Hard digital channel. tx_symbol_idx [P,S] -> detected symbol indices [P,S]."""
        P = tx_symbol_idx.shape[0]
        const = self.constellation(M).to(tx_symbol_idx.device)
        tx = const[tx_symbol_idx.long()]
        snr = self._normalize_snr(snr_db, P, tx.device)
        snr_linear = (10.0 ** (snr / 10.0)).clamp_min(1e-12).reshape(P, 1)

        if apply_fading:
            # One independent flat-fading coefficient per packet.
            h = (
                torch.randn(P, 1, device=tx.device) + 1j * torch.randn(P, 1, device=tx.device)
            ) / math.sqrt(2.0)
        else:
            h = torch.ones(P, 1, dtype=torch.complex64, device=tx.device)

        faded = tx * h
        noise_std = torch.sqrt(1.0 / (2.0 * snr_linear))
        noise = noise_std * (torch.randn_like(faded.real) + 1j * torch.randn_like(faded.real))
        rx = faded + noise

        if equalizer == "zf":
            h_safe = torch.where(h.abs() < 1e-6, h + (1e-6 + 0j), h)
            y = rx / h_safe
        elif equalizer == "mmse":
            noise_var = 1.0 / snr_linear
            coeff = torch.conj(h) / (h.abs().square() + noise_var)
            y = coeff * rx
        else:
            raise ValueError("equalizer must be 'zf' or 'mmse'")

        dist = (y.unsqueeze(-1) - const.reshape(1, 1, -1)).abs()
        return dist.argmin(dim=-1)

    def transmit_packet(
        self,
        tx_indices: torch.Tensor,
        M: int,
        snr_db: torch.Tensor | float,
        apply_fading: bool = False,
        equalizer: str = "zf",
    ):
        """
        tx_indices [P,m], where P is the number of independently faded packets.
        Packet = m*b payload bits + CRC16. Returns rx_indices [P,m], valid [P], symbols/packet.
        """
        if M not in self.SUPPORTED_MOD_ORDERS:
            raise ValueError(f"Unsupported M={M}")
        P, m = tx_indices.shape
        payload = indices_to_payload_bits(tx_indices, self.bits_per_index)
        packet = append_crc16(payload)
        packet_len = packet.shape[1]
        bps = int(math.log2(M))
        tx_symbol_idx, _ = bits_to_symbols(packet, bps)
        rx_symbol_idx = self.transmit_symbols(
            tx_symbol_idx, M=M, snr_db=snr_db, apply_fading=apply_fading, equalizer=equalizer
        )
        rx_packet = symbols_to_bits(rx_symbol_idx, bps, packet_len)
        rx_payload, valid = split_and_check_crc16(rx_packet, payload.shape[1])
        rx_indices = payload_bits_to_indices(rx_payload, self.bits_per_index, m)
        return rx_indices, valid, tx_symbol_idx.shape[1]

    # ------------------------------------------------------------ packet layout
    @staticmethod
    def _resolve_latent_hw(N: int, latent_hw: Tuple[int, int] | None) -> Tuple[int, int]:
        if latent_hw is not None:
            h, w = int(latent_hw[0]), int(latent_hw[1])
            if h * w != N:
                raise ValueError(f"latent_hw={latent_hw} has {h*w} tokens but z has N={N}")
            return h, w
        side = int(round(math.sqrt(N)))
        if side * side != N:
            raise ValueError("latent_hw must be provided for a non-square flattened latent")
        return side, side

    # ------------------------------------------------------------ full forward
    def forward(
        self,
        z: torch.Tensor,
        active_stages: int,
        snr_db: torch.Tensor | float,
        mod_orders: Sequence[int],
        apply_fading: bool = False,
        equalizer: str = "zf",
        use_crc_prefix: bool = True,
        packet_mode: str = "stage",
        latent_hw: Tuple[int, int] | None = None,
        group_hw: Tuple[int, int] = (4, 4),
    ) -> Dict[str, object]:
        if len(mod_orders) < active_stages:
            raise ValueError("mod_orders must contain at least active_stages entries")
        if packet_mode not in {"stage", "group"}:
            raise ValueError("packet_mode must be 'stage' or 'group'")

        rvq = self.rvq_encode(z, active_stages)
        idx_list = rvq["idx_list"]
        B, N, D = z.shape
        Hl, Wl = self._resolve_latent_hw(N, latent_hw)
        group_h, group_w = int(group_hw[0]), int(group_hw[1])
        groups = packet_group_indices(
            packet_mode, Hl, Wl, group_h=group_h, group_w=group_w, device=z.device
        )  # [G,m]
        G, m = groups.shape
        flat_group_ids = groups.reshape(-1)

        z_hard = torch.zeros_like(z)
        prefix_alive = torch.ones(B, G, dtype=torch.bool, device=z.device)
        prefix_len = torch.zeros(B, G, dtype=torch.long, device=z.device)
        rx_indices: List[torch.Tensor] = []
        stage_valid_raw: List[torch.Tensor] = []
        prefix_valid: List[torch.Tensor] = []
        symbol_counts: List[int] = []          # total symbols/image at each stage
        packet_symbol_counts: List[int] = []   # symbols per group packet

        snr_per_image = self._normalize_snr(snr_db, B, z.device)
        snr_per_packet = snr_per_image[:, None].expand(B, G).reshape(B * G)

        for l in range(active_stages):
            M = int(mod_orders[l])
            tx_grouped = idx_list[l][:, groups]                    # [B,G,m]
            tx_packets = tx_grouped.reshape(B * G, m)
            rx_packets, valid_packets, nsym = self.transmit_packet(
                tx_packets,
                M=M,
                snr_db=snr_per_packet,
                apply_fading=apply_fading,
                equalizer=equalizer,
            )
            rx_grouped = rx_packets.reshape(B, G, m)
            valid = valid_packets.reshape(B, G)
            if not use_crc_prefix:
                valid = torch.ones_like(valid)
            prefix_alive = prefix_alive & valid

            # Reassemble full spatial index field in the original token ordering.
            rx_idx_full = torch.empty_like(idx_list[l])
            rx_idx_full[:, flat_group_ids] = rx_grouped.reshape(B, G * m)

            # Decode codewords group-wise and apply each group's own longest-valid-prefix gate.
            rx_q_grouped = self.codebooks[l][rx_grouped.reshape(-1)].reshape(B, G, m, D)
            contribution = rx_q_grouped * prefix_alive[:, :, None, None].to(z.dtype)
            z_hard[:, flat_group_ids, :] = z_hard[:, flat_group_ids, :] + contribution.reshape(B, G * m, D)
            prefix_len = prefix_len + prefix_alive.long()

            rx_indices.append(rx_idx_full)
            stage_valid_raw.append(valid)
            prefix_valid.append(prefix_alive.clone())
            packet_symbol_counts.append(int(nsym))
            symbol_counts.append(int(nsym * G))

        # Expand group-wise prefix depth to every latent token; stage mode is uniform over all tokens.
        depth_per_group = prefix_len.float() / float(active_stages)
        depth_map = z.new_zeros((B, N, 1))
        expanded_depth = depth_per_group[:, :, None].expand(B, G, m).reshape(B, G * m)
        depth_map[:, flat_group_ids, 0] = expanded_depth

        # Deployment-matched STE: hard forward value, identity encoder gradient.
        if self.training:
            z_out = z + (z_hard - z).detach()
        else:
            z_out = z_hard.detach()

        return {
            "z_out": z_out,
            "z_hard": z_hard,
            "depth_map": depth_map.detach(),
            "tx_indices": idx_list,
            "rx_indices": rx_indices,
            "stage_valid_raw": stage_valid_raw,       # each [B,G]
            "prefix_valid": prefix_valid,             # each [B,G]
            "prefix_len": prefix_len,                 # [B,G], G=1 for stage mode
            "symbol_counts": symbol_counts,           # total channel symbols/image by stage
            "packet_symbol_counts": packet_symbol_counts,
            "packet_mode": packet_mode,
            "num_groups": int(G),
            "group_size": int(m),
            "group_hw": (int(group_h), int(group_w)) if packet_mode == "group" else (Hl, Wl),
            "latent_hw": (Hl, Wl),
            "codebook_loss": rvq["codebook_loss"],
            "commitment_loss": rvq["commitment_loss"],
            "perplexities": rvq["perplexities"],
            "codebooks_used": self.codebooks_used.detach().cpu().numpy(),
            "final_residual": rvq["final_residual"],
        }

    @torch.no_grad()
    def encode_indices(self, z: torch.Tensor, active_stages: int | None = None):
        """No-channel RVQ index extraction for the autoregressive information estimator."""
        if active_stages is None:
            active_stages = self.num_stages
        rvq = self.rvq_encode(z, active_stages, update_usage=False)
        return rvq["idx_list"], rvq["q_list"]

    def calculate_perplexity(self, idx: torch.Tensor) -> float:
        counts = torch.bincount(idx, minlength=self.K).float()
        prob = counts / counts.sum().clamp_min(1.0)
        entropy = -(prob * (prob + 1e-12).log()).sum()
        return torch.exp(entropy).item()

    @torch.no_grad()
    def reset_usage_stats(self):
        self.codebooks_used.zero_()
        self.stage_batches.zero_()

    @torch.no_grad()
    def replace_unused_codebooks(self, threshold_fraction: float = 0.01, max_stage: int | None = None):
        if max_stage is None:
            max_stage = self.num_stages
        for l in range(min(max_stage, self.num_stages)):
            usage = self.codebooks_used[l].float()
            total = usage.sum().clamp_min(1.0)
            unused = usage < threshold_fraction * total / self.K
            if int(unused.sum().item()):
                self.codebooks[l, unused].uniform_(-1.0, 1.0)
        self.reset_usage_stats()
