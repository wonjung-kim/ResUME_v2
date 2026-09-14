import argparse
import json
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from dataloader import ImageNet_Loader
from entropy_model import (
    CausalIndexEntropyModel,
    build_packet_layout,
    indices_to_sequence,
    packet_matrix,
    packet_nll_per_sample,
    token_nll_bits,
)
from networks import Digital_SemCom, MobileNet_SemCom, MobileViT_SemCom


def parse_args():
    p = argparse.ArgumentParser("Estimate theory-first RVQ packet information increments")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", choices=["gauss", "mobilenet", "mobilevit"], default="gauss")
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--bits", type=int, default=12)
    p.add_argument("--batch", type=int, default=8, help="Use a smaller batch because the AR logits are large")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--norm", action="store_true")
    p.add_argument("--snr", type=float, default=10.0,
                   help="Only controls the existing NoiseFusion encoder gating during index extraction")
    p.add_argument("--packet_mode", choices=["stage", "group"], default="stage")
    p.add_argument("--group_h", type=int, default=4)
    p.add_argument("--group_w", type=int, default=4)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_train_batches", type=int, default=0, help="0 means full loader")
    p.add_argument("--max_val_batches", type=int, default=0)
    p.add_argument("--imagenet_root", type=str, default="",
                   help="ImageNet root containing train/ and val/. If omitted, RESUME_IMAGENET_ROOT from the shell is used; there is no hard-coded fallback.")
    p.add_argument("--out", type=str, default="./information_estimate.json")
    p.add_argument("--save_entropy_model", type=str, default="")
    return p.parse_args()


def make_model(args, device, use_prefix_map=True):
    common = dict(
        num_stages=args.stages,
        vq_bitrate_per_stage=args.bits,
        embedding_dim=128,
        batch_size=args.batch,
        device=device,
        use_prefix_map=use_prefix_map,
    )
    if args.model == "gauss":
        return Digital_SemCom(128, 4, 64, **common)
    if args.model == "mobilenet":
        return MobileNet_SemCom(**common)
    return MobileViT_SemCom(**common)


@torch.no_grad()
def extract_sequence(base, x, snr, layout):
    idx_list, _, _ = base.extract_rvq_indices(x, snr, active_stages=layout.num_stages)
    return indices_to_sequence(idx_list, layout)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})
    use_prefix_map = bool(ckpt_args.get("use_prefix_map", True))
    base = make_model(args, device, use_prefix_map=use_prefix_map).to(device)
    base.load_state_dict(ckpt["end_to_end_model"], strict=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    loader = ImageNet_Loader(args.batch, 128, norm=args.norm, data_dir=args.imagenet_root or None)
    train_loader = loader.dataloader_1k()["train"]
    val_loader = loader.dataloader_1k()["val"]

    # Infer latent geometry from the actual trained backbone.
    x0, _ = next(iter(train_loader))
    x0 = x0.to(device)
    snr0 = torch.full((x0.shape[0],), args.snr, device=device)
    with torch.no_grad():
        _, origin_shape, _ = base.encode_latent(x0, snr0)
    _, Hl, Wl, _ = origin_shape
    layout = build_packet_layout(
        args.stages,
        Hl,
        Wl,
        args.packet_mode,
        group_hw=(args.group_h, args.group_w),
        device=device,
    )

    entropy_model = CausalIndexEntropyModel(
        num_codes=1 << args.bits,
        num_stages=args.stages,
        num_tokens=Hl * Wl,
        max_seq_len=layout.seq_len,
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.AdamW(entropy_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        entropy_model.train()
        total_bits = 0.0
        total_tokens = 0
        for bi, (x, _) in enumerate(tqdm(train_loader, desc=f"Entropy epoch {epoch+1}/{args.epochs}")):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            x = x.to(device)
            snr = torch.full((x.shape[0],), args.snr, device=device)
            targets = extract_sequence(base, x, snr, layout)
            optimizer.zero_grad(set_to_none=True)
            logits = entropy_model(targets, layout.stage_ids, layout.token_ids)
            B, S, K = logits.shape
            loss_nats = F.cross_entropy(logits.reshape(B * S, K), targets.reshape(B * S))
            loss_nats.backward()
            optimizer.step()
            total_bits += (loss_nats.item() / 0.6931471805599453) * B * S
            total_tokens += B * S
        print(f"train CE: {total_bits / max(1,total_tokens):.4f} bits/index")

    entropy_model.eval()
    packet_sum = torch.zeros(layout.num_packets, dtype=torch.float64, device=device)
    n_images = 0
    token_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for bi, (x, _) in enumerate(tqdm(val_loader, desc="Estimating packet information increments")):
            if args.max_val_batches and bi >= args.max_val_batches:
                break
            x = x.to(device)
            snr = torch.full((x.shape[0],), args.snr, device=device)
            targets = extract_sequence(base, x, snr, layout)
            logits = entropy_model(targets, layout.stage_ids, layout.token_ids)
            nll = token_nll_bits(logits, targets)
            per_packet = packet_nll_per_sample(nll, layout)
            packet_sum += per_packet.double().sum(dim=0)
            n_images += x.shape[0]
            token_sum += nll.sum().item()
            token_count += nll.numel()

    packet_bits = (packet_sum / max(1, n_images)).float()
    delta_matrix = packet_matrix(packet_bits, layout)  # [L,G]
    delta_stage = delta_matrix.sum(dim=1)

    if args.packet_mode == "stage":
        theory_statement = (
            "delta_l = H(A_l|A_<l) = I(Z;A_l|A_<l) for deterministic RVQ; "
            "used in the exact retained-prefix MI expression"
        )
        objective_type = "stage_exact_prefix_mi"
    else:
        theory_statement = (
            "delta_g_l = H(A_g_l|A_prec) estimated under the fixed stage-major packet order; "
            "sum delta_g_l*P(T_g>=l) is a rigorous retained-MI lower bound"
        )
        objective_type = "group_retained_mi_lower_bound"

    out = {
        "definition": theory_statement,
        "estimator": "causal autoregressive cross entropy; converges to conditional entropy with an exact model",
        "objective_type": objective_type,
        "packet_mode": args.packet_mode,
        "latent_h": int(Hl),
        "latent_w": int(Wl),
        "num_tokens": int(Hl * Wl),
        "num_groups": int(layout.num_groups),
        "group_size": int(layout.group_size),
        "group_h": int(layout.group_h),
        "group_w": int(layout.group_w),
        "num_stages": int(args.stages),
        "bits_per_index": int(args.bits),
        "delta_packet_bits": delta_matrix.detach().cpu().tolist(),
        "delta_stage_bits": delta_stage.detach().cpu().tolist(),
        # compatibility alias used by some earlier scripts
        "delta_bits_per_image": delta_stage.detach().cpu().tolist(),
        "validation_cross_entropy_bits_per_index": token_sum / max(1, token_count),
        "packet_order": "stage-major; groups ordered top-left to bottom-right; tokens raster-ordered within group",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    if args.save_entropy_model:
        os.makedirs(os.path.dirname(args.save_entropy_model) or ".", exist_ok=True)
        torch.save({"state_dict": entropy_model.state_dict(), "args": vars(args), "info": out}, args.save_entropy_model)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
