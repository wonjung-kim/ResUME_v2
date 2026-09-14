import argparse
import json
import os
import random
from itertools import combinations_with_replacement

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from dataloader import ImageNet_Loader
from networks import Digital_SemCom, MobileNet_SemCom, MobileViT_SemCom


def parse_args():
    p = argparse.ArgumentParser("Theory-first ResUME training: stage-level or group-wise packets")
    p.add_argument("--model", choices=["gauss", "mobilenet", "mobilevit"], default="gauss")
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--bits", type=int, default=12)
    p.add_argument("--batch", type=int, default=36)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--snr_min", type=float, default=0.0)
    p.add_argument("--snr_max", type=float, default=10.0)
    p.add_argument("--fading", action="store_true")
    p.add_argument("--norm", action="store_true")
    p.add_argument("--lambda_cb", type=float, default=1.0)
    p.add_argument("--lambda_commit", type=float, default=1.0)
    p.add_argument("--mod_orders", type=str, default="2,4,16,64,256")
    p.add_argument("--fixed_profile", type=str, default="",
                   help="Optional ascending profile, e.g. 2,4,16,64")
    p.add_argument("--packet_mode", choices=["stage", "group"], default="stage")
    p.add_argument("--group_h", type=int, default=4, help="Group-mode latent block height")
    p.add_argument("--group_w", type=int, default=4, help="Group-mode latent block width")
    p.add_argument("--use_prefix_map", action=argparse.BooleanOptionalAction, default=True,
                   help="Condition decoder on receiver-known normalized prefix depth")
    p.add_argument("--use_crc_prefix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--imagenet_root", type=str, default="",
                   help="ImageNet root containing train/ and val/. If omitted, RESUME_IMAGENET_ROOT or dataloader default is used.")
    p.add_argument("--out_dir", type=str, default="./output_theory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replace_every", type=int, default=0,
                   help="If >0, replace low-usage codewords every N epochs")
    return p.parse_args()


def make_model(args, device):
    common = dict(
        num_stages=args.stages,
        vq_bitrate_per_stage=args.bits,
        embedding_dim=128,
        batch_size=args.batch,
        device=device,
        use_prefix_map=args.use_prefix_map,
    )
    if args.model == "gauss":
        return Digital_SemCom(
            num_hiddens=128,
            num_residual_hiddens=64,
            num_residual_layers=4,
            **common,
        )
    if args.model == "mobilenet":
        return MobileNet_SemCom(**common)
    return MobileViT_SemCom(**common)


def choose_profile(active_stages, allowed, fixed_profile=None):
    if fixed_profile:
        if len(fixed_profile) < active_stages:
            raise ValueError("fixed_profile shorter than active_stages")
        profile = fixed_profile[:active_stages]
        if any(profile[i] > profile[i + 1] for i in range(len(profile) - 1)):
            raise ValueError("Theory-first fixed_profile must be ascending in modulation order")
        return profile
    # Reliability-order theorem permits restricting training profiles to ascending QAM orders.
    return list(random.choice(list(combinations_with_replacement(allowed, active_stages))))


def validate(model, loader, device, args):
    model.eval()
    losses = []
    profile = [4] * args.stages
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= 20:
                break
            x = x.to(device)
            B = x.shape[0]
            snr = torch.full((B,), 10.0, device=device)
            rec, details, _ = model(
                x,
                active_stages=args.stages,
                snr=snr,
                mod_orders=profile,
                apply_fading=args.fading,
                equalizer="zf",
                use_crc_prefix=args.use_crc_prefix,
                packet_mode=args.packet_mode,
                group_hw=(args.group_h, args.group_w),
                use_prefix_map=args.use_prefix_map,
            )
            losses.append(F.mse_loss(rec, x).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allowed = sorted({int(x) for x in args.mod_orders.split(",") if x})
    fixed = [int(x) for x in args.fixed_profile.split(",") if x] if args.fixed_profile else None

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    loader = ImageNet_Loader(args.batch, 128, norm=args.norm, data_dir=args.imagenet_root or None)
    train_loader = loader.dataloader_1k()["train"]
    val_loader = loader.dataloader_1k()["val"]

    model = make_model(args, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        loss_acc = rec_acc = cb_acc = cmt_acc = prefix_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=130)
        for x, _ in pbar:
            x = x.to(device)
            B = x.shape[0]

            if epoch < args.warmup_epochs:
                # Full-depth, effectively noiseless warm-up so every RVQ codebook receives signal.
                active_stages = args.stages
                profile = [4] * active_stages
                snr_scalar = 40.0
                fading = False
            else:
                active_stages = random.randint(1, args.stages)
                profile = choose_profile(active_stages, allowed, fixed)
                snr_scalar = random.uniform(args.snr_min, args.snr_max)
                fading = args.fading

            snr = torch.full((B,), float(snr_scalar), device=device)
            optimizer.zero_grad(set_to_none=True)
            rec, details, _ = model(
                x,
                active_stages=active_stages,
                snr=snr,
                mod_orders=profile,
                apply_fading=fading,
                equalizer="zf",
                use_crc_prefix=args.use_crc_prefix,
                packet_mode=args.packet_mode,
                group_hw=(args.group_h, args.group_w),
                use_prefix_map=args.use_prefix_map,
            )

            loss_rec = F.mse_loss(rec, x)
            loss = (
                loss_rec
                + args.lambda_cb * details["codebook_loss"]
                + args.lambda_commit * details["commitment_loss"]
            )
            loss.backward()
            optimizer.step()

            loss_acc += loss.item()
            rec_acc += loss_rec.item()
            cb_acc += details["codebook_loss"].item()
            cmt_acc += details["commitment_loss"].item()
            mean_prefix = details["prefix_len"].float().mean().item()
            prefix_acc += mean_prefix
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", rec=f"{loss_rec.item():.4f}",
                mode=args.packet_mode, L=active_stages, M=str(profile), snr=f"{snr_scalar:.1f}",
                prefix=f"{mean_prefix:.2f}", G=details["num_groups"],
            )

        num_batches = max(1, len(train_loader))
        val = validate(model, val_loader, device, args)
        stats = {
            "epoch": epoch + 1,
            "packet_mode": args.packet_mode,
            "group_hw": [args.group_h, args.group_w],
            "loss": loss_acc / num_batches,
            "rec": rec_acc / num_batches,
            "codebook": cb_acc / num_batches,
            "commitment": cmt_acc / num_batches,
            "mean_valid_prefix": prefix_acc / num_batches,
            "val_mse": val,
        }
        print(stats)
        with open(os.path.join(args.out_dir, "history.jsonl"), "a") as f:
            f.write(json.dumps(stats) + "\n")

        ckpt = {
            "epoch": epoch + 1,
            "end_to_end_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))
        if val < best_val:
            best_val = val
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

        if args.replace_every > 0 and (epoch + 1) % args.replace_every == 0:
            model.resume.replace_unused_codebooks(max_stage=args.stages)


if __name__ == "__main__":
    main()
