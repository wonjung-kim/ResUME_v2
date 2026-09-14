import argparse
import json
import os

import lpips
import numpy as np
import pandas as pd
import torch
import torchvision
from pytorch_msssim import ms_ssim
from tqdm.auto import tqdm

from dataloader import Kodak_Patch_Loader
from modulation_policy import load_policy, lookup_policy
from networks import Digital_SemCom, MobileNet_SemCom, MobileViT_SemCom


def parse_args():
    p = argparse.ArgumentParser("Theory-first ResUME evaluation for stage/group packet policies")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--policy", required=True, help="policy JSON from build_policy.py")
    p.add_argument("--model", choices=["gauss", "mobilenet", "mobilevit"], default="gauss")
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--bits", type=int, default=12)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--norm", action="store_true")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--snrs", type=str, default="-5,0,5,10,15,20")
    p.add_argument("--cbrs", type=str, default="0.00521,0.0104167,0.015625")
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--kodak_root", type=str, default="",
                   help="Folder containing Kodak PNG images. If omitted, RESUME_KODAK_ROOT or dataloader default is used.")
    p.add_argument("--out_dir", type=str, default="./test_output_theory")
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


def psnr_per_image(x, y, eps=1e-8):
    mse = (x - y).square().flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snrs = [float(x) for x in args.snrs.split(",") if x]
    cbrs = [float(x) for x in args.cbrs.split(",") if x]

    policy = load_policy(args.policy)
    meta = policy.get("metadata", {})
    packet_mode = meta.get("packet_mode", "stage")
    group_hw = (int(meta.get("group_h", 4)), int(meta.get("group_w", 4)))
    if meta.get("channel") and meta["channel"] != args.channel:
        raise ValueError(f"Policy was built for channel={meta['channel']}, but --channel={args.channel}")

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})
    use_prefix_map = bool(ckpt_args.get("use_prefix_map", True))
    trained_mode = ckpt_args.get("packet_mode")
    if trained_mode and trained_mode != packet_mode:
        print(f"[Warning] checkpoint trained with packet_mode={trained_mode}, evaluating policy mode={packet_mode}")

    model = make_model(args, device, use_prefix_map=use_prefix_map).to(device)
    model.load_state_dict(ckpt["end_to_end_model"], strict=True)
    model.eval()

    loader_obj = Kodak_Patch_Loader(
        patch_size=(args.img_size, args.img_size), norm=args.norm, base_path=args.kodak_root or None
    )
    test_loader = loader_obj.dataloader(batch_size=args.batch)
    mean, std = loader_obj.norm
    inv_std = [1 / s for s in std]
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_norm = torchvision.transforms.Normalize(inv_mean, inv_std)

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    apply_fading = args.channel == "rayleigh"

    for target_cbr in cbrs:
        for snr_db in snrs:
            entry = lookup_policy(policy, snr_db, target_cbr)
            active_stages = int(entry["active_stages"])
            mod_orders = [int(x) for x in entry["mod_orders"]]
            psnr_all, msssim_all, lpips_all, prefix_all, full_prefix_all = [], [], [], [], []

            desc = f"{packet_mode} CBR={target_cbr:.5f} SNR={snr_db:g} M={mod_orders}"
            for x, _ in tqdm(test_loader, desc=desc):
                x = x.to(device)
                snr = torch.full((x.shape[0],), snr_db, device=device)
                with torch.no_grad():
                    rec, details, _ = model(
                        x,
                        active_stages=active_stages,
                        snr=snr,
                        mod_orders=mod_orders,
                        apply_fading=apply_fading,
                        equalizer="zf",
                        use_crc_prefix=True,
                        packet_mode=packet_mode,
                        group_hw=group_hw,
                        use_prefix_map=use_prefix_map,
                    )
                x01 = inv_norm(x).clamp(0, 1)
                r01 = inv_norm(rec).clamp(0, 1)
                psnr_all += psnr_per_image(r01, x01).cpu().tolist()
                msssim_all += ms_ssim(
                    r01, x01, data_range=1.0, size_average=False, win_size=7
                ).cpu().tolist()
                lp = lpips_fn(r01 * 2 - 1, x01 * 2 - 1).view(-1)
                lpips_all += lp.cpu().tolist()
                pfx = details["prefix_len"].float()
                prefix_all += pfx.reshape(-1).cpu().tolist()
                full_prefix_all += (pfx == active_stages).reshape(-1).float().cpu().tolist()

            theory_value = entry.get("retained_information_bits", entry.get("retained_mi_bits", float("nan")))
            row = {
                "channel": args.channel,
                "packet_mode": packet_mode,
                "num_groups": int(meta.get("num_groups", 1)),
                "group_size": int(meta.get("group_size", meta.get("num_tokens", 0))),
                "group_hw": str(group_hw),
                "target_cbr": target_cbr,
                "actual_cbr": float(entry["cbr"]),
                "snr_db": snr_db,
                "active_stages": active_stages,
                "mod_orders": str(mod_orders),
                "theory_information_objective_bits": float(theory_value),
                "objective_type": meta.get("objective_type", "unknown"),
                "psnr": float(np.mean(psnr_all)),
                "ms_ssim": float(np.mean(msssim_all)),
                "lpips": float(np.mean(lpips_all)),
                "mean_valid_prefix": float(np.mean(prefix_all)),
                "mean_valid_prefix_fraction": float(np.mean(prefix_all) / active_stages),
                "fraction_groups_full_prefix": float(np.mean(full_prefix_all)),
            }
            results.append(row)
            print(row)

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, f"metrics_{packet_mode}_{args.channel}.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(args.out_dir, f"metrics_{packet_mode}_{args.channel}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
