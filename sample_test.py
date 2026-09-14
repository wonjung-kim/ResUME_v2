import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from networks import Digital_SemCom


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class SampleFolderDataset(Dataset):
    IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root: str, img_size: int = 128, norm: bool = True):
        self.paths = [
            os.path.join(root, f) for f in sorted(os.listdir(root)) if f.lower().endswith(self.IMG_EXT)
        ]
        if not self.paths:
            raise RuntimeError(f"No image files found in {root}")
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        ops = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        if norm:
            ops.append(transforms.Normalize(self.mean, self.std))
        self.transform = transforms.Compose(ops)
        self.norm = norm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, os.path.splitext(os.path.basename(path))[0]


def compute_psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y, reduction="none").reshape(x.size(0), -1).mean(dim=1)
    return 10 * torch.log10(1.0 / (mse + eps))


def main():
    p = argparse.ArgumentParser("Quick sample-folder test for stage/group theory-first ResUME")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--mod_orders", type=str, default="2,4,16,64")
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--bits", type=int, default=12)
    p.add_argument("--active_stages", type=int, default=4)
    p.add_argument("--packet_mode", choices=["stage", "group"], default="stage")
    p.add_argument("--group_h", type=int, default=4)
    p.add_argument("--group_w", type=int, default=4)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--out_dir", default="./sample_test_out")
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--no_norm", action="store_true")
    p.add_argument("--snrs", default="0,5,10,15,20")
    args = p.parse_args()

    snrs = [float(s) for s in args.snrs.split(",") if s]
    profile = [int(x) for x in args.mod_orders.split(",") if x][: args.active_stages]
    if len(profile) < args.active_stages:
        raise ValueError("--mod_orders must contain at least --active_stages values")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    use_prefix_map = bool(ckpt.get("args", {}).get("use_prefix_map", True))
    model = Digital_SemCom(
        num_hiddens=128,
        num_residual_hiddens=64,
        num_residual_layers=4,
        num_stages=args.stages,
        vq_bitrate_per_stage=args.bits,
        embedding_dim=128,
        batch_size=args.batch,
        device=device,
        use_prefix_map=use_prefix_map,
    ).to(device)
    model.load_state_dict(ckpt["end_to_end_model"], strict=True)
    model.eval()

    norm_flag = not args.no_norm
    dataset = SampleFolderDataset(args.sample_dir, args.img_size, norm=norm_flag)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    if norm_flag:
        inv_std = [1.0 / s for s in dataset.std]
        inv_mean = [-m / s for m, s in zip(dataset.mean, dataset.std)]
        inv_norm = transforms.Normalize(inv_mean, inv_std)
    else:
        inv_norm = lambda x: x

    for ch_name, fading in [("awgn", False), ("rayleigh", True)]:
        for snr_db in snrs:
            out_dir = os.path.join(args.out_dir, args.packet_mode, ch_name, f"snr{snr_db:g}")
            os.makedirs(out_dir, exist_ok=True)
            vals = []
            for x, names in tqdm(loader, desc=f"{args.packet_mode}/{ch_name}/{snr_db:g}dB"):
                x = x.to(device)
                snr = torch.full((x.shape[0],), snr_db, device=device)
                with torch.no_grad():
                    rec, details, _ = model(
                        x,
                        active_stages=args.active_stages,
                        snr=snr,
                        mod_orders=profile,
                        apply_fading=fading,
                        packet_mode=args.packet_mode,
                        group_hw=(args.group_h, args.group_w),
                        use_prefix_map=use_prefix_map,
                    )
                x01, r01 = inv_norm(x).clamp(0, 1), inv_norm(rec).clamp(0, 1)
                vals += compute_psnr(r01, x01).cpu().tolist()
                for j, name in enumerate(names):
                    save_image(r01[j], os.path.join(out_dir, f"{name}.png"))
            print(f"{args.packet_mode} {ch_name} {snr_db:g}dB: PSNR={np.mean(vals):.2f} dB")


if __name__ == "__main__":
    main()
