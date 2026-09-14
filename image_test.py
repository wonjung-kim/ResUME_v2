import argparse
import os
from PIL import Image, UnidentifiedImageError


def main():
    p = argparse.ArgumentParser("Find unreadable image files")
    p.add_argument("--root", default=os.environ.get("RESUME_IMAGENET_ROOT", "/mnt/data/wonjung/datasets/ImageNet/train"))
    p.add_argument("--out", default="./bad_imagenet_files.txt")
    args = p.parse_args()

    bad = []
    for dirpath, _, filenames in os.walk(args.root):
        for fn in filenames:
            if not fn.lower().endswith((".jpeg", ".jpg", ".png", ".bmp", ".webp")):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.convert("RGB").load()
            except (UnidentifiedImageError, OSError, ValueError):
                bad.append(path)

    with open(args.out, "w") as f:
        for path in bad:
            f.write(path + "\n")
    print(f"bad images: {len(bad)}, saved to {args.out}")


if __name__ == "__main__":
    main()
