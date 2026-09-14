import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _resolve_dataset_root(explicit_path, env_var, dataset_name):
    """Resolve a dataset root without any machine-specific hard-coded fallback."""
    root = explicit_path or os.environ.get(env_var, "")
    root = os.path.abspath(os.path.expanduser(root)) if root else ""
    if not root:
        raise ValueError(
            f"{dataset_name} path is not configured. "
            f"Set {env_var} in the shell (recommended) or pass the corresponding CLI path argument."
        )
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{dataset_name} root does not exist or is not a directory: {root}")
    return root


class SlidingWindowTransform:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def __call__(self, img):
        patches = []
        width, height = img.size
        for i in range(0, width, self.step_size):
            for j in range(0, height, self.step_size):
                right = min(i + self.window_size, width)
                bottom = min(j + self.window_size, height)
                left = right - self.window_size
                top = bottom - self.window_size
                patch = img.crop((left, top, right, bottom))
                patches.append((transforms.ToTensor()(patch), (left, top)))
        return patches


class DIV2K_Loader:
    def __init__(self, batch_size, data_dir=None):
        self.data_dir = _resolve_dataset_root(data_dir, "RESUME_DIV2K_ROOT", "DIV2K")
        self.train_dir = os.path.join(self.data_dir, "train")
        self.val_dir = os.path.join(self.data_dir, "val")
        self.batch_size = batch_size
        self.data_transforms = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        }


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class FlatImageDataset(Dataset):
    """Image folder without class subdirectories."""

    def __init__(self, root, transform=None, return_path=False, sort=True):
        self.root = root
        self.transform = transform
        self.return_path = return_path

        paths = []
        for ext in IMG_EXTS:
            paths += glob(os.path.join(root, f"*{ext}"))
            paths += glob(os.path.join(root, f"*{ext.upper()}"))
        if sort:
            paths = sorted(set(paths))

        if len(paths) == 0:
            raise RuntimeError(f"No images found in: {root}")
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        dummy_label = 0
        if self.return_path:
            return img, dummy_label, path
        return img, dummy_label


class ImageNet_Loader:
    def __init__(self, batch_size, size, norm=True, data_dir=None, num_workers=3):
        self.data_dir = _resolve_dataset_root(data_dir, "RESUME_IMAGENET_ROOT", "ImageNet")
        self.train_dir = os.path.join(self.data_dir, "train")
        self.val_dir = os.path.join(self.data_dir, "val")
        for split_name, split_dir in (("train", self.train_dir), ("val", self.val_dir)):
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(
                    f"Expected ImageNet {split_name}/ directory at: {split_dir}\n"
                    "Set RESUME_IMAGENET_ROOT to the parent directory containing train/ and val/."
                )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.norm = (
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
            if norm
            else [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
        self.data_transforms = {
            "train": transforms.Compose([
                transforms.CenterCrop(size * 2),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(self.norm[0], self.norm[1]),
            ]),
            "val": transforms.Compose([
                transforms.CenterCrop(size * 2),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(self.norm[0], self.norm[1]),
            ]),
        }

    def _val_has_class_folders(self):
        for name in os.listdir(self.val_dir):
            p = os.path.join(self.val_dir, name)
            if os.path.isdir(p) and not name.startswith("."):
                return True
        return False

    def dataset_load(self):
        train_dataset = datasets.ImageFolder(self.train_dir, self.data_transforms["train"])
        if self._val_has_class_folders():
            val_dataset = datasets.ImageFolder(self.val_dir, self.data_transforms["val"])
        else:
            val_dataset = FlatImageDataset(self.val_dir, transform=self.data_transforms["val"])
        return {"train": train_dataset, "val": val_dataset}

    def dataloader(self):
        ds = self.dataset_load()
        return {
            "train": DataLoader(
                ds["train"], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            ),
            "val": DataLoader(
                ds["val"], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            ),
        }

    def raw_dataloader(self):
        raw_tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        train_dataset = datasets.ImageFolder(self.train_dir, raw_tf)
        if self._val_has_class_folders():
            val_dataset = datasets.ImageFolder(self.val_dir, raw_tf)
        else:
            val_dataset = FlatImageDataset(self.val_dir, transform=raw_tf)
        return {
            "train": DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            ),
            "val": DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            ),
        }

    def dataloader_1k(self, fixed_val=False):
        ds = self.dataset_load()
        train_dataset, val_dataset = ds["train"], ds["val"]

        num_train, num_val = len(train_dataset), len(val_dataset)
        subset_train_size = max(1, int(0.1 * num_train))
        subset_val_size = max(1, int(0.1 * num_val))

        train_indices = torch.randperm(num_train).tolist()[:subset_train_size]
        if not fixed_val:
            val_indices = torch.randperm(num_val).tolist()[:subset_val_size]
        else:
            val_indices = torch.arange(num_val).tolist()[:subset_val_size]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        return {
            "train": DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            ),
            "val": DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=True,
            ),
        }


class Kodak_Patch_Loader(Dataset):
    def __init__(self, patch_size=(128, 128), norm=True, base_path=None):
        self.base_path = _resolve_dataset_root(base_path, "RESUME_KODAK_ROOT", "Kodak")
        self.image_paths = sorted(glob(os.path.join(self.base_path, "*.png")))
        if not self.image_paths:
            raise RuntimeError(
                f"No Kodak PNG images found in: {self.base_path}\n"
                "Set RESUME_KODAK_ROOT to the directory containing kodim01.png, kodim02.png, ..."
            )
        self.patch_size = patch_size
        self.patches = self.create_patches_from_images()
        self.num_patches_per_image = self.patches.shape[1]
        self.norm = (
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
            if norm
            else [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.norm[0], self.norm[1]),
        ])

    def create_patches_from_images(self):
        all_patches = []
        for image_path in self.image_paths:
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            patch_width, patch_height = self.patch_size

            padded_width = (img_width + patch_width - 1) // patch_width * patch_width
            padded_height = (img_height + patch_height - 1) // patch_height * patch_height

            padded_image = Image.new("RGB", (padded_width, padded_height))
            padded_image.paste(image, (0, 0))

            img_patches = []
            for i in range(0, padded_width, patch_width):
                for j in range(0, padded_height, patch_height):
                    box = (i, j, i + patch_width, j + patch_height)
                    patch = padded_image.crop(box)
                    img_patches.append(np.array(patch))

            all_patches.append(np.array(img_patches))
        return np.stack(all_patches)

    def __len__(self):
        return self.patches.shape[0] * self.patches.shape[1]

    def __getitem__(self, idx):
        image_idx = idx // self.num_patches_per_image
        patch_idx = idx % self.num_patches_per_image
        patch_uint8 = self.patches[image_idx, patch_idx]
        patch = self.transform(patch_uint8)
        return patch, idx

    def dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    loader = Kodak_Patch_Loader().dataloader(batch_size=1)
    print(f"Kodak patches: {len(loader.dataset)}")
