import io
import os
import random
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


@dataclass
class CFG:
    seed: int = 42
    image_size: int = 256
    patch_min: int = 40
    patch_max: int = 110
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 10
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "outputs"
    coco_train_dir: str = "datasets/coco/train2017"
    coco_val_dir: str = "datasets/coco/val2017"


# ---------------------------------------------------------------------------
# JPEG helpers
# ---------------------------------------------------------------------------

def jpeg_recompress(img: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Round-trip a [3,H,W] float tensor in [0,1] through JPEG compression.
    Returns a [3,H,W] float tensor in [0,1].
    """
    img = img.detach().cpu().clamp(0, 1)
    pil = Image.fromarray(
        (img.permute(1, 2, 0).numpy() * 255).astype("uint8"), mode="RGB"
    )
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    buf.seek(0)
    rec = Image.open(buf).convert("RGB")
    return ToTensor()(rec)


# ---------------------------------------------------------------------------
# Augmentation pipeline (applied before splice)
# ---------------------------------------------------------------------------

_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
])


def _augment(img_tensor: torch.Tensor) -> torch.Tensor:
    """Apply albumentations augmentations to a [3,H,W] float tensor in [0,1]."""
    np_img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    augmented = _AUG(image=np_img)["image"]
    return torch.from_numpy(augmented).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# COCO image loader (images only — no annotations needed)
# ---------------------------------------------------------------------------

class COCOImageFolder(Dataset):
    """
    Loads raw JPEG images from a COCO image directory.
    No annotations required — we generate synthetic splice forgeries on-the-fly.

    Expects images at:
        datasets/coco/train2017/*.jpg
        datasets/coco/val2017/*.jpg

    Run scripts/download_coco.sh to download the dataset before training.
    """

    def __init__(self, img_dir: str, image_size: int):
        path = Path(img_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"\nCOCO image directory not found: {img_dir}\n"
                f"Run  bash scripts/download_coco.sh  to download the dataset first.\n"
            )
        self.paths = sorted(path.glob("*.jpg"))
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"\nNo .jpg files found in {img_dir}\n"
                f"Run  bash scripts/download_coco.sh  to download the dataset first.\n"
            )
        self.image_size = image_size
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        tensor = self.to_tensor(img)  # [3, H, W] in [0,1]
        # Resize to model input size
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return tensor


# ---------------------------------------------------------------------------
# Splice forgery dataset
# ---------------------------------------------------------------------------

class TamperDataset(Dataset):
    """
    Splice forgery dataset built on top of COCO images.

    For each sample:
      - Loads a target image from COCO
      - Loads a different source image from COCO
      - Extracts a random rectangular patch from the source
      - Pastes it at a random location on the target
      - Returns (tampered_image [3,H,W], binary_mask [1,H,W])

    JPEG compression is applied in two realistic passes:
      1. Pre-paste (50% chance, quality 75-95): simulates the source image
         having been previously saved as a JPEG.
      2. Post-paste (80% chance, quality 70-95): simulates saving the
         forged composite as a JPEG.

    Albumentations augmentations (flips, rotation, colour jitter) are applied
    to the target image before splicing.
    """

    def __init__(self, split: str = "train", cfg: CFG = None):
        if cfg is None:
            cfg = CFG()
        self.cfg = cfg
        img_dir = cfg.coco_train_dir if split == "train" else cfg.coco_val_dir
        self.images = COCOImageFolder(img_dir, cfg.image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        cfg = self.cfg
        H = W = cfg.image_size

        # --- Target image ---
        tgt = self.images[idx]
        tgt = _augment(tgt)

        # --- Source image (different from target) ---
        src_idx = random.randrange(len(self.images))
        if src_idx == idx:
            src_idx = (src_idx + 1) % len(self.images)
        src = self.images[src_idx]

        # --- Pre-paste JPEG compression (simulates source previously saved as JPEG) ---
        if random.random() < 0.5:
            tgt = jpeg_recompress(tgt, random.randint(75, 95))

        tampered = tgt.clone()
        mask = torch.zeros((1, H, W), dtype=torch.float32)

        # --- Random splice region ---
        ps = random.randint(cfg.patch_min, cfg.patch_max)
        sx = random.randint(0, W - ps)
        sy = random.randint(0, H - ps)
        dx = random.randint(0, W - ps)
        dy = random.randint(0, H - ps)

        patch = src[:, sy:sy + ps, sx:sx + ps].clone()

        # Intensity jitter on the patch
        if random.random() < 0.5:
            patch = torch.clamp(patch * (0.8 + 0.4 * random.random()), 0, 1)

        # Soft feathering at patch edges
        if random.random() < 0.7:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, ps),
                torch.linspace(-1, 1, ps),
                indexing="ij",
            )
            r = torch.sqrt(xx ** 2 + yy ** 2)
            alpha = torch.clamp(1.0 - (r - 0.6) * 3.0, 0.0, 1.0).unsqueeze(0)
            base = tampered[:, dy:dy + ps, dx:dx + ps]
            patch = patch * alpha + base * (1 - alpha)

        # Paste patch onto target
        tampered[:, dy:dy + ps, dx:dx + ps] = patch
        mask[:, dy:dy + ps, dx:dx + ps] = 1.0

        # --- Post-paste JPEG compression (simulates saving the forged image) ---
        if random.random() < 0.8:
            tampered = jpeg_recompress(tampered, random.randint(70, 95))

        assert mask.sum() > 0, "mask must contain at least one positive pixel"

        return tampered, mask
