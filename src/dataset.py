import io
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


@dataclass
class CFG:
    seed: int = 42
    image_size: int = 128
    patch_min: int = 20
    patch_max: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 3
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "outputs"


def jpeg_recompress(img: torch.Tensor, quality: int) -> torch.Tensor:
    """
    img: torch tensor [3,H,W] in [0,1]
    returns: torch tensor [3,H,W] in [0,1]
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


class TamperDataset(Dataset):
    """
    Splice forgery dataset:
    - take a patch from a different source image
    - paste it onto the target image
    - produce binary mask of the pasted region
    """
    def __init__(self, train=True):
        self.base = CIFAR10(
            root="datasets",
            train=train,
            download=True,
            transform=ToTensor()
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        tgt, _ = self.base[idx]

        src_idx = random.randrange(len(self.base))
        if src_idx == idx:
            src_idx = (src_idx + 1) % len(self.base)
        src, _ = self.base[src_idx]

        tgt = F.interpolate(
            tgt.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        src = F.interpolate(
            src.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        if random.random() < 0.9:
            q1 = random.randint(70, 95)
            tgt = jpeg_recompress(tgt, q1)

        tampered = tgt.clone()
        mask = torch.zeros((1, 128, 128), dtype=torch.float32)

        ps = random.randint(20, 60)
        H = W = 128

        sx = random.randint(0, W - ps)
        sy = random.randint(0, H - ps)
        dx = random.randint(0, W - ps)
        dy = random.randint(0, H - ps)

        patch = src[:, sy:sy+ps, sx:sx+ps].clone()

        if random.random() < 0.5:
            patch = torch.clamp(patch * (0.8 + 0.4 * random.random()), 0, 1)

        if random.random() < 0.7:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, ps),
                torch.linspace(-1, 1, ps),
                indexing="ij"
            )
            r = torch.sqrt(xx**2 + yy**2)
            alpha = torch.clamp(1.0 - (r - 0.6) * 3.0, 0.0, 1.0).unsqueeze(0)
            base = tampered[:, dy:dy+ps, dx:dx+ps]
            patch = patch * alpha + base * (1 - alpha)

        tampered[:, dy:dy+ps, dx:dx+ps] = patch
        mask[:, dy:dy+ps, dx:dx+ps] = 1.0

        if random.random() < 0.9:
            q2 = random.randint(40, 95)
            tampered = jpeg_recompress(tampered, q2)

        if random.random() < 0.7:
            tampered = jpeg_recompress(tampered, random.randint(60, 95))

        assert mask.sum() > 0

        return tampered, mask
