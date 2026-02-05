import os
import io
from PIL import Image
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
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

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(cfg.seed)

def jpeg_recompress(img: torch.Tensor, quality: int) -> torch.Tensor:
    """
    img: torch tensor [3,H,W] in [0,1]
    returns: torch tensor [3,H,W] in [0,1]
    """
    #ensure CPU numpy
    img = img.detach().cpu().clamp(0, 1)

    pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8"), mode ="RGB") 
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    buf.seek(0)

    rec = Image.open(buf).convert("RGB")
    
    return ToTensor()(rec)

# -----------------------------
# Synthetic Tamper Dataset
# -----------------------------
class TamperDataset(Dataset):
    """
    Copy-move forgery:
    - take a patch from the image
    - paste elsewhere
    - produce binary mask
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
        # target image
        tgt, _ = self.base[idx]

        # different source image
        src_idx = random.randrange(len(self.base))
        if src_idx == idx:
            src_idx = (src_idx + 1) % len(self.base)
        src, _ = self.base[src_idx]

        # upscale
        tgt = F.interpolate(
            tgt.unsqueeze(0),
            size=(cfg.image_size, cfg.image_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        src = F.interpolate(
            src.unsqueeze(0),
            size=(cfg.image_size, cfg.image_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        if random.random() < 0.9:
            q1 = random.randint(70, 95)
            tgt = jpeg_recompress(tgt, q1)

        tampered = tgt.clone()
        mask = torch.zeros((1, cfg.image_size, cfg.image_size), dtype=torch.float32)

        ps = random.randint(cfg.patch_min, cfg.patch_max)
        H = W = cfg.image_size

        sx = random.randint(0, W - ps)
        sy = random.randint(0, H - ps)
        dx = random.randint(0, W - ps)
        dy = random.randint(0, H - ps)

        # extract patch FROM SRC (never tampered)
        patch = src[:, sy:sy+ps, sx:sx+ps].clone()

        # intensity jitter
        if random.random() < 0.5:
            patch = torch.clamp(patch * (0.8 + 0.4 * random.random()), 0, 1)

        # feather (safe)
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

        # paste (ALWAYS)
        tampered[:, dy:dy+ps, dx:dx+ps] = patch
        mask[:, dy:dy+ps, dx:dx+ps] = 1.0

        # --- JPEG #2: simulate "export/save JPEG" ---
        if random.random() < 0.9:
            q2 = random.randint(40, 95)
            # (optional) bias q2 different from q1 when both happened
            # if 'q1' in locals() and abs(q2 - q1) < 10: q2 = max(30, min(95, q1 + random.choice([-20, 20])))
            tampered = jpeg_recompress(tampered, q2)


        # JPEG recompress AFTER mask is finalized
        if random.random() < 0.7:
            tampered = jpeg_recompress(tampered, random.randint(60, 95))

        # invariant
        assert mask.sum() > 0

        return tampered, mask

 
# -----------------------------
# U-Net
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

# -----------------------------
# Loss
# -----------------------------
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()

def bce_dice_loss(logits, targets):
    return (
        F.binary_cross_entropy_with_logits(logits, targets)
        + dice_loss(logits, targets)
    )

# -----------------------------
# Visualization
# -----------------------------
@torch.no_grad()
def save_debug(model, batch, tag):
    model.eval()
    imgs, masks = batch
    imgs = imgs.to(cfg.device)
    masks = masks.to(cfg.device)

    logits = model(imgs)
    heat = torch.sigmoid(logits)

    # debug heatmap section
    print("heat stats:",
          "min", heat.min().item(),
          "max", heat.max().item(),
          "mean", heat.mean().item()
          )

    print("mask stats:",
          "mean", masks.float().mean().item()
          )

    n = min(4, imgs.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))

    if n == 1:
        axes = [axes]

    for i in range(n):
        img = imgs[i].cpu().permute(1, 2, 0).numpy()
        gt = masks[i, 0].cpu().numpy()
        h = heat[i, 0].cpu().numpy()

        axes[i][0].imshow(img)
        axes[i][0].set_title("Input")
        axes[i][0].axis("off")

        axes[i][1].imshow(gt, vmin=0, vmax=1)
        axes[i][1].set_title("GT Mask")
        axes[i][1].axis("off")

        axes[i][2].imshow(img)
        axes[i][2].imshow(h, alpha=0.5)
        axes[i][2].set_title("Pred Heatmap")
        axes[i][2].axis("off")

    path = os.path.join(cfg.out_dir, f"debug_{tag}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] {path}")

# -----------------------------
# Training
# -----------------------------
def main():
    ds = TamperDataset(train=True)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    model = UNetSmall().to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        for imgs, masks in dl:
            imgs = imgs.to(cfg.device)
            masks = masks.to(cfg.device)

            logits = model(imgs)
            loss = bce_dice_loss(logits, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 1000 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
                save_debug(model, (imgs.cpu(), masks.cpu()), f"{epoch}_{step}")

            step += 1

        save_debug(model, (imgs.cpu(), masks.cpu()), f"epoch_{epoch}")

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "unet_small.pt"))
    print("Training complete.")

if __name__ == "__main__":
    main()

