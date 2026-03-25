import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TamperDataset
from loss import bce_dice_loss
from model import UNetSmall


# re-export CFG so callers can import from train
from dataset import CFG

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def save_debug(model, batch, tag):
    model.eval()
    imgs, masks = batch
    imgs = imgs.to(cfg.device)
    masks = masks.to(cfg.device)

    logits = model(imgs)
    heat = torch.sigmoid(logits)

    print("heat stats:",
          "min", heat.min().item(),
          "max", heat.max().item(),
          "mean", heat.mean().item())
    print("mask stats:",
          "mean", masks.float().mean().item())

    n = min(4, imgs.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

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


def main():
    set_seed(cfg.seed)

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
    last_batch = None

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

            last_batch = (imgs.cpu(), masks.cpu())

            if step % 1000 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
                save_debug(model, last_batch, f"{epoch}_{step}")

            step += 1

        if last_batch is not None:
            save_debug(model, last_batch, f"epoch_{epoch}")

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "unet_small.pt"))
    print("Training complete.")


if __name__ == "__main__":
    main()
