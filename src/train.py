import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CFG, TamperDataset
from loss import bce_dice_loss
from model import UNetSmall

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_debug(model: torch.nn.Module, batch: tuple, tag: str) -> None:
    model.eval()
    imgs, masks = batch
    imgs = imgs.to(cfg.device)
    masks = masks.to(cfg.device)

    logits = model(imgs)
    heat = torch.sigmoid(logits)

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
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader: DataLoader, threshold: float = 0.5) -> dict:
    """
    Runs the model over the validation split and returns:
      - pixel_iou    : Intersection over Union at the given threshold
      - f1           : F1 / Dice score on the binary mask
      - pixel_acc    : Fraction of correctly classified pixels
    """
    model.eval()
    total_iou = 0.0
    total_f1 = 0.0
    total_acc = 0.0
    n_batches = 0

    for imgs, masks in tqdm(val_loader, desc="  Evaluating", leave=False):
        imgs = imgs.to(cfg.device)
        masks = masks.to(cfg.device)

        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        # Pixel accuracy
        total_acc += (preds == masks).float().mean().item()

        # IoU per sample, then batch mean
        intersection = (preds * masks).sum(dim=(1, 2, 3))
        union = (preds + masks).clamp(0, 1).sum(dim=(1, 2, 3))
        iou = (intersection / (union + 1e-6)).mean().item()
        total_iou += iou

        # F1 / Dice per sample, then batch mean
        tp = (preds * masks).sum(dim=(1, 2, 3))
        fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
        fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))
        f1 = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean().item()
        total_f1 += f1

        n_batches += 1

    return {
        "pixel_iou": total_iou / max(n_batches, 1),
        "f1": total_f1 / max(n_batches, 1),
        "pixel_acc": total_acc / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(cfg.seed)

    print(f"Device: {cfg.device}")
    print(f"Image size: {cfg.image_size}x{cfg.image_size}")

    train_ds = TamperDataset(split="train", cfg=cfg)
    val_ds = TamperDataset(split="val", cfg=cfg)

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = UNetSmall().to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=1e-5
    )

    step = 0
    last_batch = None
    log_every = max(1, len(train_dl) // 4)  # ~4 debug saves per epoch

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for imgs, masks in pbar:
            imgs = imgs.to(cfg.device)
            masks = masks.to(cfg.device)

            logits = model(imgs)
            loss = bce_dice_loss(logits, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            last_batch = (imgs.cpu(), masks.cpu())

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step % log_every == 0:
                save_debug(model, last_batch, f"{epoch}_{step}")

            step += 1

        scheduler.step()
        avg_loss = epoch_loss / len(train_dl)

        # Epoch-end debug save (guard against empty dataloader)
        if last_batch is not None:
            save_debug(model, last_batch, f"epoch_{epoch}")

        # Evaluation metrics
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}  avg_loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
        metrics = evaluate(model, val_dl)
        print(
            f"  Val  |  IoU={metrics['pixel_iou']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"Acc={metrics['pixel_acc']:.4f}"
        )

        model.train()

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "unet_small.pt"))
    print("\nTraining complete.")
    print(f"Model saved to {cfg.out_dir}/unet_small.pt")


if __name__ == "__main__":
    main()
