"""
infer.py — Run the trained splice-forgery detector on any image.

Usage:
    python src/infer.py --image path/to/image.jpg
    python src/infer.py --image path/to/image.jpg --model outputs/unet_small.pt
    python src/infer.py --image path/to/image.jpg --threshold 0.4

Outputs (written to the same directory as the input image):
    <name>_heatmap.png  — original image with confidence heatmap overlaid
    <name>_mask.png     — thresholded binary mask

Stdout:
    Max confidence score and predicted tampered area as % of image.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

# Allow running as  python src/infer.py  without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from dataset import CFG
from model import UNetSmall


def load_model(model_path: str, device: str) -> torch.nn.Module:
    model = UNetSmall().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    image_path: str,
    model_path: str = "outputs/unet_small.pt",
    threshold: float = 0.5,
    image_size: int = None,
    device: str = None,
) -> dict:
    """
    Run the splice detector on a single image.

    Returns a dict with keys:
        max_confidence  : float — highest predicted probability in the heatmap
        tampered_pct    : float — predicted tampered area as % of image
        heatmap_path    : str   — path to saved heatmap overlay
        mask_path       : str   — path to saved binary mask
    """
    cfg = CFG()
    if image_size is None:
        image_size = cfg.image_size
    if device is None:
        device = cfg.device

    # --- Load and prepare image ---
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_orig = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_orig.size

    to_tensor = ToTensor()
    img_tensor = to_tensor(pil_orig)  # [3, H, W] in [0,1]

    # Resize to model input size
    img_input = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).to(device)

    # --- Run model ---
    model = load_model(model_path, device)
    logits = model(img_input)
    heatmap = torch.sigmoid(logits)[0, 0].cpu()  # [H, W]

    # --- Upscale outputs back to original resolution ---
    heatmap_full = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    mask_full = (heatmap_full >= threshold).float()

    max_conf = heatmap_full.max().item()
    tampered_pct = mask_full.mean().item() * 100.0

    # --- Save outputs ---
    out_dir = img_path.parent
    stem = img_path.stem

    # Heatmap overlay
    heatmap_path = out_dir / f"{stem}_heatmap.png"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pil_orig)
    ax.imshow(heatmap_full.numpy(), alpha=0.5, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"Splice confidence  |  max={max_conf:.3f}  tampered={tampered_pct:.1f}%")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Binary mask
    mask_path = out_dir / f"{stem}_mask.png"
    mask_img = Image.fromarray((mask_full.numpy() * 255).astype("uint8"), mode="L")
    mask_img.save(mask_path)

    return {
        "max_confidence": max_conf,
        "tampered_pct": tampered_pct,
        "heatmap_path": str(heatmap_path),
        "mask_path": str(mask_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the splice-forgery detector on a single image."
    )
    parser.add_argument("--image", required=True, help="Path to input image (JPEG/PNG)")
    parser.add_argument(
        "--model",
        default="outputs/unet_small.pt",
        help="Path to trained model weights (default: outputs/unet_small.pt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for binary mask (default: 0.5)",
    )
    args = parser.parse_args()

    print(f"Image : {args.image}")
    print(f"Model : {args.model}")
    print(f"Threshold: {args.threshold}")
    print()

    result = run_inference(
        image_path=args.image,
        model_path=args.model,
        threshold=args.threshold,
    )

    print(f"Max confidence : {result['max_confidence']:.4f}")
    print(f"Tampered area  : {result['tampered_pct']:.2f}% of image")
    print()
    print(f"Heatmap saved  : {result['heatmap_path']}")
    print(f"Mask saved     : {result['mask_path']}")


if __name__ == "__main__":
    main()
