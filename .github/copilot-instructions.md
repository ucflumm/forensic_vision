# Copilot Instructions

## Project Overview

forensic-vision is a multi-pronged toolkit for detecting digital image authenticity issues. It is being built as four independent detection pillars:

1. **Image Manipulation Detection** *(In Progress)* — splice forgery detection via U-Net segmentation
2. **AI Generation Detection** *(Planned)*
3. **Metadata Analysis** *(Planned)*
4. **Steganography / Hidden Embedding** *(Planned)*

## Pillar 1: Image Manipulation Detection

Detects splice forgery (a region from one photograph composited into another — the core of most photoshopped images) using a U-Net trained on synthetically tampered COCO 2017 images generated on-the-fly.

## Running the Project

**Download COCO images first:**
```bash
bash scripts/download_coco.sh --val-only   # ~788 MB, enough to test
bash scripts/download_coco.sh              # full dataset ~18 GB
```

**Train locally (venv):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

**Train in Docker (recommended):**
```bash
docker build -t forensic-vision .
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/datasets:/app/datasets" \
  forensic-vision
```

**Run inference on any image:**
```bash
python src/infer.py --image path/to/image.jpg --model outputs/unet_small.pt
```

Datasets download to `datasets/` (gitignored). Outputs (debug PNGs, weights) go to `outputs/`.

## Architecture

### File structure
```
src/
  dataset.py   — CFG dataclass, COCOImageFolder, TamperDataset, jpeg_recompress, augmentations
  model.py     — ConvBlock (Conv→BN→ReLU×2), UNetSmall (3-level encoder/decoder)
  loss.py      — dice_loss, bce_dice_loss
  train.py     — main(), evaluate(), save_debug(), set_seed()
  infer.py     — CLI inference script, run_inference() function
scripts/
  download_coco.sh   — downloads COCO 2017 train2017 and val2017
```

### Key components

- **`CFG` dataclass** (`src/dataset.py`) — all hyperparameters. Change here first. `image_size=256`, `epochs=10`, `device` auto-detects CUDA.
- **`TamperDataset`** (`src/dataset.py`) — wraps COCO 2017; generates splice forgeries on-the-fly. Fails fast with a clear error if COCO is not downloaded.
- **`UNetSmall`** (`src/model.py`) — 3-level encoder/decoder with skip connections and BatchNorm. Channels: 3→64→128→256→128→64→1. Output is raw logits.
- **`bce_dice_loss`** (`src/loss.py`) — combined BCE + Dice loss for class imbalance.
- **`evaluate()`** (`src/train.py`) — runs on val split after each epoch; reports pixel IoU, F1/Dice, pixel accuracy.
- **`run_inference()`** (`src/infer.py`) — loads saved weights, runs on any image, returns heatmap and binary mask at original resolution.

## Key Conventions

- The model outputs **raw logits**; apply `torch.sigmoid()` before thresholding for inference.
- `cfg.device` auto-detects CUDA. For GPU, also swap to a CUDA-enabled PyTorch wheel in `requirements.txt`.
- `datasets/` is gitignored (COCO downloaded at setup time). `outputs/` is not gitignored — debug PNGs and `unet_small.pt` land there.
- `num_workers=0` is intentional to avoid multiprocessing issues in Docker.
- JPEG compression uses **2 passes only**: pre-paste (simulates source previously saved) and post-paste (simulates saving the forged image). Do not add a third pass — it destroys JPEG DCT grid inconsistency signals.
- Albumentations augmentations are applied **before** the splice operation (to the target image only).
- The `.venv/` directory is partially gitignored (only lib/bin/include excluded). Avoid committing venv files.
