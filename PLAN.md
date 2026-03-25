# Pillar 1: Image Manipulation Detection — Implementation Plan

## Overarching Goal

forensic-vision is a multi-pronged toolkit for detecting digital image authenticity issues. This plan covers **Pillar 1: Image Manipulation Detection** — specifically splice forgery, where a region from one image is composited into another (the core of most "photoshopped" images).

The end goal is a model that generalises to real-world manipulated images, not just synthetic CIFAR-10 test cases.

---

## Branch

`feature/pillar-1-image-manipulation`

---

## Commit Structure

| # | Commit message | Contents |
|---|---|---|
| 1 | `refactor: split single-file into src modules` | File structure only, zero logic changes |
| 2 | `feat: swap CIFAR-10 for COCO, fix JPEG pipeline, add augmentations` | `src/dataset.py` |
| 3 | `feat: add BatchNorm and increased capacity to UNetSmall` | `src/model.py` |
| 4 | `feat: add evaluation metrics and LR scheduler` | `src/train.py` |
| 5 | `feat: add inference script` | `src/infer.py` |
| 6 | `chore: update Dockerfile, requirements, README, copilot instructions` | Infrastructure |

---

## Phase 1 — File Structure Split

Delete `src/train_tamper_unet.py`. Replace with:

```
src/
  dataset.py   — TamperDataset (COCO), JPEG pipeline, augmentations
  model.py     — ConvBlock (with BatchNorm), UNetSmall (increased capacity)
  loss.py      — dice_loss, bce_dice_loss
  train.py     — main(), evaluate(), save_debug(), tqdm loop
  infer.py     — CLI inference on arbitrary images
scripts/
  download_coco.sh   — downloads train2017 (~18 GB) and val2017 (~788 MB)
```

No logic changes in this commit — pure reorganisation so the diff is reviewable.

---

## Phase 2 — Dataset (`src/dataset.py`)

### COCO loader

- Source: COCO 2017 — `train2017` (~118K images) for training, `val2017` (~5K images) for evaluation
- No annotations required — images only, no `pycocotools` dependency
- Loader: `pathlib.Path.glob("**/*.jpg")` over `datasets/coco/train2017` and `datasets/coco/val2017`
- **Fail fast**: if COCO directory is missing, raise `FileNotFoundError` with a clear message pointing to `scripts/download_coco.sh`

### JPEG pipeline fix

Current 3-pass JPEG destroys JPEG DCT grid inconsistencies — one of the strongest real-world splice signals. Reduce to 2 realistic passes:

| Pass | Old behaviour | New behaviour |
|---|---|---|
| Pre-paste on target | quality 70–95, 90% chance | quality 75–95, **50% chance** — simulates source image previously saved as JPEG |
| Post-paste | quality 40–95, 90% chance | quality 70–95, **80% chance** — simulates saving the forged image |
| Third pass | quality 60–95, 70% chance | **Removed** |

### Augmentations (via albumentations)

Applied to the full image *before* the splice operation:

- `HorizontalFlip(p=0.5)`
- `RandomRotate90(p=0.5)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5)`

### Other fixes

- Fix docstring: "copy-move forgery" → "splice forgery"
- `CFG.image_size`: 128 → 256 (COCO images have real texture detail at this resolution)

---

## Phase 3 — Model (`src/model.py`)

### BatchNorm in ConvBlock

Addresses the training instability visible in debug images (large blob predictions at step 200 before converging at step 400):

```
Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU
```

### Increased channel width

CIFAR-10's 32×32 blurriness made splice boundaries trivially detectable. COCO at 256×256 has real high-frequency texture — the model needs more capacity to learn subtle JPEG artifact and boundary patterns.

| Layer | Old | New |
|---|---|---|
| enc1 | 3 → 32 | 3 → 64 |
| enc2 | 32 → 64 | 64 → 128 |
| bottleneck | 64 → 128 | 128 → 256 |
| dec2 | 128 → 64 | 256 → 128 |
| dec1 | 64 → 32 | 128 → 64 |
| out | 32 → 1 | 64 → 1 |

---

## Phase 4 — Training (`src/train.py`)

### Evaluation metrics

`evaluate()` runs on `val2017` after each epoch and prints:

- **Pixel IoU** — intersection over union at 0.5 threshold
- **F1 / Dice** — on the predicted binary mask vs ground truth
- **Pixel accuracy** — fraction of correctly classified pixels

These make training progress objectively measurable rather than relying on visual debug inspection.

### LR scheduler

`CosineAnnealingLR` decays learning rate from `1e-3` to `1e-5` over all epochs. Prevents the oscillation observed in the debug PNGs between steps 200 and 400.

### tqdm progress bar

Wrap the inner training loop with `tqdm`. Already installed, currently unused.

### Bug fix

Fix possibly-unbound `imgs`/`masks` at the epoch-end `save_debug` call (LSP warning on the old line 313) — guard the call so it only runs if the dataloader yielded at least one batch.

---

## Phase 5 — Inference (`src/infer.py`)

CLI script for running the trained model against any image:

```bash
python src/infer.py --image path/to/image.jpg --model outputs/unet_small.pt
```

Output (written alongside the input file):

- `image_heatmap.png` — original image with predicted confidence heatmap overlaid at 50% opacity
- `image_mask.png` — thresholded binary mask (threshold = 0.5)
- Stdout: max confidence score, predicted tampered area as % of total image

Handles arbitrary input resolution — resizes to `CFG.image_size` for the model, overlays at original resolution.

---

## Phase 6 — Infrastructure

### `scripts/download_coco.sh`

```bash
# Usage: bash scripts/download_coco.sh
# Downloads COCO 2017 images to datasets/coco/
```

- Prints a size warning before the 18 GB train download
- Downloads and extracts `val2017.zip` (~788 MB) and `train2017.zip` (~18 GB)
- Creates `datasets/coco/train2017/` and `datasets/coco/val2017/`

### Docker

- Add `wget` and `unzip` to `apt-get` system deps
- Dataset mounted as a volume (not baked into the image):

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/datasets:/app/datasets" \
  forensic-vision
```

### `requirements.txt`

- Pin `tqdm` version for consistency
- No new packages required — all deps already present

---

## Out of Scope for This Branch

- Pillar 2: AI generation detection
- Pillar 3: Metadata analysis
- Pillar 4: Steganography / hidden embedding
- GPU cloud training setup
- Web UI or API wrapper

---

## Success Criteria

- [ ] Training runs end-to-end on COCO without errors
- [ ] Evaluation metrics print after each epoch (IoU, F1, pixel accuracy)
- [ ] `infer.py` produces a heatmap and mask on an arbitrary input image
- [ ] Debug PNGs show heatmap converging to splice region within first epoch
- [ ] All code lives in the split module structure, old single file removed
