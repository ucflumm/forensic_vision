# forensic-vision

A multi-pronged toolkit for detecting digital image authenticity issues — from pixel-level tampering to AI generation artifacts, metadata inconsistencies, and hidden embedded data.

---

## Vision

Modern image deception operates on multiple layers simultaneously. A single detection technique is not enough. **forensic-vision** is being built as a modular, composable system with four independent detection pillars that can be run individually or in combination to produce a holistic authenticity report.

| Pillar | Focus | Status |
|---|---|---|
| **Image Manipulation Detection** | Splice forgery, cloning — structural pixel-level tampering | In Progress |
| **AI Generation Detection** | Identifying AI/deepfake-generated images via artifact patterns | Planned |
| **Metadata Analysis** | EXIF data, file headers, provenance chains, timestamp inconsistencies | Planned |
| **Steganography / Hidden Embedding** | Detecting secret data embedded in image pixels (LSB, DCT, etc.) | Planned |

---

## Pillar 1: Image Manipulation Detection

Detects **splice forgery** — where a region from one photograph is composited into another, the core operation behind most photoshopped images.

A U-Net is trained to produce a binary segmentation mask over the spliced region. Training data is entirely synthetic: COCO 2017 photographs are tampered on-the-fly per sample — no pre-labelled forgery dataset required.

### Model

```
Input (3×256×256)
  └─ Encoder:     3→64 → pool → 128 → pool
  └─ Bottleneck:  128→256
  └─ Decoder:     256→128 (+ skip) → 64 (+ skip)
  └─ Output:      1×256×256 logit mask
```

Each encoder/decoder block uses `Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU`.
Loss: combined `BCEWithLogitsLoss` + Dice loss to handle class imbalance.

### Source layout

```
src/
  dataset.py   — TamperDataset (COCO), JPEG pipeline, augmentations
  model.py     — ConvBlock (BatchNorm), UNetSmall
  loss.py      — dice_loss, bce_dice_loss
  train.py     — training loop, evaluate(), save_debug()
  infer.py     — inference on arbitrary images
scripts/
  download_coco.sh   — downloads COCO 2017 images
```

### Quickstart

**1. Download COCO images**

```bash
# val2017 only (~788 MB) — enough to verify the pipeline
bash scripts/download_coco.sh --val-only

# full dataset — val2017 + train2017 (~18 GB total)
bash scripts/download_coco.sh
```

**2. Train (local venv)**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

**2. Train (Docker)**

```bash
docker build -t forensic-vision .
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/datasets:/app/datasets" \
  forensic-vision
```

Outputs are written to `outputs/`:
- `debug_{epoch}_{step}.png` — 3-column grid: input / ground-truth mask / predicted heatmap overlay
- `unet_small.pt` — saved model weights

Training prints per-epoch metrics to stdout:
```
Epoch 1/10  avg_loss=0.4821  lr=9.51e-04
  Val  |  IoU=0.3142  F1=0.4571  Acc=0.9103
```

**3. Run inference on any image**

```bash
python src/infer.py --image path/to/image.jpg
# optional flags:
#   --model outputs/unet_small.pt   (default)
#   --threshold 0.5                 (default)
```

Outputs saved alongside the input:
- `image_heatmap.png` — confidence heatmap overlaid on original at native resolution
- `image_mask.png` — thresholded binary mask

### Configuration

All hyperparameters live in the `CFG` dataclass at the top of `src/dataset.py`:

```python
image_size  = 256
patch_min   = 40
patch_max   = 110
batch_size  = 16
lr          = 1e-3
epochs      = 10
device      = "cpu"   # change to "cuda" for GPU
```
