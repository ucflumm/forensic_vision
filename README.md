# forensic-vision

A multi-pronged toolkit for detecting digital image authenticity issues — from pixel-level tampering to AI generation artifacts, metadata inconsistencies, and hidden embedded data.

---

## Vision

Modern image deception operates on multiple layers simultaneously. A single detection technique is not enough. **forensic-vision** is being built as a modular, composable system with four independent detection pillars that can be run individually or in combination to produce a holistic authenticity report.

| Pillar | Focus | Status |
|---|---|---|
| **Image Manipulation Detection** | Copy-move forgery, splicing, cloning — structural pixel-level tampering | In Progress |
| **AI Generation Detection** | Identifying AI/deepfake-generated images via artifact patterns | Planned |
| **Metadata Analysis** | EXIF data, file headers, provenance chains, timestamp inconsistencies | Planned |
| **Steganography / Hidden Embedding** | Detecting secret data embedded in image pixels (LSB, DCT, etc.) | Planned |

---

## Pillar 1: Image Manipulation Detection

Detects **copy-move forgery** — where a region of an image is copied and pasted elsewhere within the same image, sometimes with a slight brightness shift to evade naive detection.

A small U-Net is trained to produce a binary segmentation mask over the tampered region. The training dataset is entirely synthetic: CIFAR-10 images are upscaled to 128×128, then forgeries are generated on-the-fly per sample — no labelled forgery dataset required.

### Model

```
Input (3×128×128)
  └─ Encoder: 3→32 → pool → 64 → pool
  └─ Bottleneck: 64→128
  └─ Decoder: 128→64 (+ skip) → 32 (+ skip)
  └─ Output: 1×128×128 logit mask
```

Loss: combined `BCEWithLogitsLoss` + Dice loss to handle class imbalance (tampered region is a small fraction of the image).

### Quickstart

**Local (venv):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train_tamper_unet.py
```

**Docker:**

```bash
docker build -t forensic-vision .
docker run --rm -v "$(pwd)/outputs:/app/outputs" forensic-vision
```

Outputs are written to `outputs/`:
- `debug_{epoch}_{step}.png` — 3-column grid: input image / ground-truth mask / predicted heatmap overlay
- `unet_small.pt` — saved model weights

### Configuration

All hyperparameters live in the `CFG` dataclass at the top of `src/train_tamper_unet.py`:

```python
image_size  = 128
patch_min   = 20
patch_max   = 60
batch_size  = 16
lr          = 1e-3
epochs      = 3
device      = "cpu"   # change to "cuda" for GPU
```
