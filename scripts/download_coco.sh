#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# download_coco.sh
# Downloads COCO 2017 image splits to datasets/coco/
#
# Usage:
#   bash scripts/download_coco.sh             # downloads both splits
#   bash scripts/download_coco.sh --val-only  # downloads val2017 only (~788 MB)
#
# Output layout:
#   datasets/coco/train2017/   ~118K images  (~18 GB unzipped)
#   datasets/coco/val2017/     ~5K images    (~788 MB unzipped)
# -----------------------------------------------------------------------------

set -euo pipefail

VAL_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--val-only" ]] && VAL_ONLY=true
done

DEST="datasets/coco"
mkdir -p "$DEST"

# ---------------------------------------------------------------------------
# val2017  (~788 MB zip, ~788 MB unzipped)
# ---------------------------------------------------------------------------
if [ ! -d "$DEST/val2017" ] || [ -z "$(ls -A "$DEST/val2017" 2>/dev/null)" ]; then
  echo "=> Downloading COCO val2017 images (~788 MB)..."
  wget -q --show-progress -O "$DEST/val2017.zip" \
    "http://images.cocodataset.org/zips/val2017.zip"
  echo "=> Extracting val2017..."
  unzip -q "$DEST/val2017.zip" -d "$DEST"
  rm "$DEST/val2017.zip"
  echo "=> val2017 ready: $(ls "$DEST/val2017" | wc -l | tr -d ' ') images"
else
  echo "=> val2017 already present ($(ls "$DEST/val2017" | wc -l | tr -d ' ') images), skipping."
fi

# ---------------------------------------------------------------------------
# train2017 (~18 GB zip, ~18 GB unzipped)
# ---------------------------------------------------------------------------
if [ "$VAL_ONLY" = true ]; then
  echo "=> Skipping train2017 (--val-only flag set)."
  echo "=> Done. Re-run without --val-only to download train2017 (~18 GB)."
  exit 0
fi

echo ""
echo "WARNING: train2017 is approximately 18 GB."
echo "         Make sure you have sufficient disk space before continuing."
echo "         Press Ctrl+C to cancel, or wait 10 seconds to proceed..."
echo ""
sleep 10

if [ ! -d "$DEST/train2017" ] || [ -z "$(ls -A "$DEST/train2017" 2>/dev/null)" ]; then
  echo "=> Downloading COCO train2017 images (~18 GB)..."
  wget -q --show-progress -O "$DEST/train2017.zip" \
    "http://images.cocodataset.org/zips/train2017.zip"
  echo "=> Extracting train2017..."
  unzip -q "$DEST/train2017.zip" -d "$DEST"
  rm "$DEST/train2017.zip"
  echo "=> train2017 ready: $(ls "$DEST/train2017" | wc -l | tr -d ' ') images"
else
  echo "=> train2017 already present ($(ls "$DEST/train2017" | wc -l | tr -d ' ') images), skipping."
fi

echo ""
echo "=> COCO download complete."
echo "   datasets/coco/train2017/  — training images"
echo "   datasets/coco/val2017/    — validation images"
