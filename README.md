# Dog Behavior Recognition

> Automated activity classification from video using YOLOv8 + CNN-LSTM / CNN-Transformer

---

## Overview

This project classifies dog behaviour from video clips into four categories: **eat/drink, sitting, standing, and moving**. It is a three-stage pipeline: manual segment labelling, ROI extraction with YOLOv8, and sequence-based classification using a CNN-LSTM or CNN-Transformer model.

Training data comes from the Kaggle dataset [Videos - Dog (umuttuygurr)](https://www.kaggle.com/datasets/umuttuygurr/videosdog).

---

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `label.py` | Manual labelling GUI |
| 2 | `extract_roi.py` | YOLO-based ROI crop extraction |
| 3 | `train.py` | Model training (LSTM and Transformer variants) |
| 4 | `eval.py` | Evaluation and confusion matrix generation |
| 5 | `infer.py` | Real-time inference on a video file |

---

## Project Structure

```
dog-behavior/
├── data/
│   ├── videos/          # Raw MP4 files (place Kaggle downloads here)
│   ├── labels/          # CSV segment files produced by label.py
│   └── roi/             # Cropped ROI frames produced by extract_roi.py
│       ├── eating/
│       ├── drinking/
│       ├── sitting/
│       ├── standing/
│       └── moving/
├── yolo/
│   └── yolo26x-seg.pt   # YOLOv8x-seg weights
├── checkpoints/         # Saved model weights and confusion matrix PNGs
├── label.py
├── extract_roi.py
├── train.py
├── eval.py
└── infer.py
```

---

## Data

Download videos from the Kaggle dataset and place all MP4 files into `data/videos/`. When `label.py` is first run it renames every file to a zero-padded three-digit index (`001.mp4`, `002.mp4`, …) so that the labelling CSVs stay consistent with the video filenames.

---

## Stage 1 — Labelling (`label.py`)

A PyQt5 desktop GUI for marking time segments in each video and assigning one of five behaviour labels: eating, drinking, sitting, standing, or moving.

**How to use:**
1. Select a video from the list on the left
2. Play/pause with the Play/Pause button or scrub with the slider
3. Click **Start** at the beginning of a behaviour segment — the border turns yellow
4. Click **End** at the end of the segment
5. Click a behaviour label button to save the segment
6. Segments are written to `data/labels/<video_name>.csv` on video change or window close

### Screenshot
<img width="1006" height="721" alt="image" src="https://github.com/user-attachments/assets/076b2eef-b983-40a2-8fc4-a9c8aeb9eaaf" />

---

## Stage 2 — ROI Extraction (`extract_roi.py`)

Reads each labelled video and its CSV, seeks to the labelled time windows, and runs YOLOv8x-seg on batches of frames. For each frame the largest detected object is selected, its segmentation mask is applied to zero out the background, and the masked crop is saved as a JPEG under `data/roi/<label>/`.

If YOLO finds no objects in a frame, a centre crop is used as a fallback so no labelled frame is lost.

**Key settings (top of file):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 8 | Frames passed to YOLO per call |
| `FRAME_SKIP` | 1 | Process every N-th frame (1 = all frames) |
| `IMG_SIZE` | 640 | Resize before YOLO inference |

---

## Stage 3 — Training (`train.py`)

Builds fixed-length sequences of ROI crops (`SEQ_LEN = 32` frames, `STRIDE = 16`) and trains two models in sequence: CNN-LSTM and CNN-Transformer. Both share the same frozen ResNet-18 backbone as a per-frame feature extractor.

### Model Architectures

**CNN-LSTM**
- ResNet-18 (frozen) → 256-dim frame embedding
- Single-layer LSTM, hidden size 256
- Dropout 0.3 → Linear → 4-class softmax

**CNN-Transformer**
- Same ResNet-18 backbone
- 2-layer TransformerEncoder, 8 heads, FF dim 512
- Mean pooling over time → Dropout → Linear → 4-class softmax

### Training Details

| Setting | Value |
|---------|-------|
| Optimiser | Adam, LR 1e-4 |
| LR schedule | ReduceLROnPlateau |
| Loss | CrossEntropyLoss, label smoothing 0.1, inverse-sqrt class weights |
| Early stopping | Patience 10, max 100 epochs |
| Split | 80/20 train/val |
| Augmentation | Random rotation ±8°, brightness jitter, Gaussian noise, blur |

---

## Stage 4 — Evaluation (`eval.py`)

Loads checkpoints for both models and evaluates on the held-out validation split (same random seed as training). Reports accuracy, macro F1, weighted F1, and a per-class classification report. The standing class is excluded from evaluation metrics due to limited samples.

Confusion matrices (raw counts and row-normalised) are saved to `checkpoints/`.

### LSTM — Confusion Matrix
<img width="540" height="344" alt="image" src="https://github.com/user-attachments/assets/eb16c1f2-d547-4a65-8b68-15b27591227f" />

### Transformer — Confusion Matrix
<img width="535" height="346" alt="image" src="https://github.com/user-attachments/assets/d557d829-5c2b-48b6-b62b-a6c257cf2600" />

---

## Stage 5 — Inference (`infer.py`)

Runs the CNN-LSTM model on a single video file, overlaying the predicted label and confidence on each frame, and writes the result to `output.mp4`. YOLOv8 crops the dog ROI per-frame before passing to the classifier. Predictions are smoothed over a rolling window of 10 sequences to reduce flicker.

Edit `VIDEO_PATH` at the top of `infer.py` to point to your video.

### Demo
https://github.com/user-attachments/assets/6b844064-5336-43ca-9332-6cc8e5641767

---

## Requirements

### Python packages

```bash
pip install torch torchvision opencv-python ultralytics
pip install pyqt5 tqdm scikit-learn matplotlib seaborn
```

### YOLO weights

Place YOLOv8x-seg weights at `yolo/yolo26x-seg.pt`. Any YOLOv8-seg variant works — update the path inside each script if you use a different filename.

### Hardware

A CUDA GPU is strongly recommended for ROI extraction, training, and inference. CPU mode is supported but will be significantly slower.

---

## Quick Start

```bash
# 1. Place videos in data/videos/
python label.py

# 2. Extract ROI crops
python extract_roi.py

# 3. Train
python train.py

# 4. Evaluate
python eval.py

# 5. Run inference on a video
python infer.py
```
