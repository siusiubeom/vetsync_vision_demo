import os
import cv2
import csv
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch


VIDEO_DIR = "data/videos"
LABEL_DIR = "data/labels"
OUTPUT_DIR = "data/roi"

BATCH_SIZE = 8
FRAME_SKIP = 1
IMG_SIZE = 640

LABELS = ["eating", "drinking", "sitting", "standing", "moving"]


os.makedirs(OUTPUT_DIR, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model =YOLO("yolo/yolo26x-seg.pt")
model.to(device)



def process_result(frame, result, label, video_name, save_id):
    h, w = frame.shape[:2]

    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        areas = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            areas.append((x2 - x1) * (y2 - y1))

        best_idx = np.argmax(areas)

        mask = masks[best_idx]
        box = boxes[best_idx].astype(int)

        x1, y1, x2, y2 = box

        mask = cv2.resize(mask, (w, h))
        mask = (mask > 0.3).astype(np.uint8)

        roi = frame * np.expand_dims(mask, axis=-1)
        roi_crop = roi[y1:y2, x1:x2]

        if roi_crop.size > 0:
            save_path = os.path.join(
                OUTPUT_DIR,
                label,
                f"{video_name}_{save_id}.jpg"
            )
            cv2.imwrite(save_path, roi_crop)
            return save_id + 1


    cx, cy = w // 2, h // 2
    size = min(w, h) // 2

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, cx + size // 2)
    y2 = min(h, cy + size // 2)

    roi_crop = frame[y1:y2, x1:x2]

    save_path = os.path.join(
        OUTPUT_DIR,
        label,
        f"{video_name}_{save_id}.jpg"
    )
    cv2.imwrite(save_path, roi_crop)

    return save_id + 1


def process_video(video_path, label_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open(label_path, 'r') as f:
        reader = csv.DictReader(f)
        segments = list(reader)

    save_id = 0

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        label = seg["label"]

        start_frame = int(start * fps)
        end_frame = int(end * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        batch = []
        frames = []

        for frame_id in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % FRAME_SKIP != 0:
                continue

            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

            batch.append(frame_resized)
            frames.append(frame)

            if len(batch) == BATCH_SIZE:
                results = model(batch, device=device, verbose=False)

                for f, r in zip(frames, results):
                    save_id = process_result(f, r, label, video_name, save_id)

                batch = []
                frames = []

        if batch:
            results = model(batch, device=device, verbose=False)
            for f, r in zip(frames, results):
                save_id = process_result(f, r, label, video_name, save_id)

    cap.release()

videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])

for video_file in tqdm(videos):
    video_path = os.path.join(VIDEO_DIR, video_file)
    label_path = os.path.join(LABEL_DIR, video_file.replace(".mp4", ".csv"))

    if not os.path.exists(label_path):
        print(f"Missing label for {video_file}")
        continue

    process_video(video_path, label_path)

print("ROI extraction complete")