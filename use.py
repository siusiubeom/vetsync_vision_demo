import os
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO



VIDEO_PATH = "data/videos/014.mp4"
MODEL_PATH = "checkpoints/lstm.pt"

IMG_SIZE = 224
SEQ_LEN = 32

FINAL_LABELS = ["eat_drink", "sitting", "standing", "moving"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


class SmallCNN(torch.nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        from torchvision import models
        backbone = models.resnet18(weights="DEFAULT")
        self.features = torch.nn.Sequential(*list(backbone.children())[:-1])


        for p in self.features.parameters():
            p.requires_grad = False

        self.fc = torch.nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)

class CNNLSTM(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = SmallCNN()
        self.lstm = torch.nn.LSTM(256, 256, batch_first=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        f = self.backbone(x).view(b, t, -1)
        out, _ = self.lstm(f)
        return self.fc(self.dropout(out[:, -1]))

model = CNNLSTM(len(FINAL_LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


yolo = YOLO("yolo/yolo26x-seg.pt")
yolo.to(DEVICE)


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


def get_roi(frame, yolo_model):
    orig_h, orig_w = frame.shape[:2]

    resized = cv2.resize(frame, (640, 640))
    result = yolo_model(resized, verbose=False)[0]

    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best = int(np.argmax(areas))

        mask = masks[best]
        box = boxes[best]

        x1 = int(box[0] * orig_w / 640)
        y1 = int(box[1] * orig_h / 640)
        x2 = int(box[2] * orig_w / 640)
        y2 = int(box[3] * orig_h / 640)

        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        mask = cv2.resize(mask, (orig_w, orig_h))
        mask = (mask > 0.3).astype(np.uint8)

        roi = frame * np.expand_dims(mask, axis=-1)
        crop = roi[y1:y2, x1:x2]

        if crop.size > 0:
            return crop, (x1, y1, x2, y2)

    cx, cy = orig_w // 2, orig_h // 2
    size = min(orig_w, orig_h) // 2
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(orig_w, cx + size // 2)
    y2 = min(orig_h, cy + size // 2)
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


cap = cv2.VideoCapture(VIDEO_PATH)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (orig_w, orig_h))

buffer = deque(maxlen=SEQ_LEN)
prob_history = deque(maxlen=10)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi, box = get_roi(frame, yolo)

    tensor = preprocess(roi)
    buffer.append(tensor)

    label_text = "..."
    conf_text = ""

    if len(buffer) == SEQ_LEN:
        seq = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(seq)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prob_history.append(probs)
        avg_probs = np.mean(prob_history, axis=0)

        pred = int(np.argmax(avg_probs))
        conf = avg_probs[pred]

        label_text = FINAL_LABELS[pred]
        conf_text = f"{conf:.2f}"

    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"{label_text} ({conf_text})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - 40), (x1 + tw + 10, y1), (0, 255, 0), -1)
    cv2.putText(frame, text, (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    out.write(frame)

    scale = min(960 / orig_w, 720 / orig_h, 1.0)
    display = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
    cv2.imshow("Dog Behavior Detection", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
out.release()
cv2.destroyAllWindows()