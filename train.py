import os
import random
from collections import defaultdict, Counter

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models


ROI_DIR = "data/roi"
MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_MAP = {
    "eating": "eat_drink",
    "drinking": "eat_drink",
    "sitting": "sitting",
    "standing": "standing",
    "moving": "moving"
}

FINAL_LABELS = ["eat_drink", "sitting", "standing", "moving"]
LABEL_TO_IDX = {l: i for i, l in enumerate(FINAL_LABELS)}

IMG_SIZE = 224
SEQ_LEN = 32
STRIDE = 16

BATCH_SIZE = 8
EPOCHS = 100
PATIENCE = 10
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if random.random() < 0.3:
        angle = random.uniform(-8, 8)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

    if random.random() < 0.3:
        alpha = random.uniform(0.85, 1.15)
        img = np.clip(img * alpha, 0, 255).astype(np.uint8)

    if random.random() < 0.2:
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    return np.transpose(img, (2, 0, 1))


def parse_frame_id(f):
    try:
        return int(os.path.splitext(f)[0].split("_")[-1])
    except:
        return None

def parse_video_id(f):
    return "_".join(os.path.splitext(f)[0].split("_")[:-1])

def build_sequences():
    samples = []

    for orig_label in os.listdir(ROI_DIR):
        if orig_label not in LABEL_MAP:
            continue

        label = LABEL_MAP[orig_label]
        label_idx = LABEL_TO_IDX[label]

        files = os.listdir(os.path.join(ROI_DIR, orig_label))

        grouped = defaultdict(list)
        for f in files:
            fid = parse_frame_id(f)
            if fid is None:
                continue
            vid = parse_video_id(f)
            grouped[vid].append((fid, os.path.join(ROI_DIR, orig_label, f)))

        for items in grouped.values():
            items.sort()
            paths = [p for _, p in items]

            if len(paths) < SEQ_LEN:
                continue

            for i in range(0, len(paths) - SEQ_LEN + 1, STRIDE):
                samples.append({
                    "paths": paths[i:i+SEQ_LEN],
                    "label": label_idx
                })

    return samples

class ROIDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = [load_image(p) for p in s["paths"]]
        return torch.tensor(np.stack(frames)), torch.tensor(s["label"])



class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights="DEFAULT")
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        for p in self.features.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNN()
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, len(FINAL_LABELS))

    def forward(self, x):
        b,t,c,h,w = x.shape
        x = x.view(b*t,c,h,w)
        f = self.backbone(x).view(b,t,-1)
        out,_ = self.lstm(f)
        return self.fc(self.dropout(out[:,-1]))

class CNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNN()
        encoder = nn.TransformerEncoderLayer(256, 8, 512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, len(FINAL_LABELS))

    def forward(self, x):
        b,t,c,h,w = x.shape
        x = x.view(b*t,c,h,w)
        f = self.backbone(x).view(b,t,-1)
        out = self.transformer(f)
        return self.fc(self.dropout(out.mean(1)))



def train_model(name, train_loader, val_loader, weights):
    print(f"\nTraining {name}")

    model = CNNLSTM() if name == "lstm" else CNNTransformer()
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x,y in tqdm(train_loader, desc=f"{name} {epoch+1}"):
            x,y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)

        train_acc = correct / total


        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                val_correct += (pred==y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(f"\n📊 {name.upper()} Epoch {epoch+1}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{MODEL_DIR}/{name}.pt")
            print("saved")

        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("early stop")
                break

    print(f"{name} best: {best:.4f}")


def main():
    samples = build_sequences()
    print("Total samples:", len(samples))

    dataset = ROIDataset(samples)

    labels = [s["label"] for s in samples]
    counts = Counter(labels)

    weights = np.array([1 / np.sqrt(counts[i] + 1) for i in range(len(FINAL_LABELS))])
    weights = weights / weights.sum() * len(FINAL_LABELS)
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("Class weights:", weights)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    train_model("lstm", train_loader, val_loader, weights)
    train_model("transformer", train_loader, val_loader, weights)

if __name__ == "__main__":
    main()