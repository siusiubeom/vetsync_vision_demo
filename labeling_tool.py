import sys
import os
import cv2
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QSlider, QListWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


VIDEO_DIR = os.path.abspath("data/videos")
LABEL_DIR = os.path.abspath("data/labels")
os.makedirs(LABEL_DIR, exist_ok=True)

LABELS = ["eating", "drinking", "sitting", "standing", "moving"]

def rename_videos_to_index(folder):
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".mp4")
    )

    if not files:
        print("No videos to rename")
        return

    if all(os.path.splitext(f)[0].isdigit() and len(os.path.splitext(f)[0]) == 3 for f in files):
        print("Videos already in 001 format")
        return

    print("Renaming videos to 001 format...")

    temp_pairs = []
    skipped = []

    for i, f in enumerate(files):
        old_path = os.path.join(folder, f)
        temp_path = os.path.join(folder, f"__tmp__{i:03d}.mp4")
        try:
            os.rename(old_path, temp_path)
            temp_pairs.append((f, temp_path))
        except PermissionError:
            skipped.append(f)
            print(f"Skipped locked file: {f}")

    next_idx = 1
    for _, temp_path in temp_pairs:
        while True:
            new_name = f"{next_idx:03d}.mp4"
            new_path = os.path.join(folder, new_name)
            if not os.path.exists(new_path):
                break
            next_idx += 1

        os.rename(temp_path, new_path)
        next_idx += 1

    print(f"Renamed {len(temp_pairs)} videos")
    if skipped:
        print(f"{len(skipped)} file(s) were locked and left unchanged")
class VideoLabeler(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dog Behavior Labeler")
        self.setGeometry(100, 100, 1000, 700)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.fps = 30
        self.frame_id = 0
        self.total_frames = 0

        self.start_time = None
        self.segments = []
        self.current_video = None

        self.is_seeking = False

        self.init_ui()
        self.load_videos()

    def init_ui(self):
        layout = QVBoxLayout()

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.video_label = QLabel("No video loaded")
        self.video_label.setFixedHeight(400)
        self.video_label.setStyleSheet("border: 2px solid black;")
        layout.addWidget(self.video_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderPressed.connect(self.start_seek)
        self.slider.sliderReleased.connect(self.end_seek)
        layout.addWidget(self.slider)

        btn_layout = QHBoxLayout()

        QPushButton("Play/Pause", clicked=self.play_pause)
        self.play_btn = QPushButton("Play/Pause")
        self.play_btn.clicked.connect(self.play_pause)
        btn_layout.addWidget(self.play_btn)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.mark_start)
        btn_layout.addWidget(self.start_btn)

        self.end_btn = QPushButton("End")
        self.end_btn.clicked.connect(self.mark_end)
        btn_layout.addWidget(self.end_btn)

        layout.addLayout(btn_layout)


        label_layout = QHBoxLayout()
        for label in LABELS:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, l=label: self.assign_label(l))
            label_layout.addWidget(btn)

        layout.addLayout(label_layout)

        self.video_list = QListWidget()
        self.video_list.itemClicked.connect(self.load_selected_video)
        layout.addWidget(self.video_list)

        self.setLayout(layout)

    def load_videos(self):
        if not os.path.exists(VIDEO_DIR):
            self.status_label.setText("Folder missing")
            return

        files = os.listdir(VIDEO_DIR)

        self.video_files = [
            os.path.join(VIDEO_DIR, f)
            for f in files if f.lower().endswith(".mp4")
        ]

        self.video_list.clear()
        for f in self.video_files:
            self.video_list.addItem(os.path.basename(f))

        if self.video_files:
            self.load_video(self.video_files[0])

    def load_selected_video(self, item):
        index = self.video_list.row(item)
        self.load_video(self.video_files[index])

    def load_video(self, path):
        self.save_current()

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.status_label.setText("Cannot open video")
            return

        self.current_video = path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_id = 0

        self.slider.setMaximum(self.total_frames)

        self.segments = []
        self.start_time = None

        self.status_label.setText(f"Loaded: {os.path.basename(path)}")

    def play_pause(self):
        if not self.cap:
            return

        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(int(1000 / self.fps))

    def next_frame(self):
        if self.is_seeking:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        self.frame_id += 1
        self.slider.setValue(self.frame_id)

        self.display_frame(frame)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        label_w = self.video_label.width()
        label_h = self.video_label.height()

        scale = min(label_w / w, label_h / h)
        resized = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        img = QImage(resized.data, resized.shape[1], resized.shape[0],
                     ch * resized.shape[1], QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(img))

    def start_seek(self):
        self.is_seeking = True
        self.timer.stop()

    def end_seek(self):
        self.frame_id = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)

        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)

        self.is_seeking = False

    def mark_start(self):
        self.start_time = self.frame_id / self.fps
        self.status_label.setText(f"START @ {self.start_time:.2f}s")

        self.video_label.setStyleSheet("border: 3px solid yellow;")

    def mark_end(self):
        self.end_time = self.frame_id / self.fps
        self.status_label.setText(f"END @ {self.end_time:.2f}s")

        self.video_label.setStyleSheet("border: 2px solid black;")

    def assign_label(self, label):
        if self.start_time is None:
            self.status_label.setText("No START set")
            return

        self.segments.append((self.start_time, self.end_time, label))
        self.status_label.setText(f"Saved: {label}")

        self.start_time = None

    def save_current(self):
        if not self.current_video or not self.segments:
            return

        name = os.path.splitext(os.path.basename(self.current_video))[0]
        save_path = os.path.join(LABEL_DIR, f"{name}.csv")

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["start", "end", "label"])
            writer.writerows(self.segments)

        print("Saved →", save_path)

    def closeEvent(self, event):
        self.save_current()
        event.accept()


if __name__ == "__main__":
    rename_videos_to_index(VIDEO_DIR)
    app = QApplication(sys.argv)
    window = VideoLabeler()
    window.show()
    sys.exit(app.exec_())