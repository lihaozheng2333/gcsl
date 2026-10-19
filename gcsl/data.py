"""Load face-frame manifests organized by video."""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VideoFrames(Dataset):
    def __init__(self, manifest, root, processor, frames_per_video=8, training=False):
        self.videos = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.root = Path(root)
        self.processor = processor
        self.training = training
        self.frames_per_video = frames_per_video
        if frames_per_video <= 0 or not self.videos:
            raise ValueError("The video manifest must be nonempty and frames_per_video must be positive")
        seen = set()
        for video in self.videos:
            if video["video_id"] in seen or video["label"] not in (0, 1) or not video["frames"]:
                raise ValueError("Video IDs must be unique, labels must be 0/1, and frame lists must be nonempty")
            seen.add(video["video_id"])
        self.resample()

    def resample(self):
        self.samples = []
        for video in self.videos:
            frames = video["frames"]
            count = min(len(frames), self.frames_per_video)
            if self.training:
                selected = random.sample(frames, count)
            else:
                selected = [frames[i] for i in np.linspace(0, len(frames) - 1, count, dtype=int)]
            self.samples.extend((p, video["label"], video["video_id"]) for p in selected)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, video_id = self.samples[index]
        with Image.open(self.root / path) as image:
            image = image.convert("RGB")
            if self.training and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixels, label, video_id
