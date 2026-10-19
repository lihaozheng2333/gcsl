"""Full-prefix inference with frame-level and video-level AUC."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPImageProcessor

from gcsl.data import VideoFrames
from gcsl.model import GCSLDetector


def main():
    parser = argparse.ArgumentParser(description="Evaluate GCSL")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="outputs/gcsl/evaluation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    model = GCSLDetector(cfg["backbone"], cfg["ranks"], cfg["nst"])
    model.load_adapter(checkpoint["adapter"])
    model.to(args.device).eval()
    processor = CLIPImageProcessor.from_pretrained(cfg["backbone"])
    dataset = VideoFrames(args.test_json, args.data_root, processor, cfg["test_frames"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=cfg["workers"])
    labels_all, scores_all = [], []
    videos = defaultdict(list)
    with torch.inference_mode():
        for images, labels, ids in tqdm(loader, desc="Evaluation"):
            scores = model(images.to(args.device)).softmax(dim=-1)[:, 1].cpu().tolist()
            for label, score, video_id in zip(labels.tolist(), scores, ids):
                labels_all.append(label)
                scores_all.append(score)
                videos[video_id].append((label, score))
    def auc(labels, scores):
        return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
    video_predictions = [{"video_id": key, "label": rows[0][0],
                          "score": float(np.mean([row[1] for row in rows]))}
                         for key, rows in videos.items()]
    result = {"frame_auc": auc(labels_all, scores_all),
              "video_auc": auc([v["label"] for v in video_predictions],
                               [v["score"] for v in video_predictions]),
              "num_frames": len(labels_all), "num_videos": len(videos),
              "videos": video_predictions}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "videos"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
