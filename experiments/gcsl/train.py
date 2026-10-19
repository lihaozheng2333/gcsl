"""Train the detector with cumulative-prefix supervision."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPImageProcessor

from gcsl.data import VideoFrames
from gcsl.model import GCSLDetector


def main():
    parser = argparse.ArgumentParser(description="Train GCSL")
    parser.add_argument("--config", default="configs/gcsl.yaml")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="outputs/gcsl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    log_path = logs / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    processor = CLIPImageProcessor.from_pretrained(cfg["backbone"])
    dataset = VideoFrames(args.train_json, args.data_root, processor, cfg["train_frames"], True)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["workers"], pin_memory=args.device.startswith("cuda"))
    # Run layer-wise SVD on CPU before moving the model to the training device.
    model = GCSLDetector(cfg["backbone"], cfg["ranks"], cfg["nst"]).to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    use_amp = cfg["amp"] and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    device_type = torch.device(args.device).type
    for epoch in range(cfg["epochs"]):
        dataset.resample()
        model.train()
        totals = {"loss": 0., "cls": 0., "orth": 0.}
        count = 0
        for images, labels, _ in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            model.sample_prefix()
            with torch.autocast(device_type=device_type, enabled=use_amp):
                cls_loss = F.cross_entropy(model(images), labels)
            orth = model.orthogonality_loss() if cfg["lambda_orth"] else cls_loss.new_zeros(())
            loss = cls_loss + cfg["lambda_orth"] * orth
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch = labels.numel()
            count += batch
            for key, value in zip(totals, (loss, cls_loss, orth)):
                totals[key] += float(value.detach()) * batch
        record = {"epoch": epoch + 1, **{k: v / count for k, v in totals.items()}}
        print(json.dumps(record, ensure_ascii=False))
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Save adapter factors and the classification head without duplicating pretrained backbone weights.
        torch.save({"adapter": model.adapter_state_dict(), "config": cfg,
                    "epoch": epoch + 1}, output / f"epoch_{epoch + 1:02d}.pt")


if __name__ == "__main__":
    main()
