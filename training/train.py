"""Fine-tune and evaluate a YOLO model on a YOLO-format custom dataset.

Example:
    python training/train.py --data datasets/my-data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse training and evaluation options."""

    parser = argparse.ArgumentParser(description="Train and evaluate a custom Ultralytics YOLO detector.")
    parser.add_argument("--data", required=True, help="Path to a YOLO-format data.yaml file.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint or model config.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 lets Ultralytics auto-select it.")
    parser.add_argument("--device", default=None, help="cpu, 0, 0,1, or another Ultralytics device value.")
    parser.add_argument("--project", default="runs/visionlens")
    parser.add_argument("--name", default="custom-detector")
    return parser.parse_args()


def main() -> None:
    """Train, validate, and print the primary detection metrics."""

    args = parse_args()
    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "plots": True,
    }
    if args.device:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        project=args.project,
        name=f"{args.name}-val",
        plots=True,
    )
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    output_dir = Path(args.project) / f"{args.name}-val"
    print(f"Validation artifacts (including confusion matrix when supported): {output_dir}")
    print(f"Best weights to copy into VisionLens: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

