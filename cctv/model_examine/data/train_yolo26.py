"""
Train YOLO26 on the model_examine dataset.

Example:
  python -m cctv.model_examine.data.train_yolo26 --epochs 100 --imgsz 640
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 on model_examine dataset")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--data",
        type=str,
        default=str(repo_root / "cctv" / "model_examine" / "data" / "data.yaml"),
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(repo_root / "models" / "yolo26s.pt"),
        help="Base model weights or model yaml",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="cuda, cpu, or 0/1")
    parser.add_argument(
        "--project",
        type=str,
        default=str(repo_root / "models" / "runs"),
        help="Project directory for runs",
    )
    parser.add_argument("--name", type=str, default="yolo26_cash", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume training")

    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        task="detect",
    )


if __name__ == "__main__":
    main()
