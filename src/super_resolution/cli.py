"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import evaluate_checkpoint, upscale_image
from .training import TrainingConfig, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run a PyTorch SRCNN model")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train an SRCNN checkpoint")
    train.add_argument("--train-dir", type=Path, required=True)
    train.add_argument("--validation-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--scale", type=int, default=2)
    train.add_argument("--patch-size", type=int, default=96)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    infer = commands.add_parser("infer", help="Upscale one image from a trained checkpoint")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--scale", type=int)
    infer.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    evaluate = commands.add_parser("evaluate", help="Evaluate a checkpoint on HR images")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--patch-size", type=int)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        train_model(
            TrainingConfig(
                train_dir=args.train_dir,
                validation_dir=args.validation_dir,
                output_dir=args.output_dir,
                scale=args.scale,
                patch_size=args.patch_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                workers=args.workers,
                seed=args.seed,
                device=args.device,
            )
        )
        return
    if args.command == "infer":
        result = upscale_image(
            args.input,
            args.output,
            args.checkpoint,
            scale=args.scale,
            device_name=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    evaluate_checkpoint(
        args.data_dir,
        args.checkpoint,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

