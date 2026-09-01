"""Command-line entry point for CelebA-HQ manifest preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .splits import write_split_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a local CelebA-HQ directory and write deterministic split manifests"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("splits/celeba_hq"))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-images",
        type=int,
        default=30_000,
        help="expected count; use 0 only for a deliberately incomplete local subset",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    expected_images = None if args.expected_images == 0 else args.expected_images
    metadata = write_split_manifests(
        args.data_dir,
        args.output_dir,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        expected_images=expected_images,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
