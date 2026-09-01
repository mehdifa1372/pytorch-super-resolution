"""Generate a tiny synthetic image dataset for an end-to-end smoke run."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def generate_image(path: Path, *, seed: int, size: int = 128) -> None:
    generator = random.Random(seed)
    background = tuple(generator.randint(15, 80) for _ in range(3))
    image = Image.new("RGB", (size, size), background)
    drawing = ImageDraw.Draw(image)
    for _ in range(18):
        left = generator.randint(0, size - 24)
        top = generator.randint(0, size - 24)
        right = generator.randint(left + 8, min(size, left + 48))
        bottom = generator.randint(top + 8, min(size, top + 48))
        color = tuple(generator.randint(80, 255) for _ in range(3))
        if generator.random() < 0.5:
            drawing.rectangle((left, top, right, bottom), fill=color)
        else:
            drawing.ellipse((left, top, right, bottom), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_count <= 0 or args.validation_count <= 0:
        raise SystemExit("image counts must be positive")
    for split, count, offset in (
        ("train", args.train_count, 0),
        ("validation", args.validation_count, 10_000),
    ):
        for index in range(count):
            generate_image(
                args.output_dir / split / f"synthetic-{index:03d}.png",
                seed=args.seed + offset + index,
            )
    print(f"Generated demo data under {args.output_dir}")


if __name__ == "__main__":
    main()

