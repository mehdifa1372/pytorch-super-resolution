"""Deterministic, image-disjoint dataset split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .data import IMAGE_EXTENSIONS


def discover_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"image directory does not exist: {root}")
    files = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"no supported images found in {root}")
    return files


def _score(path: Path, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{path.as_posix()}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def split_paths(
    paths: Iterable[Path],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Path]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")
    ordered = sorted(set(paths), key=lambda path: (_score(path, seed), path.as_posix()))
    if len(ordered) < 3:
        raise ValueError("at least three images are required to create three splits")
    train_size = min(len(ordered) - 2, max(1, int(len(ordered) * train_fraction)))
    validation_size = min(
        len(ordered) - train_size - 1,
        max(1, int(len(ordered) * validation_fraction)),
    )
    validation_end = train_size + validation_size
    return {
        "train": ordered[:train_size],
        "validation": ordered[train_size:validation_end],
        "test": ordered[validation_end:],
    }


def write_split_manifests(
    root: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
    expected_images: int | None = 30_000,
) -> dict[str, object]:
    root = Path(root)
    output_dir = Path(output_dir)
    images = discover_images(root)
    if expected_images is not None and len(images) != expected_images:
        raise ValueError(
            f"expected {expected_images} images, found {len(images)}; "
            "verify the CelebA-HQ preparation or override --expected-images"
        )
    splits = split_paths(
        images,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in splits.items():
        content = "".join(f"{path.as_posix()}\n" for path in paths)
        (output_dir / f"{name}.txt").write_text(content, encoding="utf-8")
    metadata: dict[str, object] = {
        "dataset": "CelebA-HQ",
        "image_count": len(images),
        "seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "split_counts": {name: len(paths) for name, paths in splits.items()},
        "split_unit": "image",
        "identity_disjoint": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
