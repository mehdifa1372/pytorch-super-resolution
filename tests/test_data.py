from pathlib import Path

import pytest
from PIL import Image

from super_resolution.data import SuperResolutionDataset


def test_dataset_produces_aligned_tensor_pairs(tmp_path: Path):
    Image.new("RGB", (100, 80), "navy").save(tmp_path / "sample.png")
    dataset = SuperResolutionDataset(tmp_path, scale=2, patch_size=64, training=False)
    bicubic, target = dataset[0]
    assert bicubic.shape == target.shape == (3, 64, 64)


def test_dataset_requires_images(tmp_path: Path):
    with pytest.raises(ValueError, match="no supported images"):
        SuperResolutionDataset(tmp_path)


def test_dataset_reads_root_relative_manifest(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64), "navy").save(images / "included.png")
    Image.new("RGB", (64, 64), "red").save(images / "excluded.png")
    manifest = tmp_path / "train.txt"
    manifest.write_text("images/included.png\n", encoding="utf-8")

    dataset = SuperResolutionDataset(tmp_path, manifest=manifest, patch_size=64)

    assert len(dataset) == 1
    assert dataset.files[0].name == "included.png"


def test_dataset_rejects_manifest_path_traversal(tmp_path: Path):
    manifest = tmp_path / "train.txt"
    manifest.write_text("../outside.png\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe path"):
        SuperResolutionDataset(tmp_path, manifest=manifest)
