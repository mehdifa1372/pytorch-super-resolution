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

