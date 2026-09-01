"""On-the-fly paired image datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .images import bicubic_pair, pil_to_tensor

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _read_manifest(root: Path, manifest: Path) -> list[Path]:
    """Resolve a newline-delimited, root-relative image manifest safely."""
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    files: list[Path] = []
    seen: set[Path] = set()
    resolved_root = root.resolve()
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        relative = Path(entry)
        candidate = (root / relative).resolve()
        if relative.is_absolute() or resolved_root not in candidate.parents:
            raise ValueError(f"unsafe path in {manifest}:{line_number}: {entry}")
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image in {manifest}:{line_number}: {entry}")
        if not candidate.is_file():
            raise ValueError(f"missing image in {manifest}:{line_number}: {entry}")
        if candidate in seen:
            raise ValueError(f"duplicate image in {manifest}:{line_number}: {entry}")
        seen.add(candidate)
        files.append(candidate)
    if not files:
        raise ValueError(f"manifest contains no images: {manifest}")
    return files


class SuperResolutionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Generate bicubic input/target pairs from a directory of high-resolution images."""

    def __init__(
        self,
        root: str | Path,
        *,
        scale: int = 2,
        patch_size: int | None = 96,
        training: bool = True,
        manifest: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.scale = scale
        self.patch_size = patch_size
        self.training = training
        if not self.root.is_dir():
            raise ValueError(f"image directory does not exist: {self.root}")
        if scale < 2:
            raise ValueError("scale must be at least 2")
        if patch_size is not None and (patch_size <= 0 or patch_size % scale):
            raise ValueError("patch_size must be positive and divisible by scale")
        self.files = (
            _read_manifest(self.root, Path(manifest))
            if manifest is not None
            else sorted(
                path for path in self.root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )
        if not self.files:
            raise ValueError(f"no supported images found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def _crop(self, image: Image.Image) -> Image.Image:
        if self.patch_size is None:
            return image
        size = self.patch_size
        if image.width < size or image.height < size:
            raise ValueError(
                f"{image.width}x{image.height} image is smaller than the {size}x{size} patch"
            )
        if self.training:
            left = int(torch.randint(0, image.width - size + 1, (1,)).item())
            top = int(torch.randint(0, image.height - size + 1, (1,)).item())
        else:
            left = (image.width - size) // 2
            top = (image.height - size) // 2
        return image.crop((left, top, left + size, top + size))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.files[index]) as source:
            image = self._crop(source.convert("RGB"))
            bicubic, target = bicubic_pair(image, self.scale)
        return pil_to_tensor(bicubic), pil_to_tensor(target)
