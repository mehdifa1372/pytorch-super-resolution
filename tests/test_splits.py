from pathlib import Path

import pytest

from super_resolution.splits import split_paths, write_split_manifests


def test_split_paths_are_deterministic_and_disjoint():
    paths = [Path(f"{index:05d}.png") for index in range(20)]

    first = split_paths(paths, seed=7)
    second = split_paths(reversed(paths), seed=7)

    assert first == second
    assert len(first["train"]) == 16
    assert len(first["validation"]) == 2
    assert len(first["test"]) == 2
    assert not (set(first["train"]) & set(first["test"]))


def test_manifest_writer_checks_expected_count(tmp_path: Path):
    (tmp_path / "one.png").touch()

    with pytest.raises(ValueError, match="expected 30000 images"):
        write_split_manifests(tmp_path, tmp_path / "splits")
