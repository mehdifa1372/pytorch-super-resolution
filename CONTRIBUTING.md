# Contributing

Keep changes reproducible and focused. New model or training claims must state the dataset, license, split, degradation process, scale, color space, border policy, seed, hardware, runtime, and bicubic baseline.

```bash
pip install -e ".[dev]"
ruff check src scripts tests
pytest -q
```

Do not commit datasets, generated images, credentials, training runs, or model checkpoints. Publish checkpoints as versioned release assets only after completing the model card and recording a SHA-256 digest.

