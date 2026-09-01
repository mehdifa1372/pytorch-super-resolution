# Experimental protocol

## Primary experiment

Train SRCNN at x2 on the fixed CelebA-HQ training manifest. Select checkpoints using validation
PSNR only, then evaluate once on the test manifest. Run seeds 17, 42, and 73 and report the mean
and standard deviation of each test metric and gain over bicubic.

| Run | Seed | Commit | Best epoch | Bicubic PSNR | SRCNN PSNR | Δ PSNR | SRCNN SSIM | Runtime |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Pending | 17 | — | — | — | — | — | — | — |
| Pending | 42 | — | — | — | — | — | — | — |
| Pending | 73 | — | — | — | — | — | — | — |

## Controlled ablations

Change one factor at a time from the primary configuration:

- Scale: x2 versus x4.
- Patch size: 64, 96, and 128 pixels.
- Training duration: learning curves and early stopping behavior.
- Model: bicubic, SRCNN, and a future residual/sub-pixel baseline.
- Degradation: bicubic-only versus a documented blur/noise/compression pipeline.

## Reporting checklist

- Commit SHA and clean/dirty repository state
- Manifest and checkpoint SHA-256 hashes
- Dataset access date and terms reviewed
- Seed and deterministic settings
- Hardware, software versions, elapsed time, and peak memory
- Scale, degradation, crop, border, and color-space policies
- Bicubic baseline evaluated on identical tensors
- Fixed qualitative examples plus failure cases
- No dataset images redistributed in commits, releases, or CI artifacts
