# SRCNN CelebA-HQ checkpoint model card

Complete every pending field before publishing weights or reporting results.

## Model details

- Architecture: SRCNN, RGB input/output
- Scale factor: x2
- Parameters: _pending_
- Checkpoint SHA-256: _pending_
- Code commit: _pending_
- PyTorch/CUDA versions: _pending_
- Hardware and training runtime: _pending_

## Intended use

Non-commercial research on fidelity-oriented super-resolution of aligned face crops. The model
is not intended for biometric identification, identity verification, surveillance, forensic
enhancement, or consequential decisions.

## Training data and protocol

- Dataset: CelebA-HQ, 30,000 aligned 1024×1024 face images
- Dataset access/date: _pending_
- Split metadata and manifest hashes: _pending_
- Training seed(s): _pending_
- Degradation: bicubic downsampling and enlargement, scale x2
- Crop size and augmentation: _pending_
- Optimizer, learning rate, batch size, epochs: _pending_

The dataset is not included with this checkpoint. CelebA is available for non-commercial
research and prohibits redistribution; users must obtain it independently and accept its terms.

## Evaluation

Report mean metrics on the untouched test manifest. Do not tune on this split.

| Seed | Split | Bicubic PSNR | SRCNN PSNR | Δ PSNR | Bicubic SSIM | SRCNN SSIM | Δ SSIM |
|---:|---|---:|---:|---:|---:|---:|---:|
| _pending_ | test | — | — | — | — | — | — |

Record RGB versus luminance, border crop, aggregation method, and exact evaluation command.
Include mean and standard deviation across at least three seeds for the main result.

## Limitations and risks

- The image-level split is not guaranteed to be identity-disjoint.
- CelebA-HQ contains demographic and selection biases inherited from celebrity imagery.
- Synthetic bicubic degradation does not represent every camera, compression, blur, or noise process.
- Pixel losses can oversmooth texture, while generated details may be visually plausible but false.
- Outputs are not evidence of identity, facial attributes, or events.

## Qualitative analysis

Add a fixed, non-cherry-picked panel containing the same test examples for bicubic and SRCNN.
Document failures involving hair, eyes, teeth, accessories, text, occlusion, pose, and background.
