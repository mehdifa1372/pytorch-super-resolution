# CelebA-HQ dataset protocol

## Scope

CelebA-HQ is a high-quality derivative of CelebA with 30,000 aligned face images at 1024×1024.
This repository uses it only as a non-commercial research benchmark for image reconstruction.

## Access and redistribution

1. Review the official CelebA agreement and the CelebA-HQ preparation instructions.
2. Obtain the source data through an authorized channel and prepare it locally.
3. Do not commit, upload, mirror, or redistribute images or derived image data through this repo.
4. Do not publish a trained checkpoint until you have verified that its distribution and intended
   use comply with all applicable dataset terms.

CelebA's official agreement restricts the data to non-commercial research, prohibits commercial
exploitation, and prohibits publishing or distributing the dataset. The images were collected
from the internet and are not owned by the dataset maintainers. Consult the source terms rather
than treating this summary as legal advice:

- CelebA project and agreement: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- CelebA-HQ preparation source: https://github.com/tkarras/progressive_growing_of_gans
- TensorFlow Datasets notes (manual preparation required):
  https://www.tensorflow.org/datasets/catalog/celeb_a_hq

## Reproducible split

`scripts/prepare_celeba_hq.py` discovers image files recursively and orders them by a SHA-256
score derived from the seed and root-relative path. With seed 42, the default split is 80% train,
10% validation, and 10% test. It writes paths rather than copying images.

The split is image-disjoint but not guaranteed to be identity-disjoint. Identity overlap can make
face-domain generalization appear stronger than it is. Any identity-generalization claim requires
a separately documented identity mapping and group-disjoint split.

Keep generated manifests with experiment artifacts and record their SHA-256 hashes. Do not edit a
manifest after training begins.

## Bias, privacy, and evaluation

Celebrity datasets are not representative samples of the public. Appearance, age, skin tone,
gender presentation, styling, pose, lighting, image quality, and geographic context may be
unevenly distributed. Aggregate PSNR and SSIM can conceal subgroup or condition-specific failure.

Do not infer sensitive attributes. If subgroup analysis is scientifically necessary, define the
question and annotation provenance in advance, report uncertainty, avoid harmful labels, and do
not release identifiable examples beyond what the dataset terms permit.

## Required citations

```bibtex
@inproceedings{liu2015faceattributes,
  title={Deep Learning Face Attributes in the Wild},
  author={Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2015}
}

@inproceedings{karras2018progressive,
  title={Progressive Growing of GANs for Improved Quality, Stability, and Variation},
  author={Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```
