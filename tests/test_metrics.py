import torch

from super_resolution import peak_signal_to_noise_ratio, structural_similarity


def test_identical_images_have_high_psnr_and_unit_ssim():
    image = torch.rand(2, 3, 20, 20)
    assert peak_signal_to_noise_ratio(image, image) > 100
    assert torch.allclose(structural_similarity(image, image), torch.tensor(1.0), atol=1e-5)


def test_noise_reduces_quality_metrics():
    target = torch.full((1, 3, 16, 16), 0.5)
    prediction = target + 0.1
    assert peak_signal_to_noise_ratio(prediction, target) < 30
    assert structural_similarity(prediction, target) < 1

