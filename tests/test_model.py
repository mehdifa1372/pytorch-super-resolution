import pytest
import torch

from super_resolution import SRCNN


def test_model_preserves_image_shape():
    model = SRCNN()
    image = torch.rand(2, 3, 24, 31)
    assert model(image).shape == image.shape


def test_model_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="expected 3 channels"):
        SRCNN()(torch.rand(1, 1, 16, 16))


def test_model_configuration_is_checkpoint_ready():
    assert SRCNN(hidden_channels=(16, 8)).configuration() == {
        "in_channels": 3,
        "hidden_channels": [16, 8],
    }
