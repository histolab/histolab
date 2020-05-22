from src.histolab.filters.image_filters_functional import invert
from ..fixtures import RGBA
from ..util import load_expectation
import numpy as np
from PIL import ImageChops
import pytest


@pytest.mark.parametrize(
    "pil_image, expectation_image",
    (
        (
            RGBA.DIAGNOSTIC_SLIDE_THUMB,
            "pil-images-rgba/diagnostic-slide-thumb-inverted",
        ),
    ),
)
def test_invert_filter_with_rgb_hsv_image(pil_image, expectation_image):
    inverted_img = invert(pil_image)
    expected_value = load_expectation(expectation_image, type_="png")

    np.testing.assert_array_almost_equal(
        np.array(inverted_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(inverted_img, expected_value)))[0] == 0
    )


def test_invert_filter_with_rgba_image():
    inverted_img = invert(RGBA.DIAGNOSTIC_SLIDE_THUMB)
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-inverted", type_="png"
    )

    np.testing.assert_array_almost_equal(
        np.array(inverted_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(inverted_img, expected_value)))[0] == 0
    )
