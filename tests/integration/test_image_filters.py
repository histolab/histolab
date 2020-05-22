import src.histolab.filters.image_filters_functional as imf
from ..fixtures import RGBA, RGB, GS
from ..util import load_expectation
import numpy as np
from PIL import ImageChops
import pytest


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-inverted",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-inverted",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-inverted",
        ),
    ),
)
def test_invert_filter_with_rgb_image(pil_image, expected_image):
    inverted_img = imf.invert(pil_image)
    expected_value = load_expectation(expected_image, type_="png")

    np.testing.assert_array_almost_equal(
        np.array(inverted_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(inverted_img, expected_value)))[0] == 0
    )


def test_invert_filter_with_rgba_image():
    inverted_img = imf.invert(RGBA.DIAGNOSTIC_SLIDE_THUMB)
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-inverted", type_="png"
    )

    np.testing.assert_array_almost_equal(
        np.array(inverted_img), np.array(expected_value)
    )

    assert (
        np.unique(np.array(ImageChops.difference(inverted_img, expected_value)))[0] == 0
    )


def test_invert_filter_with_gs_image():
    inverted_img = imf.invert(GS.DIAGNOSTIC_SLIDE_THUMB_GS)
    expected_value = load_expectation(
        "pil-images-gs/diagnostic-slide-thumb-gs-inverted", type_="png"
    )

    np.testing.assert_array_almost_equal(
        np.array(inverted_img), np.array(expected_value)
    )

    assert (
        np.unique(np.array(ImageChops.difference(inverted_img, expected_value)))[0] == 0
    )


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-to-hed",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-rgb-to-hed",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-rgb-to-hed",
        ),
    ),
)
def test_rgb_to_hed_filter_with_rgb_image(pil_image, expected_image):
    hed_img = imf.rgb_to_hed(pil_image)
    expected_value = load_expectation(expected_image, type_="png")

    np.testing.assert_array_almost_equal(np.array(hed_img), np.array(expected_value))

    assert np.unique(np.array(ImageChops.difference(hed_img, expected_value)))[0] == 0


def test_rgb_to_hed_filter_with_rgba_image():
    hed_img = imf.rgb_to_hed(RGBA.DIAGNOSTIC_SLIDE_THUMB)
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-rgba-to-hed", type_="png"
    )

    np.testing.assert_array_almost_equal(np.array(hed_img), np.array(expected_value))

    assert np.unique(np.array(ImageChops.difference(hed_img, expected_value)))[0] == 0


def test_rgb_to_hed_raises_exception_on_gs_image():
    grayscale_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    with pytest.raises(Exception) as err:
        imf.rgb_to_hed(grayscale_img)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input image must be RGB or RGBA"


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-to-hsv",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-rgb-to-hsv",
        ),
    ),
)
def test_rgb_to_hsv_filter_with_rgb_image(pil_image, expected_image):
    hsv_img = imf.rgb_to_hsv(pil_image)
    expected_value = load_expectation(expected_image, type_="png")

    np.testing.assert_array_almost_equal(np.array(hsv_img), np.array(expected_value))

    assert np.unique(np.array(ImageChops.difference(hsv_img, expected_value)))[0] == 0


def test_rgb_to_hsv_raises_exception_on_rgba_image():
    grayscale_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    with pytest.raises(Exception) as err:
        imf.rgb_to_hsv(grayscale_img)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input image must be RGB"


def test_rgb_to_hsv_raises_exception_on_gs_image():
    grayscale_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    with pytest.raises(Exception) as err:
        imf.rgb_to_hsv(grayscale_img)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input image must be RGB"


def test_stretch_contrast_filter_on_rgba_image():
    hed_img = imf.stretch_contrast(RGBA.DIAGNOSTIC_SLIDE_THUMB, 40, 60)
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-stretch-contrast", type_="png"
    )

    np.testing.assert_array_almost_equal(np.array(hed_img), np.array(expected_value))

    assert np.unique(np.array(ImageChops.difference(hed_img, expected_value)))[0] == 0


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-stretch-contrast",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-stretch-contrast",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-stretch-contrast",
        ),
    ),
)
def test_stretch_contrast_filter_on_rgb_image(pil_image, expected_image):
    stretch_img = imf.stretch_contrast(pil_image)
    expected_value = load_expectation(expected_image, type_="png")

    np.testing.assert_array_almost_equal(
        np.array(stretch_img), np.array(expected_value)
    )

    assert (
        np.unique(np.array(ImageChops.difference(stretch_img, expected_value)))[0] == 0
    )


@pytest.mark.parametrize(
    "low, high", ((300, 40,), (300, 600,), (40, 500,),),
)
def test_stretch_contrast_raises_exception_on_ranges(low, high):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    with pytest.raises(Exception) as err:
        imf.stretch_contrast(rgba_img, low, high)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "low and high values must be in range [0, 255]"
