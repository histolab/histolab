import pytest
import numpy as np

import src.histolab.filters.image_filters_functional as imf

from PIL import ImageChops

from ..fixtures import RGBA, RGB, GS
from ..util import load_expectation


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
    expected_value = load_expectation(expected_image, type_="png")

    inverted_img = imf.invert(pil_image)

    np.testing.assert_array_almost_equal(
        np.array(inverted_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(inverted_img, expected_value)))[0] == 0
    )


def test_invert_filter_with_rgba_image():
    rgba_image = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-inverted", type_="png"
    )

    inverted_img = imf.invert(rgba_image)

    np.testing.assert_array_almost_equal(
        np.array(inverted_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(inverted_img, expected_value)))[0] == 0
    )


def test_invert_filter_with_gs_image():
    gs_image = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "pil-images-gs/diagnostic-slide-thumb-gs-inverted", type_="png"
    )

    inverted_img = imf.invert(gs_image)

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
    expected_value = load_expectation(expected_image, type_="png")

    hed_img = imf.rgb_to_hed(pil_image)

    np.testing.assert_array_almost_equal(np.array(hed_img), np.array(expected_value))
    assert np.unique(np.array(ImageChops.difference(hed_img, expected_value)))[0] == 0


def test_rgb_to_hed_filter_with_rgba_image():
    img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-rgba-to-hed", type_="png"
    )

    hed_img = imf.rgb_to_hed(img)

    np.testing.assert_array_almost_equal(np.array(hed_img), np.array(expected_value))
    assert np.unique(np.array(ImageChops.difference(hed_img, expected_value)))[0] == 0


def test_rgb_to_hed_raises_exception_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS

    with pytest.raises(Exception) as err:
        imf.rgb_to_hed(gs_img)

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
    expected_value = load_expectation(expected_image, type_="png")

    hsv_img = imf.rgb_to_hsv(pil_image)

    np.testing.assert_array_almost_equal(np.array(hsv_img), np.array(expected_value))
    assert np.unique(np.array(ImageChops.difference(hsv_img, expected_value)))[0] == 0


def test_rgb_to_hsv_raises_exception_on_rgba_image():
    gs_img = RGBA.DIAGNOSTIC_SLIDE_THUMB

    with pytest.raises(Exception) as err:
        imf.rgb_to_hsv(gs_img)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input image must be RGB"


def test_rgb_to_hsv_raises_exception_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS

    with pytest.raises(Exception) as err:
        imf.rgb_to_hsv(gs_img)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input image must be RGB"


def test_stretch_contrast_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-stretch-contrast", type_="png"
    )

    stretched_img = imf.stretch_contrast(rgba_img, 40, 60)

    np.testing.assert_array_almost_equal(
        np.array(stretched_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(stretched_img, expected_value)))[0]
        == 0
    )


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
    expected_value = load_expectation(expected_image, type_="png")

    stretched_img = imf.stretch_contrast(pil_image)

    np.testing.assert_array_almost_equal(
        np.array(stretched_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(stretched_img, expected_value)))[0]
        == 0
    )


def test_stretch_contrast_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "pil-images-gs/diagnostic-slide-thumb-gs-stretch-contrast", type_="png"
    )

    stretched_img = imf.stretch_contrast(gs_img, 40, 60)

    np.testing.assert_array_almost_equal(
        np.array(stretched_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(stretched_img, expected_value)))[0]
        == 0
    )


@pytest.mark.parametrize(
    "low, high",
    (
        (300, 40),
        (300, 600),
        (40, 500,),
        (-10, 340),
        (-200, 300),
        (-40, -60),
        (None, 50),
        (50, None),
        (None, None),
    ),
)
def test_stretch_contrast_raises_exception_on_ranges(low, high):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB

    with pytest.raises(Exception) as err:
        imf.stretch_contrast(rgba_img, low, high)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "low and high values must be in range [0, 255]"


def test_histogram_equalization_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-histogram-equalization", type_="png"
    )

    hist_equ_img = imf.histogram_equalization(rgba_img, 200)

    np.testing.assert_array_almost_equal(
        np.array(hist_equ_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(hist_equ_img, expected_value)))[0] == 0
    )


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-histogram-equalization",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-histogram-equalization",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-histogram-equalization",
        ),
    ),
)
def test_histogram_equalization_filter_on_rgb_image(pil_image, expected_image):
    expected_value = load_expectation(expected_image, type_="png")

    hist_equ_img = imf.histogram_equalization(pil_image, 200)

    np.testing.assert_array_almost_equal(
        np.array(hist_equ_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(hist_equ_img, expected_value)))[0] == 0
    )


def test_histogram_equalization_filter_on_gs_image():
    img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "pil-images-gs/diagnostic-slide-thumb-gs-histogram-equalization", type_="png"
    )

    hist_equ_img = imf.histogram_equalization(img, 200)

    np.testing.assert_array_almost_equal(
        np.array(hist_equ_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(hist_equ_img, expected_value)))[0] == 0
    )


def test_adaptive_equalization_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-adaptive-equalization", type_="png"
    )

    adap_equ_img = imf.adaptive_equalization(rgba_img, 200, 0.2)

    np.testing.assert_array_almost_equal(
        np.array(adap_equ_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(adap_equ_img, expected_value)))[0] == 0
    )


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-adaptive-equalization",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-adaptive-equalization",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-adaptive-equalization",
        ),
    ),
)
def test_adaptive_equalization_filter_on_rgb_image(pil_image, expected_image):
    expected_value = load_expectation(expected_image, type_="png")

    adap_equ_img = imf.adaptive_equalization(pil_image, 200, 0.2)

    np.testing.assert_array_almost_equal(
        np.array(adap_equ_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(adap_equ_img, expected_value)))[0] == 0
    )


def test_adaptive_equalization_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "pil-images-gs/diagnostic-slide-thumb-gs-adaptive-equalization", type_="png"
    )

    adap_equ_img = imf.adaptive_equalization(gs_img, 200, 0.2)

    np.testing.assert_array_almost_equal(
        np.array(adap_equ_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(adap_equ_img, expected_value)))[0] == 0
    )


@pytest.mark.parametrize(
    "nbins, clip_limit", ((-10, 340), (-40, -60), (None, 50), (None, None),),
)
def test_adaptive_equalization_raises_exception_on_params(nbins, clip_limit):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB

    with pytest.raises(Exception) as err:
        imf.adaptive_equalization(rgba_img, nbins, clip_limit)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Number of histogram bins must be positive integer"


def test_local_equalization_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "pil-images-gs/diagnostic-slide-thumb-gs-local-equalization", type_="png"
    )

    local_equ_img = imf.local_equalization(gs_img, 80)

    np.testing.assert_array_almost_equal(
        np.array(local_equ_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(local_equ_img, expected_value)))[0]
        == 0
    )


@pytest.mark.parametrize(
    "pil_rgb_image",
    (
        RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
        RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
        RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
        RGBA.DIAGNOSTIC_SLIDE_THUMB,
    ),
)
def test_local_equalization_raises_exception_on_rgb_images(pil_rgb_image):

    with pytest.raises(Exception) as err:
        imf.local_equalization(pil_rgb_image, 80)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input must be 2D."


def test_kmeans_segmentation_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-kmeans-segmentation", type_="png"
    )

    kmeans_segmentation_img = imf.kmeans_segmentation(rgba_img, 20.6, 300)

    np.testing.assert_array_almost_equal(
        np.array(kmeans_segmentation_img), np.array(expected_value)
    )
    assert (
        np.unique(
            np.array(ImageChops.difference(kmeans_segmentation_img, expected_value))
        )[0]
        == 0
    )


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-kmeans-segmentation",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-kmeans-segmentation",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-kmeans-segmentation",
        ),
    ),
)
def test_kmeans_segmentation_filter_on_rgb_image(pil_image, expected_image):
    expected_value = load_expectation(expected_image, type_="png")

    kmeans_segmentation_img = imf.kmeans_segmentation(pil_image, 20.6, 300)

    np.testing.assert_array_almost_equal(
        np.array(kmeans_segmentation_img), np.array(expected_value)
    )
    assert (
        np.unique(
            np.array(ImageChops.difference(kmeans_segmentation_img, expected_value))
        )[0]
        == 0
    )


def test_kmeans_segmentation_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "pil-images-gs/diagnostic-slide-thumb-gs-kmeans-segmentation", type_="png"
    )

    kmeans_segmentation_img = imf.kmeans_segmentation(gs_img, 20.6, 300)

    np.testing.assert_array_almost_equal(
        np.array(kmeans_segmentation_img), np.array(expected_value)
    )
    assert (
        np.unique(
            np.array(ImageChops.difference(kmeans_segmentation_img, expected_value))
        )[0]
        == 0
    )


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-rag-threshold",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-rag-threshold",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-rag-threshold",
        ),
    ),
)
def test_rag_threshold_filter_on_rgb_image(pil_image, expected_image):
    expected_value = load_expectation(expected_image, type_="png")

    rag_threshold_img = imf.rag_threshold(pil_image, 20.6, 650, 15)

    np.testing.assert_array_almost_equal(
        np.array(rag_threshold_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(rag_threshold_img, expected_value)))[0]
        == 0
    )


def test_rag_threshold_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "pil-images-gs/diagnostic-slide-thumb-gs-rag-threshold", type_="png"
    )

    rag_threshold_img = imf.rag_threshold(gs_img, 20.6, 650, 15)

    np.testing.assert_array_almost_equal(
        np.array(rag_threshold_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(rag_threshold_img, expected_value)))[0]
        == 0
    )


def test_rag_threshold_raises_exception_on_rgba_images():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB

    with pytest.raises(Exception) as err:
        imf.rag_threshold(rgba_img, 20, 50, 3.5)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input image cannot be RGBA"


def test_hysteresis_threshold_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-hysteresis-threshold", type_="png"
    )

    hysteresis_threshold_img = imf.hysteresis_threshold(rgba_img, 10.6, 200)

    np.testing.assert_array_almost_equal(
        np.array(hysteresis_threshold_img), np.array(expected_value)
    )
    assert (
        np.unique(
            np.array(ImageChops.difference(hysteresis_threshold_img, expected_value))
        )[0]
        == 0
    )


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-hysteresis-threshold",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-hysteresis-threshold",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-hysteresis-threshold",
        ),
    ),
)
def test_hysteresis_threshold_filter_on_rgb_image(pil_image, expected_image):
    expected_value = load_expectation(expected_image, type_="png")

    hysteresis_threshold_img = imf.hysteresis_threshold(pil_image, 10.6, 200)

    np.testing.assert_array_almost_equal(
        np.array(hysteresis_threshold_img), np.array(expected_value)
    )
    assert (
        np.unique(
            np.array(ImageChops.difference(hysteresis_threshold_img, expected_value))
        )[0]
        == 0
    )


def test_hysteresis_threshold_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "pil-images-gs/diagnostic-slide-thumb-gs-hysteresis-threshold", type_="png"
    )

    hysteresis_threshold_img = imf.hysteresis_threshold(gs_img, 10.6, 200)

    np.testing.assert_array_almost_equal(
        np.array(hysteresis_threshold_img), np.array(expected_value)
    )
    assert (
        np.unique(
            np.array(ImageChops.difference(hysteresis_threshold_img, expected_value))
        )[0]
        == 0
    )


@pytest.mark.parametrize(
    "low, high", ((None, 50), (-250, None), (None, None),),
)
def test_hysteresis_threshold_raises_exception_on_thresholds(low, high):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB

    with pytest.raises(Exception) as err:
        imf.hysteresis_threshold(rgba_img, low, high)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "thresholds cannot be None"
