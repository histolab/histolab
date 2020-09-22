import operator

import numpy as np
import pytest
from PIL import ImageChops

import histolab.filters.image_filters_functional as imf

from ..fixtures import GS, RGB, RGBA
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
    assert str(err.value) == "Input image must be RGB."


def test_hematoxylin_channel_filter_with_rgb_image():
    img = RGB.TCGA_LUNG_RGB
    expected_value = load_expectation(
        "pil-images-rgb/tcga-lung-rgb-hematoxylin-channel", type_="png"
    )

    h_channel = imf.hematoxylin_channel(img)

    np.testing.assert_array_almost_equal(np.array(h_channel), np.array(expected_value))
    assert np.unique(np.array(ImageChops.difference(h_channel, expected_value)))[0] == 0


def test_hematoxylin_channel_filter_with_rgba_image():
    img = RGBA.TCGA_LUNG
    expected_value = load_expectation(
        "pil-images-rgba/tcga-lung-hematoxylin-channel", type_="png"
    )

    hematoxylin_img = imf.hematoxylin_channel(img)

    np.testing.assert_array_almost_equal(
        np.array(hematoxylin_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(hematoxylin_img, expected_value)))[0]
        == 0
    )


def test_hematoxylin_channel_raises_exception_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS

    with pytest.raises(ValueError) as err:
        imf.hematoxylin_channel(gs_img)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input image must be RGB/RGBA."


def test_eosin_channel_filter_with_rgb_image():
    img = RGB.TCGA_LUNG_RGB
    expected_value = load_expectation(
        "pil-images-rgb/tcga-lung-rgb-eosin-channel", type_="png"
    )

    eosin_img = imf.eosin_channel(img)

    np.testing.assert_array_almost_equal(np.array(eosin_img), np.array(expected_value))
    assert np.unique(np.array(ImageChops.difference(eosin_img, expected_value)))[0] == 0


def test_eosin_channel_filter_with_rgba_image():
    img = RGBA.TCGA_LUNG
    expected_value = load_expectation(
        "pil-images-rgba/tcga-lung-eosin-channel", type_="png"
    )

    eosin_img = imf.eosin_channel(img)

    np.testing.assert_array_almost_equal(np.array(eosin_img), np.array(expected_value))
    assert np.unique(np.array(ImageChops.difference(eosin_img, expected_value)))[0] == 0


def test_eosin_channel_raises_exception_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS

    with pytest.raises(ValueError) as err:
        imf.eosin_channel(gs_img)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input image must be RGB/RGBA."


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
        (40, 500),
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
    "nbins, clip_limit", ((-10, 340), (-40, -60), (None, 50), (None, None))
)
def test_adaptive_equalization_raises_exception_on_params(nbins, clip_limit):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB

    with pytest.raises(Exception) as err:
        imf.adaptive_equalization(rgba_img, nbins, clip_limit)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Number of histogram bins must be a positive integer"


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

    kmeans_segmentation_img = imf.kmeans_segmentation(rgba_img, 800, 10)

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

    kmeans_segmentation_img = imf.kmeans_segmentation(pil_image, 800, 10)

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

    kmeans_segmentation_img = imf.kmeans_segmentation(gs_img, 800, 10)

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

    rag_threshold_img = imf.rag_threshold(pil_image, 650, 20.6, 9)

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

    rag_threshold_img = imf.rag_threshold(gs_img, 650, 20.6, 15)

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


@pytest.mark.parametrize("low, high", ((None, 50), (-250, None), (None, None)))
def test_hysteresis_threshold_raises_exception_on_thresholds(low, high):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB

    with pytest.raises(Exception) as err:
        imf.hysteresis_threshold(rgba_img, low, high)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "thresholds cannot be None"


@pytest.mark.parametrize(
    "pil_image, expected_image, disk_size",
    (
        (
            GS.DIAGNOSTIC_SLIDE_THUMB_GS,
            "pil-images-gs/diagnostic-slide-thumb-gs1-local-otsu",
            10,
        ),
        (
            GS.DIAGNOSTIC_SLIDE_THUMB_GS,
            "pil-images-gs/diagnostic-slide-thumb-gs2-local-otsu",
            3.8,
        ),
        (
            GS.DIAGNOSTIC_SLIDE_THUMB_GS,
            "pil-images-gs/diagnostic-slide-thumb-gs3-local-otsu",
            0,
        ),
        (
            GS.DIAGNOSTIC_SLIDE_THUMB_GS,
            "pil-images-gs/diagnostic-slide-thumb-gs4-local-otsu",
            np.sqrt(2),
        ),
    ),
)
def test_local_otsu_threshold_filter_on_gs_image(pil_image, expected_image, disk_size):
    expected_value = load_expectation(expected_image, type_="png")

    local_otsu_img = imf.local_otsu_threshold(pil_image, disk_size)

    np.testing.assert_array_almost_equal(
        np.array(local_otsu_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(local_otsu_img, expected_value)))[0]
        == 0
    )


@pytest.mark.parametrize(
    "pil_image, disk_size, expected_exception, expected_message",
    (
        (RGBA.DIAGNOSTIC_SLIDE_THUMB, 6, ValueError, "Input must be 2D."),
        (RGBA.DIAGNOSTIC_SLIDE_THUMB, -10, ValueError, "Input must be 2D."),
        (RGB.DIAGNOSTIC_SLIDE_THUMB_RGB, 10, ValueError, "Input must be 2D."),
        (RGB.DIAGNOSTIC_SLIDE_THUMB_HSV, 0, ValueError, "Input must be 2D."),
        (RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR, -3, ValueError, "Input must be 2D."),
        (
            GS.DIAGNOSTIC_SLIDE_THUMB_GS,
            -10,
            ValueError,
            "Disk size must be a positive number.",
        ),
        (
            GS.DIAGNOSTIC_SLIDE_THUMB_GS,
            None,
            ValueError,
            "Disk size must be a positive number.",
        ),
        (
            GS.DIAGNOSTIC_SLIDE_THUMB_GS,
            np.inf,
            ValueError,
            "Disk size must be a positive number.",
        ),
    ),
)
def test_local_otsu_threshold_raises_right_exceptions(
    pil_image, disk_size, expected_exception, expected_message
):
    with pytest.raises(expected_exception) as err:
        imf.local_otsu_threshold(pil_image, disk_size)

    assert isinstance(err.value, expected_exception)
    assert str(err.value) == expected_message


# -------- Branching function --------


def test_hysteresis_threshold_mask_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "mask-arrays/diagnostic-slide-thumb-hysteresis-threshold-mask", type_="npy"
    )

    hysteresis_threshold_mask = imf.hysteresis_threshold_mask(rgba_img, 30, 200)

    np.testing.assert_array_equal(hysteresis_threshold_mask, expected_value)


@pytest.mark.parametrize("low, high", ((None, 50), (-250, None), (None, None)))
def test_hysteresis_threshold_mask_raises_exception_on_thresholds(low, high):
    rgb_img = RGB.DIAGNOSTIC_SLIDE_THUMB_RGB

    with pytest.raises(Exception) as err:
        imf.hysteresis_threshold_mask(rgb_img, low, high)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "thresholds cannot be None"


@pytest.mark.parametrize(
    "pil_image, expected_array, low, high",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "mask-arrays/diagnostic-slide-thumb-rgb1-hysteresis-threshold-mask",
            50,
            80,
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "mask-arrays/diagnostic-slide-thumb-rgb2-hysteresis-threshold-mask",
            -20,
            -20,
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "mask-arrays/diagnostic-slide-thumb-ycbcr-hysteresis-threshold-mask",
            -20,
            100,
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "mask-arrays/diagnostic-slide-thumb-hsv-hysteresis-threshold-mask",
            30,
            -90,
        ),
    ),
)
def test_hysteresis_threshold_mask_filter_on_rgb_image(
    pil_image, expected_array, low, high
):
    expected_value = load_expectation(expected_array, type_="npy")

    hysteresis_threshold_mask = imf.hysteresis_threshold_mask(pil_image, low, high)

    np.testing.assert_array_almost_equal(hysteresis_threshold_mask, expected_value)


@pytest.mark.parametrize(
    "expected_array, low, high",
    (
        ("mask-arrays/diagnostic-slide-thumb-gs1-hysteresis-threshold-mask", 50, 100),
        ("mask-arrays/diagnostic-slide-thumb-gs2-hysteresis-threshold-mask", -250, 10),
        ("mask-arrays/diagnostic-slide-thumb-gs3-hysteresis-threshold-mask", 40, -20),
        ("mask-arrays/diagnostic-slide-thumb-gs4-hysteresis-threshold-mask", -10, -10),
    ),
)
def test_hysteresis_threshold_mask_filter_on_gs_image(expected_array, low, high):
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(expected_array, type_="npy")

    hysteresis_threshold_mask = imf.hysteresis_threshold_mask(gs_img, low, high)

    np.testing.assert_array_equal(hysteresis_threshold_mask, expected_value)


def test_otsu_threshold_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "mask-arrays/diagnostic-slide-thumb-otsu-threshold-mask", type_="npy"
    )

    otsu_threshold_mask = imf.otsu_threshold(rgba_img, operator.gt)

    np.testing.assert_array_equal(otsu_threshold_mask, expected_value)


@pytest.mark.parametrize(
    "pil_image, expected_array",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "mask-arrays/diagnostic-slide-thumb-gs-otsu-threshold-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "mask-arrays/diagnostic-slide-thumb-ycbcr-otsu-threshold-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "mask-arrays/diagnostic-slide-thumb-hsv-otsu-threshold-mask",
        ),
    ),
)
def test_otsu_threshold_filter_on_rgb_image(pil_image, expected_array):
    expected_value = load_expectation(expected_array, type_="npy")

    otsu_threshold_mask = imf.otsu_threshold(pil_image, operator.gt)

    np.testing.assert_array_equal(otsu_threshold_mask, expected_value)


def test_otsu_threshold_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "mask-arrays/diagnostic-slide-thumb-gs-otsu-threshold-mask", type_="npy"
    )

    otsu_threshold_mask = imf.otsu_threshold(gs_img, operator.gt)

    np.testing.assert_array_equal(otsu_threshold_mask, expected_value)


def test_filter_entropy_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "mask-arrays/diagnostic-slide-thumb-gs-filter-entropy-mask", type_="npy"
    )

    filter_entropy_mask = imf.filter_entropy(gs_img, 8, 4.5)

    np.testing.assert_array_equal(filter_entropy_mask, expected_value)


@pytest.mark.parametrize(
    "pil_image",
    (
        RGBA.DIAGNOSTIC_SLIDE_THUMB,
        RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
        RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
        RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
    ),
)
def test_filter_entropy_raises_exception_on_rgb_image(pil_image):
    with pytest.raises(Exception) as err:
        imf.filter_entropy(pil_image)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input must be 2D."


def test_canny_edges_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "mask-arrays/diagnostic-slide-thumb-gs-canny-edges-mask", type_="npy"
    )

    canny_edges_mask = imf.canny_edges(gs_img, 0.7, 1, 15)

    np.testing.assert_array_equal(canny_edges_mask, expected_value)


@pytest.mark.parametrize(
    "pil_image",
    (
        RGBA.DIAGNOSTIC_SLIDE_THUMB,
        RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
        RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
        RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
    ),
)
def test_canny_edges_raises_exception_on_rgb_image(pil_image):
    with pytest.raises(Exception) as err:
        imf.canny_edges(pil_image)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input must be 2D."


def test_grays_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "mask-arrays/diagnostic-slide-thumb-grays-mask", type_="npy"
    )

    grays_mask = imf.grays(rgba_img, 8)

    np.testing.assert_array_equal(grays_mask, expected_value)


@pytest.mark.parametrize(
    "pil_image, expected_value, threshold",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "mask-arrays/diagnostic-slide-thumb-rgb-grays-mask",
            20,
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "mask-arrays/diagnostic-slide-thumb-hsv-grays-mask",
            0,
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "mask-arrays/diagnostic-slide-thumb-ycbcr-grays-mask",
            -5,
        ),
    ),
)
def test_grays_filter_on_rgb_image(pil_image, expected_value, threshold):
    expected_value = load_expectation(expected_value, type_="npy")

    grays_mask = imf.grays(pil_image, threshold)

    np.testing.assert_array_equal(grays_mask, expected_value)


def test_grays_raises_exception_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS

    with pytest.raises(Exception) as err:
        imf.grays(gs_img, 9)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input must be 3D."


# TODO test recursive green channel filter


@pytest.mark.parametrize(
    "red_thresh, green_thresh, blue_thresh, expected_value",
    (
        (15, 200, 50, "mask-arrays/diagnostic-slide-thumb-rgba1-red-filter-mask"),
        (0, 20, 230, "mask-arrays/diagnostic-slide-thumb-rgba2-red-filter-mask"),
        (140, 160, 0, "mask-arrays/diagnostic-slide-thumb-rgba3-red-filter-mask"),
    ),
)
def test_red_filter_on_rgba_image(
    red_thresh, green_thresh, blue_thresh, expected_value
):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(expected_value, type_="npy")

    red_filter_mask = imf.red_filter(rgba_img, red_thresh, green_thresh, blue_thresh)

    np.testing.assert_array_equal(red_filter_mask, expected_value)


@pytest.mark.parametrize(
    "pil_img, red_thresh, green_thresh, blue_thresh, expected_value",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            15,
            200,
            50,
            "mask-arrays/diagnostic-slide-thumb-rgb-red-filter-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            0,
            20,
            230,
            "mask-arrays/diagnostic-slide-thumb-hsv-red-filter-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            140,
            160,
            0,
            "mask-arrays/diagnostic-slide-thumb-ycbcr-red-filter-mask",
        ),
    ),
)
def test_red_filter_on_rgb_image(
    pil_img, red_thresh, green_thresh, blue_thresh, expected_value
):
    expected_value = load_expectation(expected_value, type_="npy")

    red_filter_mask = imf.red_filter(pil_img, red_thresh, green_thresh, blue_thresh)

    np.testing.assert_array_equal(red_filter_mask, expected_value)


@pytest.mark.parametrize(
    "pil_img, red_thresh, green_thresh, blue_thresh, expected_exception, "
    "expected_message",
    (
        (GS.DIAGNOSTIC_SLIDE_THUMB_GS, 15, 200, 50, ValueError, "Input must be 3D."),
        (GS.DIAGNOSTIC_SLIDE_THUMB_GS, -15, 0, 20, ValueError, "Input must be 3D."),
        (
            RGBA.DIAGNOSTIC_SLIDE_THUMB,
            0,
            -200,
            50,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            0,
            -200,
            50,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            0,
            260,
            50,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            0,
            200,
            500,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            0,
            200,
            -500,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            -20,
            -40,
            -90,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            290,
            30,
            70,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
    ),
)
def test_red_filter_raises_right_exceptions(
    pil_img, red_thresh, green_thresh, blue_thresh, expected_exception, expected_message
):
    with pytest.raises(expected_exception) as err:
        imf.red_filter(pil_img, red_thresh, green_thresh, blue_thresh)

    assert isinstance(err.value, expected_exception)
    assert str(err.value) == expected_message


def test_red_pen_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-red-pen-filter", type_="png"
    )

    red_pen_filter_img = imf.red_pen_filter(rgba_img)

    np.testing.assert_array_equal(red_pen_filter_img, expected_value)


@pytest.mark.parametrize(
    "pil_img, expected_value",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-red-pen-filter",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-red-pen-filter",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-red-pen-filter",
        ),
    ),
)
def test_red_pen_filter_on_rgb_image(pil_img, expected_value):
    expected_value = load_expectation(expected_value, type_="png")

    red_pen_filter_img = imf.red_pen_filter(pil_img)

    np.testing.assert_array_almost_equal(
        np.array(red_pen_filter_img), np.array(expected_value)
    )
    assert (
        np.unique(np.array(ImageChops.difference(red_pen_filter_img, expected_value)))[
            0
        ]
        == 0
    )


@pytest.mark.parametrize(
    "red_thresh, green_thresh, blue_thresh, expectation",
    (
        (20, 190, 50, "mask-arrays/diagnostic-slide-thumb-rgba1-green-filter-mask"),
        (0, 40, 30, "mask-arrays/diagnostic-slide-thumb-rgba2-green-filter-mask"),
        (0, 180, 90, "mask-arrays/diagnostic-slide-thumb-rgba3-green-filter-mask"),
    ),
)
def test_green_filter_on_rgba_image(red_thresh, green_thresh, blue_thresh, expectation):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB

    green_filter_mask = imf.green_filter(
        rgba_img, red_thresh, green_thresh, blue_thresh
    )

    np.testing.assert_array_equal(
        green_filter_mask, load_expectation(expectation, type_="npy")
    )


@pytest.mark.parametrize(
    "green_thresh, avoid_overmask, overmask_thresh, expectation",
    (
        (
            200,
            True,
            50,
            "mask-arrays/diagnostic-slide-thumb-rgba1-green-ch-filter-mask",
        ),
        (0, True, 50, "mask-arrays/diagnostic-slide-thumb-rgba2-green-ch-filter-mask"),
        (
            200,
            False,
            50,
            "mask-arrays/diagnostic-slide-thumb-rgba3-green-ch-filter-mask",
        ),
        (0, False, 50, "mask-arrays/diagnostic-slide-thumb-rgba4-green-ch-filter-mask"),
    ),
)
def test_green_channel_filter_on_rgba_image(
    green_thresh, avoid_overmask, overmask_thresh, expectation
):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB

    green_ch_filter_mask = imf.green_channel_filter(
        rgba_img, green_thresh, avoid_overmask, overmask_thresh
    )

    np.testing.assert_array_equal(
        green_ch_filter_mask, load_expectation(expectation, type_="npy")
    )
    assert rgba_img.size == green_ch_filter_mask.T.shape


@pytest.mark.parametrize("green_threshold", (256, -1))
def test_green_channel_filter_with_wrong_threshold(green_threshold):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    with pytest.raises(ValueError) as err:
        imf.green_channel_filter(rgba_img, green_threshold)

    assert isinstance(err.value, ValueError)
    assert str(err.value) == "threshold must be in range [0, 255]"


@pytest.mark.parametrize(
    "rgb_img, expectation",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "mask-arrays/diagnostic-slide-thumb-rgb-pen-marks-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "mask-arrays/diagnostic-slide-thumb-hsv-pen-marks-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "mask-arrays/diagnostic-slide-thumb-ycbcr-pen-marks-mask",
        ),
        (RGB.TCGA_LUNG_RGB, "mask-arrays/tcga-lung-rgb-pen-marks-mask"),
    ),
)
def test_pen_marks_filter_on_rgb_image(rgb_img, expectation):
    pen_marks_filter = imf.pen_marks(rgb_img)

    np.testing.assert_array_equal(
        pen_marks_filter, load_expectation(expectation, type_="npy")
    )


def test_pen_marks_filter_with_wrong_img_mode():
    with pytest.raises(ValueError) as err:
        imf.pen_marks(RGBA.DIAGNOSTIC_SLIDE_THUMB)

    assert isinstance(err.value, ValueError)
    assert str(err.value) == "Image input must be RGB, got RGBA."


@pytest.mark.parametrize(
    "pil_img, red_thresh, green_thresh, blue_thresh, expected_value",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            15,
            200,
            50,
            "mask-arrays/diagnostic-slide-thumb-rgb-green-filter-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            0,
            20,
            230,
            "mask-arrays/diagnostic-slide-thumb-hsv-green-filter-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            140,
            160,
            0,
            "mask-arrays/diagnostic-slide-thumb-ycbcr-green-filter-mask",
        ),
    ),
)
def test_green_filter_on_rgb_image(
    pil_img, red_thresh, green_thresh, blue_thresh, expected_value
):
    expected_value = load_expectation(expected_value, type_="npy")

    red_filter_mask = imf.green_filter(pil_img, red_thresh, green_thresh, blue_thresh)

    np.testing.assert_array_equal(red_filter_mask, expected_value)


@pytest.mark.parametrize(
    "pil_img, red_thresh, green_thresh, blue_thresh, expected_exception, "
    "expected_message",
    (
        (GS.DIAGNOSTIC_SLIDE_THUMB_GS, 15, 200, 50, ValueError, "Input must be 3D."),
        (GS.DIAGNOSTIC_SLIDE_THUMB_GS, -15, 0, 20, ValueError, "Input must be 3D."),
        (
            RGBA.DIAGNOSTIC_SLIDE_THUMB,
            0,
            -200,
            50,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            0,
            -200,
            50,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            0,
            260,
            50,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            0,
            200,
            500,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            0,
            200,
            -500,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            -20,
            -40,
            -90,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            290,
            30,
            70,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
    ),
)
def test_green_filter_raises_right_exceptions(
    pil_img, red_thresh, green_thresh, blue_thresh, expected_exception, expected_message
):
    with pytest.raises(expected_exception) as err:
        imf.green_filter(pil_img, red_thresh, green_thresh, blue_thresh)

    assert isinstance(err.value, expected_exception)
    assert str(err.value) == expected_message


def test_green_pen_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-green-pen-filter", type_="png"
    )

    green_pen_filter_img = imf.green_pen_filter(rgba_img)

    np.testing.assert_array_equal(green_pen_filter_img, expected_value)


@pytest.mark.parametrize(
    "pil_img, expected_value",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-green-pen-filter",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-green-pen-filter",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-green-pen-filter",
        ),
    ),
)
def test_green_pen_filter_on_rgb_image(pil_img, expected_value):
    expected_value = load_expectation(expected_value, type_="png")

    green_pen_filter_img = imf.green_pen_filter(pil_img)

    np.testing.assert_array_equal(green_pen_filter_img, expected_value)


@pytest.mark.parametrize(
    "red_thresh, green_thresh, blue_thresh, expected_value",
    (
        (15, 200, 50, "mask-arrays/diagnostic-slide-thumb-rgba1-blue-filter-mask"),
        (0, 20, 230, "mask-arrays/diagnostic-slide-thumb-rgba2-blue-filter-mask"),
        (140, 160, 0, "mask-arrays/diagnostic-slide-thumb-rgba3-blue-filter-mask"),
    ),
)
def test_blue_filter_on_rgba_image(
    red_thresh, green_thresh, blue_thresh, expected_value
):
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(expected_value, type_="npy")

    blue_filter_mask = imf.blue_filter(rgba_img, red_thresh, green_thresh, blue_thresh)

    np.testing.assert_array_equal(blue_filter_mask, expected_value)


@pytest.mark.parametrize(
    "pil_img, red_thresh, green_thresh, blue_thresh, expected_value",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            15,
            200,
            50,
            "mask-arrays/diagnostic-slide-thumb-rgb-blue-filter-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            0,
            20,
            230,
            "mask-arrays/diagnostic-slide-thumb-hsv-blue-filter-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            140,
            160,
            230,
            "mask-arrays/diagnostic-slide-thumb-ycbcr-blue-filter-mask",
        ),
    ),
)
def test_blue_filter_on_rgb_image(
    pil_img, red_thresh, green_thresh, blue_thresh, expected_value
):
    expected_value = load_expectation(expected_value, type_="npy")

    blue_filter_mask = imf.blue_filter(pil_img, red_thresh, green_thresh, blue_thresh)

    np.testing.assert_array_equal(blue_filter_mask, expected_value)


@pytest.mark.parametrize(
    "pil_img, red_thresh, green_thresh, blue_thresh, expected_exception, "
    "expected_message",
    (
        (GS.DIAGNOSTIC_SLIDE_THUMB_GS, 15, 200, 50, ValueError, "Input must be 3D."),
        (GS.DIAGNOSTIC_SLIDE_THUMB_GS, -15, 0, 20, ValueError, "Input must be 3D."),
        (
            RGBA.DIAGNOSTIC_SLIDE_THUMB,
            0,
            -200,
            50,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            0,
            -200,
            50,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            0,
            260,
            50,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            0,
            200,
            500,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            0,
            200,
            -500,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            -20,
            -40,
            -90,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            290,
            30,
            70,
            ValueError,
            "RGB Thresholds must be in range [0, 255]",
        ),
    ),
)
def test_blue_filter_raises_right_exceptions(
    pil_img, red_thresh, green_thresh, blue_thresh, expected_exception, expected_message
):
    with pytest.raises(expected_exception) as err:
        imf.blue_filter(pil_img, red_thresh, green_thresh, blue_thresh)

    assert isinstance(err.value, expected_exception)
    assert str(err.value) == expected_message


def test_blue_pen_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "pil-images-rgba/diagnostic-slide-thumb-blue-pen-filter", type_="png"
    )

    blue_pen_filter_img = imf.blue_pen_filter(rgba_img)

    np.testing.assert_array_equal(blue_pen_filter_img, expected_value)


@pytest.mark.parametrize(
    "pil_img, expected_value",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-blue-pen-filter",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-blue-pen-filter",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-blue-pen-filter",
        ),
    ),
)
def test_blue_pen_filter_on_rgb_image(pil_img, expected_value):
    expected_value = load_expectation(expected_value, type_="png")

    blue_pen_filter_img = imf.blue_pen_filter(pil_img)

    np.testing.assert_array_equal(blue_pen_filter_img, expected_value)


# TODO: manage general pen marks


def test_yen_threshold_filter_on_rgba_image():
    rgba_img = RGBA.DIAGNOSTIC_SLIDE_THUMB
    expected_value = load_expectation(
        "mask-arrays/diagnostic-slide-thumb-yen-threshold-mask", type_="npy"
    )

    yen_threshold_mask = imf.yen_threshold(rgba_img)

    np.testing.assert_array_equal(yen_threshold_mask, expected_value)


@pytest.mark.parametrize(
    "pil_image, expected_array",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "mask-arrays/diagnostic-slide-thumb-rgb-yen-threshold-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "mask-arrays/diagnostic-slide-thumb-ycbcr-yen-threshold-mask",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "mask-arrays/diagnostic-slide-thumb-hsv-yen-threshold-mask",
        ),
    ),
)
def test_yen_threshold_filter_on_rgb_image(pil_image, expected_array):
    expected_value = load_expectation(expected_array, type_="npy")

    yen_threshold_mask = imf.yen_threshold(pil_image)

    np.testing.assert_array_equal(yen_threshold_mask, expected_value)


def test_yen_threshold_filter_on_gs_image():
    gs_img = GS.DIAGNOSTIC_SLIDE_THUMB_GS
    expected_value = load_expectation(
        "mask-arrays/diagnostic-slide-thumb-gs-yen-threshold-mask", type_="npy"
    )

    yen_threshold_mask = imf.yen_threshold(gs_img)

    np.testing.assert_array_equal(yen_threshold_mask, expected_value)


@pytest.mark.parametrize(
    "pil_image, expected_image",
    (
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_RGB,
            "pil-images-rgb/diagnostic-slide-thumb-rgb-to-lab",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_HSV,
            "pil-images-rgb/diagnostic-slide-thumb-hsv-rgb-to-lab",
        ),
        (
            RGB.DIAGNOSTIC_SLIDE_THUMB_YCBCR,
            "pil-images-rgb/diagnostic-slide-thumb-ycbcr-rgb-to-lab",
        ),
    ),
)
def test_rgb_to_lab_filter_with_rgb_image(pil_image, expected_image):
    expected_value = load_expectation(expected_image, type_="png")

    lab_img = imf.rgb_to_lab(pil_image)

    np.testing.assert_array_almost_equal(np.array(lab_img), np.array(expected_value))
    assert np.unique(np.array(ImageChops.difference(lab_img, expected_value)))[0] == 0


@pytest.mark.parametrize(
    "pil_image", (RGBA.DIAGNOSTIC_SLIDE_THUMB, GS.DIAGNOSTIC_SLIDE_THUMB_GS),
)
def test_rgb_to_lab_raises_exception_on_gs_and_rgba_image(pil_image):
    with pytest.raises(Exception) as err:
        imf.rgb_to_lab(pil_image)

    assert isinstance(err.value, Exception)
    assert str(err.value) == "Input image must be RGB"
