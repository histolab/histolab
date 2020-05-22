# encoding: utf-8

from src.histolab.filters import image_filters as imf
import PIL
import numpy as np

from ..unitutil import (
    function_mock,
    PILImageMock,
)


class DescribeImageFilters(object):
    def it_calls_invert_filter_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_invert = function_mock(
            request, "src.histolab.filters.image_filters_functional.invert"
        )
        F_invert.return_value = image
        invert = imf.Invert()

        invert(image)

        F_invert.assert_called_once_with(image)
        assert type(invert(image)) == PIL.Image.Image

    def it_calls_pil_grayscale(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        grayscale_filter = function_mock(request, "PIL.ImageOps.grayscale")
        grayscale_filter.return_value = image
        grayscale = imf.RgbToGrayscale()

        grayscale(image)

        grayscale_filter.assert_called_once_with(image)
        assert type(grayscale(image)) == PIL.Image.Image

    def it_calls_rgb_to_hed_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_rgb_to_hed = function_mock(
            request, "src.histolab.filters.image_filters_functional.rgb_to_hed"
        )
        F_rgb_to_hed.return_value = image
        rgb_to_hed = imf.RgbToHed()

        rgb_to_hed(image)

        F_rgb_to_hed.assert_called_once_with(image)
        assert type(rgb_to_hed(image)) == PIL.Image.Image

    def it_calls_rgb_to_hsv_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_rgb_to_hsv = function_mock(
            request, "src.histolab.filters.image_filters_functional.rgb_to_hsv"
        )
        F_rgb_to_hsv.return_value = image
        rgb_to_hsv = imf.RgbToHsv()

        rgb_to_hsv(image)

        F_rgb_to_hsv.assert_called_once_with(image)
        assert type(rgb_to_hsv(image)) == PIL.Image.Image

    def it_calls_stretch_contrast_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_stretch_contrast = function_mock(
            request, "src.histolab.filters.image_filters_functional.stretch_contrast"
        )
        F_stretch_contrast.return_value = image
        stretch_contrast = imf.StretchContrast()

        stretch_contrast(image, 0, 100)

        F_stretch_contrast.assert_called_once_with(image, 0, 100)
        assert type(stretch_contrast(image, 0, 100)) == PIL.Image.Image

    def it_calls_histogram_equalization_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_histogram_equalization = function_mock(
            request,
            "src.histolab.filters.image_filters_functional.histogram_equalization",
        )
        F_histogram_equalization.return_value = image
        histogram_equalization = imf.HistogramEqualization()

        histogram_equalization(image, 200)

        F_histogram_equalization.assert_called_once_with(image, 200)
        assert type(histogram_equalization(image, 200)) == PIL.Image.Image

    def it_calls_adaptive_equalization_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_adaptive_equalization = function_mock(
            request,
            "src.histolab.filters.image_filters_functional.adaptive_equalization",
        )
        F_adaptive_equalization.return_value = image
        adaptive_equalization = imf.AdaptiveEqualization()

        adaptive_equalization(image, 250, 0.2)

        F_adaptive_equalization.assert_called_once_with(image, 250, 0.2)
        assert type(adaptive_equalization(image, 250, 0.2)) == PIL.Image.Image

    def it_calls_local_equalization_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_local_equalization = function_mock(
            request, "src.histolab.filters.image_filters_functional.local_equalization",
        )
        F_local_equalization.return_value = image
        local_equalization = imf.LocalEqualization()

        local_equalization(image, 5)

        F_local_equalization.assert_called_once_with(image, 5)
        assert type(local_equalization(image, 5)) == PIL.Image.Image

    def it_calls_kmeans_segmentation_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_kmeans_segmentation = function_mock(
            request,
            "src.histolab.filters.image_filters_functional.kmeans_segmentation",
        )
        F_kmeans_segmentation.return_value = image
        kmeans_segmentation = imf.KmeansSegmentation()

        kmeans_segmentation(image, 5, 400)

        F_kmeans_segmentation.assert_called_once_with(image, 5, 400)
        assert type(kmeans_segmentation(image, 5, 400)) == PIL.Image.Image

    def it_calls_rag_threshold_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_rag_threshold = function_mock(
            request, "src.histolab.filters.image_filters_functional.rag_threshold",
        )
        F_rag_threshold.return_value = image
        rag_threshold = imf.RagThreshold()

        rag_threshold(image, 3, 600, 15)

        F_rag_threshold.assert_called_once_with(image, 3, 600, 15)
        assert type(rag_threshold(image, 3, 600, 15)) == PIL.Image.Image

    def it_applies_hysteresis_threshold(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_hysteresis_threshold = function_mock(
            request,
            "src.histolab.filters.image_filters_functional.hysteresis_threshold",
        )
        F_hysteresis_threshold.return_value = image
        hysteresis_threshold = imf.HysteresisThreshold()

        hysteresis_threshold(image, 20, 150)

        F_hysteresis_threshold.assert_called_once_with(image, 20, 150)
        assert type(hysteresis_threshold(image, 20, 150)) == PIL.Image.Image

    def it_applies_hysteresis_threshold_mask_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_hysteresis_threshold_mask = function_mock(
            request,
            "src.histolab.filters.image_filters_functional.hysteresis_threshold_mask",
        )
        F_hysteresis_threshold_mask.return_value = np.array(image)
        hysteresis_threshold_mask = imf.HysteresisThresholdMask()

        hysteresis_threshold_mask(image, 30, 170)

        F_hysteresis_threshold_mask.assert_called_once_with(image, 30, 170)
        assert type(hysteresis_threshold_mask(image, 30, 170)) == np.ndarray

    def it_calls_otsu_threshold_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_otsu_threshold = function_mock(
            request, "src.histolab.filters.image_filters_functional.otsu_threshold",
        )
        F_otsu_threshold.return_value = np.array(image)
        otsu_threshold = imf.OtsuThreshold()

        otsu_threshold(image)

        F_otsu_threshold.assert_called_once_with(image)
        assert type(otsu_threshold(image)) == np.ndarray

    def it_calls_local_otsu_threshold_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_local_otsu_threshold = function_mock(
            request,
            "src.histolab.filters.image_filters_functional.local_otsu_threshold",
        )
        F_local_otsu_threshold.return_value = np.array(image)
        local_otsu_threshold = imf.LocalOtsuThreshold()

        local_otsu_threshold(image, 5)

        F_local_otsu_threshold.assert_called_once_with(image, 5)
        assert type(local_otsu_threshold(image, 5)) == np.ndarray

    def it_calls_filter_entropy_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_filter_entropy = function_mock(
            request, "src.histolab.filters.image_filters_functional.filter_entropy",
        )
        F_filter_entropy.return_value = np.array(image)
        filter_entropy = imf.FilterEntropy()

        filter_entropy(image, 3, 6)

        F_filter_entropy.assert_called_once_with(image, 3, 6)
        assert type(filter_entropy(image, 3, 6)) == np.ndarray

    def it_calls_canny_edges_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_canny_edges = function_mock(
            request, "src.histolab.filters.image_filters_functional.canny_edges",
        )
        F_canny_edges.return_value = np.array(image)
        canny_edges = imf.CannyEdges()

        canny_edges(image, 0.8, 0.3, 13)

        F_canny_edges.assert_called_once_with(image, 0.8, 0.3, 13)
        assert type(canny_edges(image, 0.8, 0.3, 13)) == np.ndarray

    def it_calls_grays_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_grays = function_mock(
            request, "src.histolab.filters.image_filters_functional.grays",
        )
        F_grays.return_value = np.array(image)
        grays = imf.Grays()

        grays(image, 20)

        F_grays.assert_called_once_with(image, 20)
        assert type(grays(image, 20)) == np.ndarray

    def it_calls_green_channel_filter_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_green_channel_filter = function_mock(
            request,
            "src.histolab.filters.image_filters_functional.green_channel_filter",
        )
        F_green_channel_filter.return_value = np.array(image)
        green_channel_filter = imf.GreenChannelFilter()

        green_channel_filter(image, 250, False, 85.0)

        F_green_channel_filter.assert_called_once_with(image, 250, False, 85.0)
        assert type(green_channel_filter(image, 250, False, 85.0)) == np.ndarray

    def it_calls_red_filter_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_red_filter = function_mock(
            request, "src.histolab.filters.image_filters_functional.red_filter",
        )
        F_red_filter.return_value = np.array(image)
        red_filter = imf.RedFilter()

        red_filter(image, 180, 100, 85)

        F_red_filter.assert_called_once_with(image, 180, 100, 85)
        assert type(red_filter(image, 180, 100, 85)) == np.ndarray

    def it_calls_red_pen_filter_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_red_pen_filter = function_mock(
            request, "src.histolab.filters.image_filters_functional.red_pen_filter",
        )
        F_red_pen_filter.return_value = np.array(image)
        red_pen_filter = imf.RedPenFilter()

        red_pen_filter(image)

        F_red_pen_filter.assert_called_once_with(image)
        assert type(red_pen_filter(image)) == np.ndarray

    def it_calls_green_filter_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_green_filter = function_mock(
            request, "src.histolab.filters.image_filters_functional.green_filter",
        )
        F_green_filter.return_value = np.array(image)
        green_filter = imf.GreenFilter()

        green_filter(image, 150, 160, 140)

        F_green_filter.assert_called_once_with(image, 150, 160, 140)
        assert type(green_filter(image, 150, 160, 140)) == np.ndarray

    def it_calls_green_pen_filter_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_green_pen_filter = function_mock(
            request, "src.histolab.filters.image_filters_functional.green_pen_filter",
        )
        F_green_pen_filter.return_value = np.array(image)
        green_pen_filter = imf.GreenPenFilter()

        green_pen_filter(image)

        F_green_pen_filter.assert_called_once_with(image)
        assert type(green_pen_filter(image)) == np.ndarray

    def it_calls_blue_filter_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_blue_filter = function_mock(
            request, "src.histolab.filters.image_filters_functional.blue_filter",
        )
        F_blue_filter.return_value = np.array(image)
        blue_filter = imf.BlueFilter()

        blue_filter(image, 60, 120, 190)

        F_blue_filter.assert_called_once_with(image, 60, 120, 190)
        assert type(blue_filter(image, 60, 120, 190)) == np.ndarray

    def it_calls_blue_pen_filter_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_blue_pen_filter = function_mock(
            request, "src.histolab.filters.image_filters_functional.blue_pen_filter",
        )
        F_blue_pen_filter.return_value = np.array(image)
        blue_pen_filter = imf.BluePenFilter()

        blue_pen_filter(image)

        F_blue_pen_filter.assert_called_once_with(image)
        assert type(blue_pen_filter(image)) == np.ndarray
