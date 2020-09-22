# encoding: utf-8

import operator

import numpy as np
import PIL

from histolab.filters import image_filters as imf

from ...unitutil import PILIMG, NpArrayMock, function_mock


class DescribeImageFilters:
    def it_calls_invert_filter_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_invert = function_mock(
            request, "histolab.filters.image_filters_functional.invert"
        )
        F_invert.return_value = image
        invert = imf.Invert()

        invert(image)

        F_invert.assert_called_once_with(image)
        assert type(invert(image)) == PIL.Image.Image

    def it_calls_pil_grayscale(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        grayscale_filter = function_mock(request, "PIL.ImageOps.grayscale")
        grayscale_filter.return_value = image
        grayscale = imf.RgbToGrayscale()

        grayscale(image)

        grayscale_filter.assert_called_once_with(image)
        assert type(grayscale(image)) == PIL.Image.Image

    def it_calls_rgb_to_hed_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_rgb_to_hed = function_mock(
            request, "histolab.filters.image_filters_functional.rgb_to_hed"
        )
        F_rgb_to_hed.return_value = image
        rgb_to_hed = imf.RgbToHed()

        rgb_to_hed(image)

        F_rgb_to_hed.assert_called_once_with(image)
        assert type(rgb_to_hed(image)) == PIL.Image.Image

    def it_calls_hematoxylin_channel_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_hematoxylin_channel = function_mock(
            request, "histolab.filters.image_filters_functional.hematoxylin_channel"
        )
        F_hematoxylin_channel.return_value = image
        hematoxylin_channel = imf.HematoxylinChannel()

        hematoxylin_channel(image)

        F_hematoxylin_channel.assert_called_once_with(image)
        assert type(hematoxylin_channel(image)) == PIL.Image.Image

    def it_calls_eosin_channel_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_eosin_channel = function_mock(
            request, "histolab.filters.image_filters_functional.eosin_channel"
        )
        F_eosin_channel.return_value = image
        eosin_channel = imf.EosinChannel()

        eosin_channel(image)

        F_eosin_channel.assert_called_once_with(image)
        assert type(eosin_channel(image)) == PIL.Image.Image

    def it_calls_rgb_to_hsv_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_rgb_to_hsv = function_mock(
            request, "histolab.filters.image_filters_functional.rgb_to_hsv"
        )
        F_rgb_to_hsv.return_value = image
        rgb_to_hsv = imf.RgbToHsv()

        rgb_to_hsv(image)

        F_rgb_to_hsv.assert_called_once_with(image)
        assert type(rgb_to_hsv(image)) == PIL.Image.Image

    def it_calls_stretch_contrast_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_stretch_contrast = function_mock(
            request, "histolab.filters.image_filters_functional.stretch_contrast"
        )
        F_stretch_contrast.return_value = image
        stretch_contrast = imf.StretchContrast(0, 100)

        stretch_contrast(image)

        F_stretch_contrast.assert_called_once_with(image, 0, 100)
        assert type(stretch_contrast(image)) == PIL.Image.Image

    def it_calls_histogram_equalization_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_histogram_equalization = function_mock(
            request, "histolab.filters.image_filters_functional.histogram_equalization"
        )
        F_histogram_equalization.return_value = image
        histogram_equalization = imf.HistogramEqualization(200)

        histogram_equalization(image)

        F_histogram_equalization.assert_called_once_with(image, 200)
        assert type(histogram_equalization(image)) == PIL.Image.Image

    def it_calls_adaptive_equalization_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_adaptive_equalization = function_mock(
            request, "histolab.filters.image_filters_functional.adaptive_equalization"
        )
        F_adaptive_equalization.return_value = image
        adaptive_equalization = imf.AdaptiveEqualization(250, 0.2)

        adaptive_equalization(image)

        F_adaptive_equalization.assert_called_once_with(image, 250, 0.2)
        assert type(adaptive_equalization(image)) == PIL.Image.Image

    def it_calls_local_equalization_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_local_equalization = function_mock(
            request, "histolab.filters.image_filters_functional.local_equalization"
        )
        F_local_equalization.return_value = image
        local_equalization = imf.LocalEqualization(5)

        local_equalization(image)

        F_local_equalization.assert_called_once_with(image, 5)
        assert type(local_equalization(image)) == PIL.Image.Image

    def it_calls_kmeans_segmentation_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_kmeans_segmentation = function_mock(
            request, "histolab.filters.image_filters_functional.kmeans_segmentation"
        )
        F_kmeans_segmentation.return_value = image
        kmeans_segmentation = imf.KmeansSegmentation(5, 400)

        kmeans_segmentation(image)

        F_kmeans_segmentation.assert_called_once_with(image, 5, 400)
        assert type(kmeans_segmentation(image)) == PIL.Image.Image

    def it_calls_rag_threshold_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_rag_threshold = function_mock(
            request, "histolab.filters.image_filters_functional.rag_threshold"
        )
        F_rag_threshold.return_value = image
        rag_threshold = imf.RagThreshold(3, 600, 15)

        rag_threshold(image)

        F_rag_threshold.assert_called_once_with(image, 3, 600, 15)
        assert type(rag_threshold(image)) == PIL.Image.Image

    def it_applies_hysteresis_threshold(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_hysteresis_threshold = function_mock(
            request, "histolab.filters.image_filters_functional.hysteresis_threshold"
        )
        F_hysteresis_threshold.return_value = image
        hysteresis_threshold = imf.HysteresisThreshold(20, 150)

        hysteresis_threshold(image)

        F_hysteresis_threshold.assert_called_once_with(image, 20, 150)
        assert type(hysteresis_threshold(image)) == PIL.Image.Image

    def it_applies_hysteresis_threshold_mask_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_hysteresis_threshold_mask = function_mock(
            request,
            "histolab.filters.image_filters_functional.hysteresis_threshold_mask",
        )
        F_hysteresis_threshold_mask.return_value = np.array(image)
        hysteresis_threshold_mask = imf.HysteresisThresholdMask(30, 170)

        hysteresis_threshold_mask(image)

        F_hysteresis_threshold_mask.assert_called_once_with(image, 30, 170)
        assert type(hysteresis_threshold_mask(image)) == np.ndarray

    def it_calls_otsu_threshold_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_otsu_threshold = function_mock(
            request, "histolab.filters.image_filters_functional.otsu_threshold"
        )
        F_otsu_threshold.return_value = np.array(image)
        otsu_threshold = imf.OtsuThreshold()

        otsu_threshold(image)

        F_otsu_threshold.assert_called_once_with(image)
        assert type(otsu_threshold(image)) == np.ndarray

    def it_calls_local_otsu_threshold_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_local_otsu_threshold = function_mock(
            request, "histolab.filters.image_filters_functional.local_otsu_threshold"
        )
        F_local_otsu_threshold.return_value = np.array(image)
        local_otsu_threshold = imf.LocalOtsuThreshold(5)

        local_otsu_threshold(image)

        F_local_otsu_threshold.assert_called_once_with(image, 5)
        assert type(local_otsu_threshold(image)) == np.ndarray

    def it_calls_filter_entropy_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_filter_entropy = function_mock(
            request, "histolab.filters.image_filters_functional.filter_entropy"
        )
        F_filter_entropy.return_value = np.array(image)
        filter_entropy = imf.FilterEntropy(3, 6)

        filter_entropy(image)

        F_filter_entropy.assert_called_once_with(image, 3, 6)
        assert type(filter_entropy(image)) == np.ndarray

    def it_calls_canny_edges_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_canny_edges = function_mock(
            request, "histolab.filters.image_filters_functional.canny_edges"
        )
        F_canny_edges.return_value = np.array(image)
        canny_edges = imf.CannyEdges(0.8, 0.3, 13)

        canny_edges(image)

        F_canny_edges.assert_called_once_with(image, 0.8, 0.3, 13)
        assert type(canny_edges(image)) == np.ndarray

    def it_calls_grays_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_grays = function_mock(
            request, "histolab.filters.image_filters_functional.grays"
        )
        F_grays.return_value = np.array(image)
        grays = imf.Grays(20)

        grays(image)

        F_grays.assert_called_once_with(image, 20)
        assert type(grays(image)) == np.ndarray

    def it_calls_green_channel_filter_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_green_channel_filter = function_mock(
            request, "histolab.filters.image_filters_functional.green_channel_filter"
        )
        F_green_channel_filter.return_value = np.array(image)
        green_channel_filter = imf.GreenChannelFilter(250, False, 85.0)

        green_channel_filter(image)

        F_green_channel_filter.assert_called_once_with(image, 250, False, 85.0)
        assert type(green_channel_filter(image)) == np.ndarray

    def it_calls_red_filter_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_red_filter = function_mock(
            request, "histolab.filters.image_filters_functional.red_filter"
        )
        F_red_filter.return_value = np.array(image)
        red_filter = imf.RedFilter(180, 100, 85)

        red_filter(image)

        F_red_filter.assert_called_once_with(image, 180, 100, 85)
        assert type(red_filter(image)) == np.ndarray

    def it_calls_red_pen_filter_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_red_pen_filter = function_mock(
            request, "histolab.filters.image_filters_functional.red_pen_filter"
        )
        F_red_pen_filter.return_value = np.array(image)
        red_pen_filter = imf.RedPenFilter()

        red_pen_filter(image)

        F_red_pen_filter.assert_called_once_with(image)
        assert type(red_pen_filter(image)) == np.ndarray

    def it_calls_green_filter_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_green_filter = function_mock(
            request, "histolab.filters.image_filters_functional.green_filter"
        )
        F_green_filter.return_value = np.array(image)
        green_filter = imf.GreenFilter(150, 160, 140)

        green_filter(image)

        F_green_filter.assert_called_once_with(image, 150, 160, 140)
        assert type(green_filter(image)) == np.ndarray

    def it_calls_green_pen_filter_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_green_pen_filter = function_mock(
            request, "histolab.filters.image_filters_functional.green_pen_filter"
        )
        F_green_pen_filter.return_value = np.array(image)
        green_pen_filter = imf.GreenPenFilter()

        green_pen_filter(image)

        F_green_pen_filter.assert_called_once_with(image)
        assert type(green_pen_filter(image)) == np.ndarray

    def it_calls_blue_filter_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_blue_filter = function_mock(
            request, "histolab.filters.image_filters_functional.blue_filter"
        )
        F_blue_filter.return_value = np.array(image)
        blue_filter = imf.BlueFilter(60, 120, 190)

        blue_filter(image)

        F_blue_filter.assert_called_once_with(image, 60, 120, 190)
        assert type(blue_filter(image)) == np.ndarray

    def it_calls_blue_pen_filter_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_blue_pen_filter = function_mock(
            request, "histolab.filters.image_filters_functional.blue_pen_filter"
        )
        F_blue_pen_filter.return_value = np.array(image)
        blue_pen_filter = imf.BluePenFilter()

        blue_pen_filter(image)

        F_blue_pen_filter.assert_called_once_with(image)
        assert type(blue_pen_filter(image)) == np.ndarray

    def it_calls_pen_marks_filter_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_pen_marks = function_mock(
            request, "histolab.filters.image_filters_functional.pen_marks"
        )
        F_pen_marks.return_value = np.array(image)
        pen_marks = imf.PenMarks()

        pen_marks(image)

        F_pen_marks.assert_called_once_with(image)
        assert type(pen_marks(image)) == np.ndarray

    def it_calls_np_to_pil(self, request):
        array = NpArrayMock.ONES_30X30_UINT8
        util_np_to_pil = function_mock(request, "histolab.util.np_to_pil")
        util_np_to_pil.return_value = PIL.Image.fromarray(array)
        to_pil_image = imf.ToPILImage()

        to_pil_image(array)

        util_np_to_pil.assert_called_once_with(array)
        assert type(to_pil_image(array)) == PIL.Image.Image

    def it_calls_apply_mask_image(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        mask = NpArrayMock.ONES_500X500X4_BOOL
        util_apply_mask_image = function_mock(request, "histolab.util.apply_mask_image")
        util_apply_mask_image.return_value = PIL.Image.fromarray(np.array(image) * mask)
        class_apply_mask_image = imf.ApplyMaskImage(image)

        class_apply_mask_image(mask)

        util_apply_mask_image.assert_called_once_with(image, mask)
        assert type(util_apply_mask_image(image, mask)) == PIL.Image.Image

    def it_calls_lambda_filter(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image_np = np.array(image)
        fun_ = function_mock(request, "numpy.array")
        fun_.return_value = image_np
        lambda_filter = imf.Lambda(fun_)

        lambda_filter(image)

        fun_.assert_called_once_with(image)
        assert type(lambda_filter(image)) == np.ndarray

    def it_calls_yen_threshold(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_yen_threshold = function_mock(
            request, "histolab.filters.image_filters_functional.yen_threshold"
        )
        F_yen_threshold.return_value = np.array(image)
        yen_threshold = imf.YenThreshold()

        yen_threshold(image)

        F_yen_threshold.assert_called_once_with(image, operator.lt)
        assert type(yen_threshold(image)) == np.ndarray

    def it_calls_rgb_to_lab_functional(self, request):
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        F_rgb_to_lab = function_mock(
            request, "histolab.filters.image_filters_functional.rgb_to_lab"
        )
        F_rgb_to_lab.return_value = image
        rgb_to_lab = imf.RgbToLab()

        rgb_to_lab(image)

        F_rgb_to_lab.assert_called_once_with(image)
        assert type(rgb_to_lab(image)) == PIL.Image.Image
