# encoding: utf-8

import os

import pytest

from src.histolab.filters.image_filters import *
from PIL import ImageOps

from ..unitutil import (
    dict_list_eq,
    property_mock,
    initializer_mock,
    class_mock,
    instance_mock,
    method_mock,
    function_mock,
    PILImageMock,
    ANY,
)


class DescribeImageFilters(object):
    def it_calls_invert_filter_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_invert = function_mock(
            request, "src.histolab.filters.image_filters_functional.invert"
        )
        F_invert.return_value = image
        invert = Invert()
        invert(image)

        F_invert.assert_called_once_with(image)
        assert type(F_invert.return_value) == type(image)

    def it_calls_pil_greyscale(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        greyscale_filter = function_mock(request, "PIL.ImageOps.grayscale")
        greyscale_filter.return_value = image
        greyscale = RgbToGreyscale()
        greyscale(image)

        greyscale_filter.assert_called_once_with(image)
        assert type(greyscale_filter.return_value) == type(image)

    def it_calls_rgb_to_hed_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_rgb_to_hed = function_mock(
            request, "src.histolab.filters.image_filters_functional.rgb_to_hed"
        )
        F_rgb_to_hed.return_value = image
        rgb_to_hed = RgbToHed()
        rgb_to_hed(image)

        F_rgb_to_hed.assert_called_once_with(image)
        assert type(F_rgb_to_hed.return_value) == type(image)

    def it_calls_rgb_to_hsv_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_rgb_to_hsv = function_mock(
            request, "src.histolab.filters.image_filters_functional.rgb_to_hsv"
        )
        F_rgb_to_hsv.return_value = image
        rgb_to_hsv = RgbToHsv()
        rgb_to_hsv(image)

        F_rgb_to_hsv.assert_called_once_with(image)
        assert type(F_rgb_to_hsv.return_value) == type(image)

    def it_calls_stretch_contrast_functional(self, request):
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        F_stretch_contrast = function_mock(
            request, "src.histolab.filters.image_filters_functional.stretch_contrast"
        )
        F_stretch_contrast.return_value = image
        stretch_contrast = StretchContrast()
        stretch_contrast(image, 0, 100)

        F_stretch_contrast.assert_called_once_with(image, 0, 100)
        assert type(F_stretch_contrast.return_value) == type(image)
