# encoding: utf-8

import numpy as np
import skimage.morphology

from histolab.filters import morphological_filters as mof

from ...unitutil import NpArrayMock, function_mock
from ...base import IMAGE1_RGB, IMAGE2_RGBA


class DescribeMorphologicalFilters(object):
    def it_calls_remove_small_objects_filter_functional(self, request):
        img_arr = IMAGE1_RGB
        F_remove_small_objects = function_mock(
            request,
            "histolab.filters.morphological_filters_functional.remove_small_objects",
        )
        F_remove_small_objects.return_value = img_arr
        remove_small_objects = mof.RemoveSmallObjects()

        remove_small_objects(img_arr)

        F_remove_small_objects.assert_called_once_with(img_arr, 3000, True, 95)
        assert type(remove_small_objects(img_arr)) == np.ndarray

    def it_calls_remove_small_holes_filter_functional(self, request):
        img_arr = IMAGE2_RGBA
        F_remove_small_holes = function_mock(
            request, "skimage.morphology.remove_small_holes"
        )
        F_remove_small_holes.return_value = img_arr
        remove_small_holes = mof.RemoveSmallHoles()

        remove_small_holes(img_arr)

        F_remove_small_holes.assert_called_once_with(img_arr, 3000)
        assert type(remove_small_holes(img_arr)) == np.ndarray

    def it_calls_binary_erosion_filter_functional(self, request):
        mask_arr = NpArrayMock.ONES_500X500X4_BOOL
        disk = skimage.morphology.disk(5)
        F_binary_erosion = function_mock(
            request, "scipy.ndimage.morphology.binary_erosion"
        )
        F_binary_erosion.return_value = mask_arr

        binary_erosion = mof.BinaryErosion()

        binary_erosion(mask_arr)

        np.testing.assert_array_equal(
            F_binary_erosion.call_args_list[0][0][0], mask_arr
        )
        np.testing.assert_array_equal(F_binary_erosion.call_args_list[0][0][1], disk)
        assert F_binary_erosion.call_args_list[0][0][2] == 1
        assert type(binary_erosion(mask_arr)) == np.ndarray

    def it_calls_binary_dilation_filter_functional(self, request):
        mask_arr = NpArrayMock.ONES_500X500X4_BOOL
        disk = skimage.morphology.disk(5)
        F_binary_dilation = function_mock(
            request, "scipy.ndimage.morphology.binary_dilation"
        )
        F_binary_dilation.return_value = mask_arr

        binary_dilation = mof.BinaryDilation()

        binary_dilation(mask_arr)

        np.testing.assert_array_equal(
            F_binary_dilation.call_args_list[0][0][0], mask_arr
        )
        np.testing.assert_array_equal(F_binary_dilation.call_args_list[0][0][1], disk)
        assert F_binary_dilation.call_args_list[0][0][2] == 1
        assert type(binary_dilation(mask_arr)) == np.ndarray

    def it_calls_binary_opening_filter_functional(self, request):
        mask_arr = NpArrayMock.ONES_500X500X4_BOOL
        disk = skimage.morphology.disk(3)
        F_binary_opening = function_mock(
            request, "scipy.ndimage.morphology.binary_opening"
        )
        F_binary_opening.return_value = mask_arr

        binary_opening = mof.BinaryOpening()

        binary_opening(mask_arr)

        np.testing.assert_array_equal(
            F_binary_opening.call_args_list[0][0][0], mask_arr
        )
        np.testing.assert_array_equal(F_binary_opening.call_args_list[0][0][1], disk)
        assert F_binary_opening.call_args_list[0][0][2] == 1
        assert type(binary_opening(mask_arr)) == np.ndarray

    def it_calls_binary_closing_filter_functional(self, request):
        mask_arr = NpArrayMock.ONES_500X500X4_BOOL
        disk = skimage.morphology.disk(3)
        F_binary_closing = function_mock(
            request, "scipy.ndimage.morphology.binary_closing"
        )
        F_binary_closing.return_value = mask_arr

        binary_closing = mof.BinaryClosing()

        binary_closing(mask_arr)

        np.testing.assert_array_equal(
            F_binary_closing.call_args_list[0][0][0], mask_arr
        )
        np.testing.assert_array_equal(F_binary_closing.call_args_list[0][0][1], disk)
        assert F_binary_closing.call_args_list[0][0][2] == 1
        assert type(binary_closing(mask_arr)) == np.ndarray
