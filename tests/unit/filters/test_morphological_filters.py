# encoding: utf-8

import numpy as np
import pytest
import skimage.morphology

from histolab.filters import morphological_filters as mof

from ...base import IMAGE1_RGB, IMAGE2_RGBA
from ...unitutil import NpArrayMock, function_mock


class DescribeMorphologicalFilters:
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

    def but_it_raises_exception_when_call_binary_erosion_non_mask_array(self, request):
        array = np.array([(1, 2, 3), (4, 5, 6)])
        F_binary_erosion = function_mock(
            request, "scipy.ndimage.morphology.binary_erosion"
        )
        F_binary_erosion.return_value = array

        with pytest.raises(ValueError) as err:
            binary_erosion = mof.BinaryErosion()
            binary_erosion(array)

        assert str(err.value) == "Mask must be binary"

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

    def but_it_raises_exception_when_call_binary_dilation_non_mask_array(self, request):
        array = np.array([(1, 2, 3), (4, 5, 6)])
        F_binary_dilation = function_mock(
            request, "scipy.ndimage.morphology.binary_dilation"
        )
        F_binary_dilation.return_value = array

        with pytest.raises(ValueError) as err:
            binary_dilation = mof.BinaryDilation()
            binary_dilation(array)

        assert str(err.value) == "Mask must be binary"

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

    def but_it_raises_exception_when_call_binary_opening_non_mask_array(self, request):
        array = np.array([(1, 2, 3), (4, 5, 6)])
        F_binary_opening = function_mock(
            request, "scipy.ndimage.morphology.binary_opening"
        )
        F_binary_opening.return_value = array

        with pytest.raises(ValueError) as err:
            binary_opening = mof.BinaryOpening()
            binary_opening(array)

        assert str(err.value) == "Mask must be binary"

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

    def but_it_raises_exception_when_call_binary_closing_non_mask_array(self, request):
        array = np.array([(1, 2, 3), (4, 5, 6)])
        F_binary_closing = function_mock(
            request, "scipy.ndimage.morphology.binary_closing"
        )
        F_binary_closing.return_value = array

        with pytest.raises(ValueError) as err:
            binary_closing = mof.BinaryClosing()
            binary_closing(array)

        assert str(err.value) == "Mask must be binary"

    def it_calls_white_top_hat_filter(self, request):
        mask_arr = NpArrayMock.ONES_500X500X4_BOOL
        disk = skimage.morphology.disk(5)
        _white_top_hat = function_mock(request, "skimage.morphology.white_tophat")
        _white_top_hat.return_value = mask_arr

        white_top_hat = mof.WhiteTopHat(disk)

        white_top_hat(mask_arr)

        np.testing.assert_array_equal(_white_top_hat.call_args_list[0][0][0], mask_arr)
        np.testing.assert_array_equal(_white_top_hat.call_args_list[0][0][1], disk)
        assert type(white_top_hat(mask_arr)) == np.ndarray

    def it_calls_watershed_segmentation_functional(self, request):
        mask_arr = NpArrayMock.ONES_500X500X4_BOOL
        F_watershed_segmentation = function_mock(
            request,
            "histolab.filters.morphological_filters_functional.watershed_segmentation",
        )
        F_watershed_segmentation.return_value = mask_arr
        watershed_segmentation = mof.WatershedSegmentation()

        watershed_segmentation(mask_arr)

        F_watershed_segmentation.assert_called_once_with(mask_arr, 6)
        assert type(watershed_segmentation(mask_arr)) == np.ndarray
