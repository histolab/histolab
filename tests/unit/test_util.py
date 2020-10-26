# encoding: utf-8

"""Unit test suite for histolab.util module."""

import operator

import pytest

import numpy as np
from histolab.types import CP, Region
from histolab.util import (
    apply_mask_image,
    lazyproperty,
    np_to_pil,
    polygon_to_mask_array,
    region_coordinates,
    scale_coordinates,
    threshold_to_mask,
)
from tests.base import (
    IMAGE1_GRAY,
    IMAGE1_RGB,
    IMAGE1_RGBA,
    IMAGE2_GRAY,
    IMAGE2_RGB,
    IMAGE2_RGBA,
    IMAGE3_GRAY_BLACK,
    IMAGE3_RGB_BLACK,
    IMAGE3_RGBA_BLACK,
    IMAGE4_GRAY_WHITE,
    IMAGE4_RGB_WHITE,
    IMAGE4_RGBA_WHITE,
)

from ..fixtures import MASKNPY, NPY
from ..util import load_expectation, load_python_expression


@pytest.mark.parametrize(
    "ref_coords, ref_size, target_size, expected_value",
    (
        (CP(0, 2, 4, 5), (10, 10), (5, 5), (0, 1, 2, 2)),
        (CP(90, 112, 124, 125), (100, 100), (95, 95), (85, 106, 117, 118)),
    ),
)
def test_scale_coordinates(ref_coords, ref_size, target_size, expected_value):
    x_ul, y_ul, x_br, y_br = expected_value

    scaled_coords = scale_coordinates(ref_coords, ref_size, target_size)

    assert scaled_coords == CP(x_ul, y_ul, x_br, y_br)


@pytest.mark.parametrize(
    "img_array, expected_mode, expected_size, expected_type, expected_array",
    (
        (NPY.NP_TO_PIL_L, "L", (2, 2), np.uint8, "python-expr/np-to-pil-l"),
        (NPY.NP_TO_PIL_LA, "LA", (2, 2), np.uint8, "python-expr/np-to-pil-la"),
        (NPY.NP_TO_PIL_RGBA, "RGBA", (4, 4), np.uint8, "python-expr/np-to-pil-rgba"),
    ),
)
def test_util_np_to_pil(
    img_array, expected_mode, expected_size, expected_type, expected_array
):
    pil_img = np_to_pil(img_array)

    assert pil_img.mode == expected_mode
    assert pil_img.size == expected_size
    assert np.array(pil_img).dtype == expected_type
    np.testing.assert_array_almost_equal(
        np.array(pil_img), load_python_expression(expected_array)
    )


@pytest.mark.parametrize(
    "img, threshold, relate, expected_array",
    (
        (IMAGE1_GRAY, 160, operator.gt, "python-expr/threshold-to-mask-160"),
        (IMAGE2_GRAY, 140, operator.lt, "python-expr/threshold-to-mask-140"),
        (IMAGE3_GRAY_BLACK, 0, operator.gt, "python-expr/5-x-5-zeros"),
        (IMAGE3_GRAY_BLACK, 0, operator.lt, "python-expr/5-x-5-zeros"),
        (IMAGE3_GRAY_BLACK, 1, operator.lt, "python-expr/5-x-5-ones"),
        (IMAGE4_GRAY_WHITE, 0, operator.gt, "python-expr/5-x-5-ones"),
        (IMAGE4_GRAY_WHITE, 1, operator.lt, "python-expr/5-x-5-zeros"),
        (IMAGE1_RGB, 200, operator.lt, "python-expr/threshold-to-mask-200"),
        (IMAGE2_RGB, 39, operator.gt, "python-expr/threshold-to-mask-39"),
        (IMAGE3_RGB_BLACK, 0, operator.gt, "python-expr/5-x-5-x-3-zeros"),
        (IMAGE3_RGB_BLACK, 0, operator.lt, "python-expr/5-x-5-x-3-zeros"),
        (IMAGE3_RGB_BLACK, 1, operator.lt, "python-expr/5-x-5-x-3-ones"),
        (IMAGE4_RGB_WHITE, 0, operator.gt, "python-expr/5-x-5-x-3-ones"),
        (IMAGE4_RGB_WHITE, 1, operator.lt, "python-expr/5-x-5-x-3-zeros"),
        (IMAGE1_RGBA, 178, operator.gt, "python-expr/threshold-to-mask-178"),
        (IMAGE2_RGBA, 37, operator.lt, "python-expr/threshold-to-mask-37"),
        (IMAGE3_RGBA_BLACK, 2, operator.gt, "python-expr/5-x-5-x-4-zeros"),
        (IMAGE3_RGBA_BLACK, 0, operator.lt, "python-expr/5-x-5-x-4-zeros"),
        (IMAGE3_RGBA_BLACK, 1, operator.lt, "python-expr/5-x-5-x-4-ones"),
        (IMAGE4_RGBA_WHITE, 0, operator.gt, "python-expr/5-x-5-x-4-ones"),
        (IMAGE4_RGBA_WHITE, 1, operator.lt, "python-expr/5-x-5-x-4-zeros"),
    ),
)
def test_util_threshold_to_mask(img, threshold, relate, expected_array):
    mask = threshold_to_mask(np_to_pil(img), threshold, relate)

    np.testing.assert_array_equal(mask, load_python_expression(expected_array))


@pytest.mark.parametrize(
    "img, mask, expected_array",
    (
        (
            NPY.APPLY_MASK_IMAGE_F1,
            MASKNPY.APPLY_MASK_IMAGE_F1,
            "python-expr/apply-mask-image-exp1",
        ),
        (
            NPY.APPLY_MASK_IMAGE_F2,
            MASKNPY.APPLY_MASK_IMAGE_F2,
            "python-expr/apply-mask-image-exp2",
        ),
        (
            NPY.APPLY_MASK_IMAGE_F3,
            MASKNPY.APPLY_MASK_IMAGE_F3,
            "python-expr/apply-mask-image-exp3",
        ),
        (
            NPY.APPLY_MASK_IMAGE_F4,
            MASKNPY.APPLY_MASK_IMAGE_F4,
            "python-expr/apply-mask-image-exp4",
        ),
    ),
)
def test_apply_mask_image(img, mask, expected_array):
    masked_image = apply_mask_image(img, mask)

    np.testing.assert_array_almost_equal(
        np.array(masked_image), load_python_expression(expected_array)
    )


@pytest.mark.parametrize(
    "dims, vertices, expected_array",
    (
        ((5, 5), CP(0, 3, 2, 5), "mask-arrays/polygon-to-mask-array-0325"),
        ((5, 6), CP(1, 0, 2, 0), "mask-arrays/polygon-to-mask-array-1020"),
        ((5, 5), CP(2, 1, 4, 3), "mask-arrays/polygon-to-mask-array-2143"),
    ),
)
def test_util_polygon_to_mask_array(dims, vertices, expected_array):
    polygon_mask = polygon_to_mask_array(dims, vertices)

    np.testing.assert_array_almost_equal(
        polygon_mask, load_expectation(expected_array, type_="npy")
    )


def test_region_coordinates():
    region = Region(index=0, area=14, bbox=(0, 1, 1, 2), center=(0.5, 0.5))
    region_coords_ = region_coordinates(region)

    assert region_coords_ == CP(x_ul=1, y_ul=0, x_br=2, y_br=1)


class DescribeLazyPropertyDecorator:
    """Tests @lazyproperty decorator class."""

    def it_is_a_property_object_on_class_access(self, Obj):
        assert isinstance(Obj.fget, property)

    def and_it_adopts_the_docstring_of_the_decorated_method(self, Obj):
        assert Obj.fget.__doc__ == "Docstring of Obj.fget method definition."

    def it_only_calculates_value_on_first_call(self, obj):
        assert obj.fget == 1
        assert obj.fget == 1

    def it_raises_on_attempt_to_assign(self, obj):
        assert obj.fget == 1
        with pytest.raises(AttributeError):
            obj.fget = 42
        assert obj.fget == 1
        assert obj.fget == 1

    # fixture components ---------------------------------------------

    @pytest.fixture
    def Obj(self):
        class Obj:
            @lazyproperty
            def fget(self):
                """Docstring of Obj.fget method definition."""
                if not hasattr(self, "_n"):
                    self._n = 0
                self._n += 1
                return self._n

        return Obj

    @pytest.fixture
    def obj(self, Obj):
        return Obj()
