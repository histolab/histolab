# encoding: utf-8

"""Unit test suite for histolab.util module."""

import operator
from collections import namedtuple

import numpy as np
import pytest
from histolab.types import CP, Region
from histolab.util import (
    apply_mask_image,
    lazyproperty,
    method_dispatch,
    np_to_pil,
    random_choice_true_mask2d,
    rectangle_to_mask,
    region_coordinates,
    regions_from_binary_mask,
    regions_to_binary_mask,
    scale_coordinates,
    threshold_to_mask,
)

from ..base import (
    COMPLEX_MASK,
    COMPLEX_MASK4,
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
from ..unitutil import function_mock
from ..util import load_expectation, load_python_expression

RegionProps = namedtuple("RegionProps", ("area", "bbox", "centroid", "coords"))


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
        (
            NPY.NP_TO_PIL_RGBA_FLOAT01,
            "RGBA",
            (4, 4),
            np.uint8,
            "python-expr/np-to-pil-rgba",
        ),
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
        ((5, 5), CP(0, 3, 2, 5), "mask-arrays/polygon-to-mask-array-0325"),  # square
        ((5, 5), CP(2, 1, 4, 3), "mask-arrays/polygon-to-mask-array-2143"),  # square
        (
            (5, 6),
            CP(1, 0, 2, 0),
            "mask-arrays/polygon-to-mask-array-1020",
        ),  # line - all false
    ),
)
def test_util_rectangle_to_mask(dims, vertices, expected_array):
    rectangle_mask = rectangle_to_mask(dims, vertices)

    assert rectangle_mask.shape == dims
    np.testing.assert_array_almost_equal(
        rectangle_mask, load_expectation(expected_array, type_="npy")
    )


def test_region_coordinates():
    region = Region(index=0, area=14, bbox=(0, 1, 1, 2), center=(0.5, 0.5), coords=None)
    region_coords_ = region_coordinates(region)

    assert region_coords_ == CP(x_ul=0, y_ul=1, x_br=1, y_br=2)


@pytest.mark.parametrize("seed", [(i,) for i in range(10)])
def test_random_choice_true_mask2d(seed):
    np.random.seed(seed)

    col, row = random_choice_true_mask2d(COMPLEX_MASK)

    assert COMPLEX_MASK[row, col]


def test_regions_to_binary_mask():
    regions = [
        Region(
            index=None,
            area=None,
            bbox=None,
            center=None,
            coords=np.array([[1, 3], [2, 3], [3, 2], [3, 3]]),
        ),
        Region(
            index=None,
            area=None,
            bbox=None,
            center=None,
            coords=np.array([[3, 7], [4, 7], [4, 8]]),
        ),
    ]

    binary_mask_regions = regions_to_binary_mask(regions, dims=(10, 10))

    assert type(binary_mask_regions) == np.ndarray
    assert binary_mask_regions.dtype == bool
    np.testing.assert_array_almost_equal(
        binary_mask_regions,
        load_expectation("mask-arrays/regions-to-binary-mask", type_="npy"),
    )


@pytest.mark.parametrize(
    "mask, region_props, label_return_value, expected_bbox",
    [
        (
            np.array([[True, False], [True, True]]),
            [
                RegionProps(
                    area=3,
                    bbox=(0, 1, 2, 3),
                    centroid=(0.6666666666666666, 0.3333333333333333),
                    coords=np.array([[0, 0], [1, 0], [1, 1]]),
                )
            ],
            [[1, 0], [1, 1]],
            (1, 0, 3, 2),
        ),
        (
            COMPLEX_MASK4,
            [
                RegionProps(
                    area=20,
                    bbox=(2, 2, 9, 7),
                    centroid=(4.7, 4.45),
                    coords=np.array(
                        [
                            [2, 5],
                            [3, 3],
                            [3, 4],
                            [3, 5],
                            [3, 6],
                            [4, 3],
                            [4, 4],
                            [4, 5],
                            [4, 6],
                            [5, 2],
                            [5, 3],
                            [5, 4],
                            [5, 5],
                            [5, 6],
                            [6, 3],
                            [6, 4],
                            [6, 5],
                            [6, 6],
                            [7, 5],
                            [8, 5],
                        ]
                    ),
                )
            ],
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            (2, 2, 7, 9),
        ),
    ],
)
def test_regions_from_binary_mask(
    request, mask, region_props, label_return_value, expected_bbox
):
    label = function_mock(request, "histolab.util.label")
    regionprops = function_mock(request, "histolab.util.regionprops")
    regionprops.return_value = region_props
    label(mask).return_value = label_return_value

    regions_from_binary_mask_ = regions_from_binary_mask(mask)

    regionprops.assert_called_once_with(label(mask))
    assert type(regions_from_binary_mask_) == list
    assert len(regions_from_binary_mask_) == 1
    assert type(regions_from_binary_mask_[0]) == Region
    assert regions_from_binary_mask_[0].index == 0
    assert regions_from_binary_mask_[0].area == region_props[0].area
    assert regions_from_binary_mask_[0].bbox == expected_bbox
    assert regions_from_binary_mask_[0].center == region_props[0].centroid
    np.testing.assert_array_equal(
        regions_from_binary_mask_[0].coords, region_props[0].coords
    )


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


def test_method_dispatch():
    class Obj(object):
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        @method_dispatch
        def get(self, arg):
            return getattr(self, arg, None)

        @get.register(list)
        def _(self, arg):
            return [self.get(x) for x in arg]

    obj = Obj(a=1, b=2, c=3)
    assert obj.get("b") == 2
    assert obj.get(["a", "c"]) == [1, 3]
