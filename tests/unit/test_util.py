# encoding: utf-8

"""Unit test suite for src.histolab.util module."""

import pytest
import numpy as np
import operator

from src.histolab.util import (
    lazyproperty,
    np_to_pil,
    threshold_to_mask,
    polygon_to_mask_array,
)
from tests.base import (
    IMAGE1_GREY,
    IMAGE2_GREY,
    IMAGE3_GREY_BLACK,
    IMAGE4_GREY_WHITE,
    IMAGE1_RGB,
    IMAGE2_RGB,
    IMAGE3_RGB_BLACK,
    IMAGE4_RGB_WHITE,
    IMAGE1_RGBA,
    IMAGE2_RGBA,
    IMAGE3_RGBA_BLACK,
    IMAGE4_RGBA_WHITE,
)


@pytest.mark.parametrize(
    "img_array, expected_mode, expected_size, expected_type, expected_array",
    (
        (
            np.array([[True, False], [True, True]]),
            "L",
            (2, 2),
            np.uint8,
            [[255, 0], [255, 255]],
        ),
        (
            np.array([[[27.0, 7.0], [66.0, 84.0]], [[77.0, 63.0], [98.0, 77.0]]]),
            "LA",
            (2, 2),
            np.uint8,
            [[[229, 249], [190, 172]], [[179, 193], [158, 179]]],
        ),
        (
            np.array(
                [
                    [
                        [87, 130, 227, 200],
                        [67, 234, 45, 68],
                        [244, 207, 19, 21],
                        [216, 213, 220, 16],
                    ],
                    [
                        [16, 245, 236, 78],
                        [240, 202, 186, 24],
                        [201, 78, 14, 243],
                        [43, 78, 74, 83],
                    ],
                    [
                        [16, 35, 219, 6],
                        [97, 130, 251, 151],
                        [184, 20, 109, 216],
                        [200, 5, 179, 155],
                    ],
                    [
                        [1, 39, 195, 204],
                        [90, 179, 247, 35],
                        [129, 95, 194, 116],
                        [228, 120, 156, 170],
                    ],
                ]
            ),
            "RGBA",
            (4, 4),
            np.uint8,
            [
                [
                    [87, 130, 227, 200],
                    [67, 234, 45, 68],
                    [244, 207, 19, 21],
                    [216, 213, 220, 16],
                ],
                [
                    [16, 245, 236, 78],
                    [240, 202, 186, 24],
                    [201, 78, 14, 243],
                    [43, 78, 74, 83],
                ],
                [
                    [16, 35, 219, 6],
                    [97, 130, 251, 151],
                    [184, 20, 109, 216],
                    [200, 5, 179, 155],
                ],
                [
                    [1, 39, 195, 204],
                    [90, 179, 247, 35],
                    [129, 95, 194, 116],
                    [228, 120, 156, 170],
                ],
            ],
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
    np.testing.assert_array_almost_equal(np.array(pil_img), expected_array)


def test_util_threshold_to_mask(threshold_to_mask_fixture):
    (img, threshold, relate, expected_array) = threshold_to_mask_fixture
    mask = threshold_to_mask(np_to_pil(img), threshold, relate)

    np.testing.assert_array_equal(mask, expected_array)


@pytest.mark.parametrize(
    "dims, vertices, expected_array",
    (
        (
            (5, 5),
            [(0, 3), (2, 3), (2, 5), (0, 5)],
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                ]
            ),
        ),
        (
            (5, 6),
            [(1, 0), (1, 1), (2, 0), (2, 1)],
            np.array(
                [
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            (5, 5),
            [(2, 1), (4, 1), (4, 3), (2, 3)],
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
    ),
)
def test_util_polygon_to_mask_array(dims, vertices, expected_array):
    polygon_mask = polygon_to_mask_array(dims, vertices)

    np.testing.assert_array_almost_equal(polygon_mask, expected_array)


class DescribeLazyPropertyDecorator(object):
    """Tests @lazyproperty decorator class."""

    def it_is_a_lazyproperty_object_on_class_access(self, Obj):
        assert isinstance(Obj.fget, lazyproperty)

    def but_it_adopts_the_name_of_the_decorated_method(self, Obj):
        assert Obj.fget.__name__ == "fget"

    def and_it_adopts_the_module_of_the_decorated_method(self, Obj):
        # ---the module name actually, not a module object
        assert Obj.fget.__module__ == __name__

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
        class Obj(object):
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


@pytest.fixture(
    params=[
        (
            IMAGE1_GREY,
            160,
            operator.gt,
            np.array(
                [
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                    [True, True, False, False, False],
                    [False, False, False, False, True],
                    [False, True, False, True, False],
                ]
            ),
        ),
        (
            IMAGE2_GREY,
            140,
            operator.lt,
            np.array(
                [
                    [False, False, False, True, True],
                    [True, True, True, True, True],
                    [False, False, False, False, False],
                    [True, False, False, True, True],
                    [False, True, False, True, False],
                ]
            ),
        ),
        (IMAGE3_GREY_BLACK, 0, operator.gt, np.zeros((5, 5), dtype=bool),),
        (IMAGE3_GREY_BLACK, 0, operator.lt, np.zeros((5, 5), dtype=bool),),
        (IMAGE3_GREY_BLACK, 1, operator.lt, np.ones((5, 5), dtype=bool),),
        (IMAGE4_GREY_WHITE, 0, operator.gt, np.ones((5, 5), dtype=bool),),
        (IMAGE4_GREY_WHITE, 1, operator.lt, np.zeros((5, 5), dtype=bool),),
        (
            IMAGE1_RGB,
            200,
            operator.lt,
            np.array(
                [
                    [
                        [False, False, False],
                        [False, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                    ],
                    [
                        [True, False, False],
                        [True, False, True],
                        [True, True, True],
                        [True, False, False],
                        [True, True, False],
                    ],
                    [
                        [True, True, True],
                        [True, False, True],
                        [True, True, False],
                        [False, True, True],
                        [True, True, True],
                    ],
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, False, False],
                        [True, True, True],
                        [True, True, True],
                    ],
                    [
                        [False, True, True],
                        [True, False, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                    ],
                ]
            ),
        ),
        (
            IMAGE2_RGB,
            39,
            operator.gt,
            np.array(
                [
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [False, True, False],
                    ],
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, True, False],
                        [True, True, False],
                        [True, True, True],
                    ],
                    [
                        [True, True, True],
                        [False, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, False],
                    ],
                    [
                        [True, True, False],
                        [True, True, True],
                        [True, True, False],
                        [False, True, True],
                        [False, True, True],
                    ],
                    [
                        [True, True, True],
                        [False, False, True],
                        [True, False, False],
                        [True, True, True],
                        [True, True, True],
                    ],
                ]
            ),
        ),
        (IMAGE3_RGB_BLACK, 0, operator.gt, np.zeros((5, 5, 3), dtype=bool),),
        (IMAGE3_RGB_BLACK, 0, operator.lt, np.zeros((5, 5, 3), dtype=bool),),
        (IMAGE3_RGB_BLACK, 1, operator.lt, np.ones((5, 5, 3), dtype=bool),),
        (IMAGE4_RGB_WHITE, 0, operator.gt, np.ones((5, 5, 3), dtype=bool),),
        (IMAGE4_RGB_WHITE, 1, operator.lt, np.zeros((5, 5, 3), dtype=bool),),
        (
            IMAGE1_RGBA,
            178,
            operator.gt,
            np.array(
                [
                    [
                        [False, False, True, True],
                        [False, True, False, False],
                        [False, True, True, True],
                        [False, False, True, False],
                        [True, False, True, False],
                    ],
                    [
                        [True, False, False, False],
                        [False, False, True, False],
                        [True, False, False, False],
                        [False, False, True, False],
                        [False, True, False, False],
                    ],
                    [
                        [False, False, True, True],
                        [False, False, False, False],
                        [False, True, False, False],
                        [False, False, True, True],
                        [True, False, True, False],
                    ],
                    [
                        [True, False, False, True],
                        [True, True, True, False],
                        [False, True, True, True],
                        [True, False, True, True],
                        [False, False, False, False],
                    ],
                    [
                        [False, False, True, True],
                        [False, True, False, False],
                        [True, False, False, True],
                        [True, False, False, False],
                        [False, False, False, True],
                    ],
                ]
            ),
        ),
        (
            IMAGE2_RGBA,
            37,
            operator.lt,
            np.array(
                [
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, True, False],
                        [False, False, False, False],
                    ],
                    [
                        [True, False, True, False],
                        [True, False, False, False],
                        [False, False, False, True],
                        [False, False, False, False],
                        [False, False, True, False],
                    ],
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [True, False, False, False],
                        [False, False, False, False],
                        [False, False, True, True],
                    ],
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [True, True, True, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ],
                    [
                        [True, False, False, False],
                        [True, False, False, False],
                        [False, False, False, False],
                        [True, False, False, False],
                        [False, False, False, False],
                    ],
                ]
            ),
        ),
        (IMAGE3_RGBA_BLACK, 2, operator.gt, np.zeros((5, 5, 4), dtype=bool),),
        (IMAGE3_RGBA_BLACK, 0, operator.lt, np.zeros((5, 5, 4), dtype=bool),),
        (IMAGE3_RGBA_BLACK, 1, operator.lt, np.ones((5, 5, 4), dtype=bool),),
        (IMAGE4_RGBA_WHITE, 0, operator.gt, np.ones((5, 5, 4), dtype=bool),),
        (IMAGE4_RGBA_WHITE, 1, operator.lt, np.zeros((5, 5, 4), dtype=bool),),
    ]
)
def threshold_to_mask_fixture(request):
    (img, threshold, relate, expected_array) = request.param
    return img, threshold, relate, expected_array
