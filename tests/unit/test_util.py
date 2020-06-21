# encoding: utf-8

"""Unit test suite for histolab.util module."""

import operator

import numpy as np
import pytest
from tests.base import (
    IMAGE1_GREY,
    IMAGE1_RGB,
    IMAGE1_RGBA,
    IMAGE2_GREY,
    IMAGE2_RGB,
    IMAGE2_RGBA,
    IMAGE3_GREY_BLACK,
    IMAGE3_RGB_BLACK,
    IMAGE3_RGBA_BLACK,
    IMAGE4_GREY_WHITE,
    IMAGE4_RGB_WHITE,
    IMAGE4_RGBA_WHITE,
    SPARSE_BASE_MASK,
    SPARSE_COMPLEX_MASK,
)

from histolab.types import CoordinatePair, Region
from histolab.util import (
    apply_mask_image,
    lazyproperty,
    np_to_pil,
    polygon_to_mask_array,
    region_coordinates,
    resize_mask,
    scale_coordinates,
    threshold_to_mask,
)

from ..util import load_expectation


@pytest.mark.parametrize(
    "ref_coords, ref_size, target_size, expected_value",
    (
        (CoordinatePair(0, 2, 4, 5), (10, 10), (5, 5), (0, 1, 2, 2)),
        (CoordinatePair(90, 112, 124, 125), (100, 100), (95, 95), (85, 106, 117, 118)),
    ),
)
def test_scale_coordinates(ref_coords, ref_size, target_size, expected_value):
    x_ul, y_ul, x_br, y_br = expected_value

    scaled_coords = scale_coordinates(ref_coords, ref_size, target_size)

    assert scaled_coords == CoordinatePair(x_ul, y_ul, x_br, y_br)


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
    img, threshold, relate, expected_array = threshold_to_mask_fixture

    mask = threshold_to_mask(np_to_pil(img), threshold, relate)

    np.testing.assert_array_equal(mask, expected_array)


def test_apply_mask_image(apply_mask_image_fixture):
    img, mask, expected_array = apply_mask_image_fixture

    masked_image = apply_mask_image(img, mask)

    np.testing.assert_array_almost_equal(np.array(masked_image), expected_array)


@pytest.mark.parametrize(
    "dims, vertices, expected_array",
    (
        (
            (5, 5),
            CoordinatePair(0, 3, 2, 5),
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
            CoordinatePair(1, 0, 2, 0),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            (5, 5),
            CoordinatePair(2, 1, 4, 3),
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


@pytest.mark.parametrize(
    "mask_array, target_dims, expected_array",
    (
        (SPARSE_BASE_MASK, (2, 2), "sparse-mask-arrays/resized-base-mask"),
        (SPARSE_COMPLEX_MASK, (5, 5), "sparse-mask-arrays/resized-complex-mask"),
    ),
)
def test_util_resize_mask(mask_array, target_dims, expected_array):
    resized_mask = resize_mask(mask_array, target_dims)

    np.testing.assert_array_almost_equal(
        resized_mask.todense(), load_expectation(expected_array, "npz").todense()
    )


def test_region_coordinates():
    region = Region(index=0, area=14, bbox=(0, 1, 1, 2), center=(0.5, 0.5))
    region_coords_ = region_coordinates(region)

    assert region_coords_ == CoordinatePair(x_ul=1, y_ul=0, x_br=2, y_br=1)


# fixtures ---------------------------------------------


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
        (IMAGE3_GREY_BLACK, 0, operator.gt, np.zeros((5, 5), dtype=bool)),
        (IMAGE3_GREY_BLACK, 0, operator.lt, np.zeros((5, 5), dtype=bool)),
        (IMAGE3_GREY_BLACK, 1, operator.lt, np.ones((5, 5), dtype=bool)),
        (IMAGE4_GREY_WHITE, 0, operator.gt, np.ones((5, 5), dtype=bool)),
        (IMAGE4_GREY_WHITE, 1, operator.lt, np.zeros((5, 5), dtype=bool)),
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
        (IMAGE3_RGB_BLACK, 0, operator.gt, np.zeros((5, 5, 3), dtype=bool)),
        (IMAGE3_RGB_BLACK, 0, operator.lt, np.zeros((5, 5, 3), dtype=bool)),
        (IMAGE3_RGB_BLACK, 1, operator.lt, np.ones((5, 5, 3), dtype=bool)),
        (IMAGE4_RGB_WHITE, 0, operator.gt, np.ones((5, 5, 3), dtype=bool)),
        (IMAGE4_RGB_WHITE, 1, operator.lt, np.zeros((5, 5, 3), dtype=bool)),
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
        (IMAGE3_RGBA_BLACK, 2, operator.gt, np.zeros((5, 5, 4), dtype=bool)),
        (IMAGE3_RGBA_BLACK, 0, operator.lt, np.zeros((5, 5, 4), dtype=bool)),
        (IMAGE3_RGBA_BLACK, 1, operator.lt, np.ones((5, 5, 4), dtype=bool)),
        (IMAGE4_RGBA_WHITE, 0, operator.gt, np.ones((5, 5, 4), dtype=bool)),
        (IMAGE4_RGBA_WHITE, 1, operator.lt, np.zeros((5, 5, 4), dtype=bool)),
    ]
)
def threshold_to_mask_fixture(request):
    (img, threshold, relate, expected_array) = request.param
    return img, threshold, relate, expected_array


@pytest.fixture(
    params=[
        (
            np.array(
                [
                    [
                        [9.53975711, 45.98541936, 156.28007954],
                        [6.73064837, 176.78741919, 174.95470397],
                        [157.17617866, 71.21380744, 219.48186052],
                        [249.72378036, 226.46022371, 119.54589884],
                        [17.29386951, 199.52296683, 43.44804103],
                    ],
                    [
                        [73.14354246, 64.67302707, 74.04000342],
                        [36.28444347, 44.30212391, 161.701794],
                        [135.21200604, 219.2487995, 69.34124309],
                        [79.08871113, 139.76394382, 75.24903853],
                        [148.42727139, 40.26703283, 155.14522723],
                    ],
                    [
                        [207.72097941, 94.85677006, 136.95335035],
                        [35.87960978, 83.0704467, 152.93280054],
                        [138.24239063, 173.45527492, 20.54340419],
                        [29.00949635, 17.22138894, 250.35899589],
                        [200.83328237, 115.16546538, 253.85420946],
                    ],
                    [
                        [151.96629846, 95.88575407, 60.66781005],
                        [163.88449905, 0.3666264, 167.66724667],
                        [250.22130117, 80.98605632, 105.51786192],
                        [72.47738539, 57.77933851, 192.59844648],
                        [155.69520169, 221.76117742, 100.33591651],
                    ],
                    [
                        [84.288965, 55.17525206, 130.47955595],
                        [106.7083522, 123.64126622, 248.81003909],
                        [117.11354434, 155.69250301, 214.32878442],
                        [16.42456011, 121.28565776, 122.80049983],
                        [154.15207853, 146.9827038, 26.26988447],
                    ],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                ]
            ),
            np.array(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [
                        [73, 64, 74],
                        [36, 44, 161],
                        [135, 219, 69],
                        [79, 139, 75],
                        [148, 40, 155],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [
                        [84, 55, 130],
                        [106, 123, 248],
                        [117, 155, 214],
                        [16, 121, 122],
                        [154, 146, 26],
                    ],
                ]
            ),
        ),
        (
            np.array(
                [
                    [
                        [94, 96, 86],
                        [239, 168, 4],
                        [60, 37, 62],
                        [221, 219, 175],
                        [10, 248, 11],
                    ],
                    [
                        [82, 206, 188],
                        [17, 90, 219],
                        [2, 17, 134],
                        [62, 28, 4],
                        [209, 124, 224],
                    ],
                    [
                        [104, 89, 239],
                        [140, 137, 95],
                        [60, 6, 105],
                        [199, 99, 19],
                        [97, 92, 234],
                    ],
                    [
                        [164, 160, 13],
                        [4, 75, 214],
                        [221, 106, 122],
                        [216, 46, 242],
                        [239, 77, 137],
                    ],
                    [
                        [227, 55, 112],
                        [156, 26, 142],
                        [175, 138, 46],
                        [95, 104, 90],
                        [190, 146, 200],
                    ],
                ]
            ),
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [
                        [94, 96, 86],
                        [239, 168, 4],
                        [60, 37, 62],
                        [221, 219, 175],
                        [10, 248, 11],
                    ],
                    [
                        [82, 206, 188],
                        [17, 90, 219],
                        [2, 17, 134],
                        [62, 28, 4],
                        [209, 124, 224],
                    ],
                    [
                        [104, 89, 239],
                        [140, 137, 95],
                        [60, 6, 105],
                        [199, 99, 19],
                        [97, 92, 234],
                    ],
                    [
                        [164, 160, 13],
                        [4, 75, 214],
                        [221, 106, 122],
                        [216, 46, 242],
                        [239, 77, 137],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]
            ),
        ),
        (
            np.array(
                [
                    [
                        [23, 153, 214, 226],
                        [82, 172, 200, 90],
                        [179, 178, 109, 250],
                        [178, 23, 132, 171],
                        [159, 104, 98, 179],
                    ],
                    [
                        [167, 246, 250, 180],
                        [93, 157, 27, 234],
                        [62, 104, 193, 14],
                        [82, 41, 14, 29],
                        [104, 89, 181, 183],
                    ],
                    [
                        [172, 206, 47, 28],
                        [185, 136, 220, 130],
                        [183, 196, 147, 210],
                        [156, 142, 209, 193],
                        [135, 0, 153, 39],
                    ],
                    [
                        [45, 157, 13, 187],
                        [98, 234, 128, 112],
                        [11, 19, 126, 139],
                        [215, 185, 148, 125],
                        [66, 240, 220, 101],
                    ],
                    [
                        [117, 196, 13, 6],
                        [126, 189, 51, 220],
                        [145, 226, 5, 147],
                        [220, 94, 237, 45],
                        [72, 193, 231, 140],
                    ],
                ]
            ),
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [
                        [23, 153, 214, 226],
                        [82, 172, 200, 90],
                        [179, 178, 109, 250],
                        [178, 23, 132, 171],
                        [159, 104, 98, 179],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [172, 206, 47, 28],
                        [185, 136, 220, 130],
                        [183, 196, 147, 210],
                        [156, 142, 209, 193],
                        [135, 0, 153, 39],
                    ],
                    [
                        [45, 157, 13, 187],
                        [98, 234, 128, 112],
                        [11, 19, 126, 139],
                        [215, 185, 148, 125],
                        [66, 240, 220, 101],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ]
            ),
        ),
        (
            np.array(
                [
                    [243, 123, 83, 251, 90],
                    [135, 162, 48, 11, 227],
                    [31, 151, 68, 192, 70],
                    [40, 220, 239, 170, 128],
                    [217, 56, 101, 111, 119],
                ]
            ),
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [243, 123, 83, 251, 90],
                    [135, 162, 48, 11, 227],
                    [31, 151, 68, 192, 70],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
    ]
)
def apply_mask_image_fixture(request):
    (img, mask, expected_array) = request.param
    return img, mask, expected_array


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
