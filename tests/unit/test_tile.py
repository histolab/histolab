import os

import numpy as np
import PIL
import pytest

from histolab.filters.image_filters import Compose
from histolab.tile import Tile
from histolab.types import CoordinatePair

from ..base import COMPLEX_MASK
from ..unitutil import (ANY, PILImageMock, class_mock, function_mock,
                        initializer_mock, method_mock, property_mock)


class Describe_Tile(object):
    def it_constructs_from_args(self, request):
        _init = initializer_mock(request, Tile)
        _image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        _level = 0
        _coords = CoordinatePair(0, 0, 50, 50)

        tile = Tile(_image, _coords, _level)

        _init.assert_called_once_with(ANY, _image, _coords, _level)
        assert isinstance(tile, Tile)

    def but_it_has_wrong_image_type(self):
        """This test simulates a wrong user behaviour, using a None object instead of a
        PIL Image for image param"""
        with pytest.raises(AttributeError) as err:
            tile = Tile(None, CoordinatePair(0, 0, 50, 50), 0)
            tile.has_enough_tissue()

        assert isinstance(err.value, AttributeError)
        assert str(err.value) == "'NoneType' object has no attribute 'convert'"

    def it_knows_its_coords(self):
        _coords = CoordinatePair(0, 0, 50, 50)
        tile = Tile(None, _coords, 0)

        coords = tile.coords

        assert coords == _coords

    def it_knows_its_image(self):
        _image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        tile = Tile(_image, None, 0)

        image = tile.image

        assert image == _image

    def it_knows_its_level(self):
        tile = Tile(None, None, 0)

        level = tile.level

        assert level == 0

    def it_can_save_the_tile_image(self, tmpdir):
        tmp_path_ = os.path.join(tmpdir.mkdir("mydir"), "mytile.png")
        _image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        tile = Tile(_image, None, 0)

        tile.save(tmp_path_)

        assert os.path.exists(tmp_path_)

    def and_it_can_save_the_tile_image_also_without_ext(self, tmpdir):
        tmp_path_ = os.path.join(tmpdir.mkdir("mydir"), "mytile")
        _image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        tile = Tile(_image, None, 0)

        tile.save(tmp_path_)

        assert os.path.exists(tmp_path_ + ".png")

    def it_knows_if_it_has_enough_tissue(
        self,
        has_enough_tissue_fixture,
        _is_almost_white,
        _has_only_some_tissue,
        _has_tissue_more_than_percent,
    ):
        (
            almost_white,
            only_some_tissue,
            tissue_more_than_percent,
            expected_value,
        ) = has_enough_tissue_fixture
        _is_almost_white.return_value = almost_white
        _has_only_some_tissue.return_value = only_some_tissue
        _has_tissue_more_than_percent.return_value = tissue_more_than_percent
        tile = Tile(None, None, 0)

        has_enough_tissue = tile.has_enough_tissue()

        assert has_enough_tissue == expected_value

    def it_knows_if_has_tissue_more_than_percent(
        self, request, has_tissue_more_than_percent_fixture
    ):
        tissue_mask, percent, expected_value = has_tissue_more_than_percent_fixture
        _tissue_mask = method_mock(request, Tile, "_tissue_mask")
        _tissue_mask.return_value = tissue_mask

        tile = Tile(None, None, 0)
        has_tissue_more_than_percent = tile._has_tissue_more_than_percent(percent)

        assert has_tissue_more_than_percent == expected_value

    def it_knows_tissue_areas_mask_filters_composition(
        self, RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_
    ):
        tile = Tile(None, None, 0)

        _enough_tissue_mask_filters_ = tile._enough_tissue_mask_filters

        RgbToGrayscale_.assert_called_once()
        OtsuThreshold_.assert_called_once()
        BinaryDilation_.assert_called_once()

        BinaryFillHoles_.assert_called_once()
        np.testing.assert_almost_equal(
            BinaryFillHoles_.call_args_list[0][1]["structure"], np.ones((5, 5))
        )
        assert _enough_tissue_mask_filters_.filters == [
            RgbToGrayscale_(),
            OtsuThreshold_(),
            BinaryDilation_(),
            BinaryFillHoles_(),
        ]
        assert type(_enough_tissue_mask_filters_) == Compose

    def it_calls_tile_tissue_mask_filters(
        self,
        request,
        RgbToGrayscale_,
        OtsuThreshold_,
        BinaryDilation_,
        BinaryFillHoles_,
    ):
        _tile_tissue_mask_filters = function_mock(
            request, "histolab.filters.compositions.tile_tissue_mask_filters"
        )
        BinaryFillHoles_.return_value = np.zeros((50, 50))
        _tile_tissue_mask_filters.return_value = Compose(
            [RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_]
        )
        image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        tile = Tile(image, None, 0)

        tile._has_only_some_tissue()

        _tile_tissue_mask_filters.assert_called_once()
        assert type(tile._has_only_some_tissue()) == np.bool_

<<<<<<< 3df9238d4cbb245c907f51adf4bfc878eeb7bcbc
    def it_knows_its_tissue_mask(
=======
    def it_knows_its_tissue_ratio(
>>>>>>> Add method for applying filters on a tile + add method to calculate the tissue ratio of a tile
        self,
        request,
        RgbToGrayscale_,
        OtsuThreshold_,
        BinaryDilation_,
        BinaryFillHoles_,
    ):
<<<<<<< 3df9238d4cbb245c907f51adf4bfc878eeb7bcbc
        BinaryFillHoles_.return_value = np.zeros((50, 50))
        filters = Compose(
            [RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_]
        )
        image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        tile = Tile(image, None, 0)

        tissue_mask = tile._tissue_mask(filters)

        assert tissue_mask.shape == (50, 50)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ([[0, 1, 1, 0, 1], [0, 1, 1, 1, 1]], 80, False),
            ([[0, 1, 1, 0, 1], [0, 1, 1, 1, 1]], 10, True),
            ([[0, 1, 1, 0, 1], [0, 1, 1, 1, 1]], 100, False),
            ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], 100, False),
            ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], 10, True),
            ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], 80, True),
            ([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], 60, True),
            ([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 60, False),
            ([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 10, False),
            ([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 3, False),
            ([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 80, False),
            ([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 100, False),
        ]
    )
    def has_tissue_more_than_percent_fixture(self, request):
        tissue_mask, percent, expected_value = request.param
        return tissue_mask, percent, expected_value

    @pytest.fixture(
        params=[
            (False, True, True, True),
            (False, False, False, False),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (True, True, True, False),
            (True, False, True, False),
            (True, True, False, False),
        ]
    )
    def has_enough_tissue_fixture(self, request):
        (
            almost_white,
            only_some_tissue,
            tissue_more_than_percent,
            expected_value,
        ) = request.param
        return (
            almost_white,
            only_some_tissue,
            tissue_more_than_percent,
            expected_value,
        )

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _is_almost_white(self, request):
        return property_mock(request, Tile, "_is_almost_white")

    @pytest.fixture
    def _has_only_some_tissue(self, request):
        return method_mock(request, Tile, "_has_only_some_tissue")

    @pytest.fixture
    def _has_tissue_more_than_percent(self, request):
        return method_mock(request, Tile, "_has_tissue_more_than_percent")
=======
        _tile_tissue_mask_filters = function_mock(
            request, "histolab.tile.compositions.tile_tissue_mask_filters"
        )
        filters = Compose(
            [RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_]
        )
        _tile_tissue_mask_filters.return_value = filters
        _apply_filters = method_mock(request, Tile, "apply_filters")
        _apply_filters.return_value = Tile(PIL.Image.fromarray(COMPLEX_MASK), None, 0)
        image = PILImageMock.DIMS_10X10_RGB_RANDOM_COLOR
        tile = Tile(image, None, 0)

        tissue_ratio = tile.tissue_ratio

        _tile_tissue_mask_filters.assert_called_once()
        _apply_filters.assert_called_once_with(tile, filters)
        assert type(tissue_ratio) == float
        assert tissue_ratio == 0.61

    def it_knows_how_to_apply_filters_PIL(self, RgbToGrayscale_):
        image_before = PILImageMock.DIMS_10X10_RGB_RANDOM_COLOR
        image_after = PILImageMock.DIMS_10X10_GRAY_RANDOM
        RgbToGrayscale_.return_value = image_after
        tile = Tile(image_before, None, 0)

        filtered_image = tile.apply_filters(RgbToGrayscale_)

        RgbToGrayscale_.assert_called_once_with(tile.image)
        assert isinstance(filtered_image, Tile)
        assert filtered_image.image == image_after
        assert filtered_image.coords is None
        assert filtered_image.level == 0
>>>>>>> Add method for applying filters on a tile + add method to calculate the tissue ratio of a tile

    def it_knows_how_to_apply_filters_np(self, OtsuThreshold_):
        image_before = PILImageMock.DIMS_10X10_RGB_RANDOM_COLOR
        image_after = PILImageMock.DIMS_10X10_GRAY_RANDOM
        OtsuThreshold_.return_value = np.array(image_after)
        tile = Tile(image_before, None, 0)

        filtered_image = tile.apply_filters(OtsuThreshold_)

        OtsuThreshold_.assert_called_once_with(tile.image)
        assert isinstance(filtered_image, Tile)
        assert filtered_image.image == image_after
        assert filtered_image.coords is None
        assert filtered_image.level == 0

    @pytest.fixture
    def RgbToGrayscale_(self, request):
        return class_mock(request, "histolab.filters.image_filters.RgbToGrayscale")

    @pytest.fixture
    def OtsuThreshold_(self, request):
        return class_mock(request, "histolab.filters.image_filters.OtsuThreshold")

    @pytest.fixture
    def BinaryDilation_(self, request):
        return class_mock(
            request, "histolab.filters.morphological_filters.BinaryDilation"
        )

    @pytest.fixture
    def BinaryFillHoles_(self, request):
        return class_mock(
            request, "histolab.filters.morphological_filters.BinaryFillHoles"
        )
