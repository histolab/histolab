import os

import pytest

import numpy as np
from histolab.filters.compositions import _TileFiltersComposition
from histolab.filters.image_filters import Compose
from histolab.tile import Tile
from histolab.types import CoordinatePair

from ..base import COMPLEX_MASK
from ..unitutil import (
    ANY,
    PILIMG,
    class_mock,
    initializer_mock,
    method_mock,
    property_mock,
)


class Describe_Tile:
    def it_constructs_from_args(self, request):
        _init = initializer_mock(request, Tile)
        _image = PILIMG.RGBA_COLOR_50X50_155_0_0
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
        _image = PILIMG.RGBA_COLOR_50X50_155_0_0
        tile = Tile(_image, None, 0)

        image = tile.image

        assert image == _image

    def it_knows_its_level(self):
        tile = Tile(None, None, 0)

        level = tile.level

        assert level == 0

    def it_can_save_the_tile_image(self, tmpdir):
        tmp_path_ = os.path.join(tmpdir.mkdir("mydir"), "mytile.png")
        _image = PILIMG.RGBA_COLOR_50X50_155_0_0
        tile = Tile(_image, None, 0)

        tile.save(tmp_path_)

        assert os.path.exists(tmp_path_)

    def and_it_can_save_the_tile_image_also_without_ext(self, tmpdir):
        tmp_path_ = os.path.join(tmpdir.mkdir("mydir"), "mytile")
        _image = PILIMG.RGBA_COLOR_50X50_155_0_0
        tile = Tile(_image, None, 0)

        tile.save(tmp_path_)

        assert os.path.exists(tmp_path_ + ".png")

    @pytest.mark.parametrize(
        "almost_white, only_some_tissue, tissue_more_than_percent, expected_value",
        (
            (False, True, True, True),
            (False, False, False, False),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (True, True, True, False),
            (True, False, True, False),
            (True, True, False, False),
        ),
    )
    def it_knows_if_it_has_enough_tissue(
        self,
        _is_almost_white,
        _has_only_some_tissue,
        _has_tissue_more_than_percent,
        almost_white,
        only_some_tissue,
        tissue_more_than_percent,
        expected_value,
    ):
        _is_almost_white.return_value = almost_white
        _has_only_some_tissue.return_value = only_some_tissue
        _has_tissue_more_than_percent.return_value = tissue_more_than_percent
        tile = Tile(None, None, 0)

        has_enough_tissue = tile.has_enough_tissue()

        assert has_enough_tissue == expected_value

    @pytest.mark.parametrize(
        "tissue_mask, percent, expected_value",
        (
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
        ),
    )
    def it_knows_if_has_tissue_more_than_percent(
        self, request, tissue_mask, percent, expected_value
    ):
        _tissue_mask = method_mock(request, Tile, "_tissue_mask")
        _tissue_mask.return_value = tissue_mask

        tile = Tile(None, None, 0)
        has_tissue_more_than_percent = tile._has_tissue_more_than_percent(percent)

        assert has_tissue_more_than_percent == expected_value

    def it_calls_tile_tissue_mask_filters(
        self,
        request,
        RgbToGrayscale_,
        OtsuThreshold_,
        BinaryDilation_,
        BinaryFillHoles_,
    ):
        _tissue_mask_filters = property_mock(
            request, _TileFiltersComposition, "tissue_mask_filters"
        )
        BinaryFillHoles_.return_value = np.zeros((50, 50))
        _tissue_mask_filters.return_value = Compose(
            [RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_]
        )
        image = PILIMG.RGBA_COLOR_50X50_155_0_0
        tile = Tile(image, None, 0)

        tile._has_only_some_tissue()

        _tissue_mask_filters.assert_called_once()
        assert type(tile._has_only_some_tissue()) == np.bool_

    def it_knows_its_tissue_mask(
        self,
        request,
        RgbToGrayscale_,
        OtsuThreshold_,
        BinaryDilation_,
        BinaryFillHoles_,
    ):
        BinaryFillHoles_.return_value = np.zeros((50, 50))
        filters = Compose(
            [RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_]
        )
        image = PILIMG.RGBA_COLOR_50X50_155_0_0
        tile = Tile(image, None, 0)

        tissue_mask = tile._tissue_mask(filters)

        assert tissue_mask.shape == (50, 50)

    def it_knows_its_tissue_ratio(
        self,
        request,
        RgbToGrayscale_,
        OtsuThreshold_,
        BinaryDilation_,
        BinaryFillHoles_,
    ):
        _tile_tissue_mask_filters = property_mock(
            request, _TileFiltersComposition, "tissue_mask_filters"
        )
        filters = Compose(
            [RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_]
        )
        _tile_tissue_mask_filters.return_value = filters
        _call = method_mock(request, Compose, "__call__")
        _call.return_value = COMPLEX_MASK
        image = PILIMG.RGB_RANDOM_COLOR_10X10
        tile = Tile(image, None, 0)

        tissue_ratio = tile.tissue_ratio

        _tile_tissue_mask_filters.assert_called_once()
        _call.assert_called_once_with(filters, image)
        assert type(tissue_ratio) == float
        assert tissue_ratio == 0.61

    def it_knows_how_to_apply_filters_PIL(self, RgbToGrayscale_):
        image_before = PILIMG.RGB_RANDOM_COLOR_10X10
        image_after = PILIMG.GRAY_RANDOM_10X10
        RgbToGrayscale_.return_value = image_after
        tile = Tile(image_before, None, 0)

        filtered_image = tile.apply_filters(RgbToGrayscale_)

        RgbToGrayscale_.assert_called_once_with(tile.image)
        assert isinstance(filtered_image, Tile)
        assert filtered_image.image == image_after
        assert filtered_image.coords is None
        assert filtered_image.level == 0

    def it_knows_how_to_apply_filters_np(self, OtsuThreshold_):
        image_before = PILIMG.RGB_RANDOM_COLOR_10X10
        image_after = PILIMG.GRAY_RANDOM_10X10
        OtsuThreshold_.return_value = np.array(image_after)
        tile = Tile(image_before, None, 0)

        filtered_image = tile.apply_filters(OtsuThreshold_)

        OtsuThreshold_.assert_called_once_with(tile.image)
        assert isinstance(filtered_image, Tile)
        assert filtered_image.image == image_after
        assert filtered_image.coords is None
        assert filtered_image.level == 0

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
