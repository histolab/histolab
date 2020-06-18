import numpy as np
import pytest

from histolab.filters.image_filters import Compose
from histolab.tile import Tile
from histolab.types import CoordinatePair

from ..unitutil import ANY, PILImageMock, class_mock, initializer_mock, property_mock


class Describe_Tile(object):
    def it_constructs_from_args(self, request):
        _init = initializer_mock(request, Tile)
        _image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        _level = 0
        _coords = CoordinatePair(0, 0, 50, 50)

        tile = Tile(_image, _level, _coords)

        _init.assert_called_once_with(ANY, _image, _level, _coords)
        assert isinstance(tile, Tile)

    def but_it_has_wrong_image_type(self):
        """This test simulates a wrong user behaviour, using a None object instead of a
        PIL Image for image param"""
        with pytest.raises(AttributeError) as err:
            tile = Tile(None, CoordinatePair(0, 0, 50, 50), 0)
            tile.has_enough_tissue()

        assert isinstance(err.value, AttributeError)
        assert str(err.value) == "'NoneType' object has no attribute 'convert'"

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

    def it_calls_enough_tissue_mask_filters(
        self,
        request,
        RgbToGrayscale_,
        OtsuThreshold_,
        BinaryDilation_,
        BinaryFillHoles_,
    ):
        _enough_tissue_mask_filters = property_mock(
            request, Tile, "_enough_tissue_mask_filters"
        )
        BinaryFillHoles_.return_value = np.zeros((50, 50))
        _enough_tissue_mask_filters.return_value = Compose(
            [RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_]
        )
        image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        tile = Tile(image, None, 0)

        tile._has_only_some_tissue()

        _enough_tissue_mask_filters.assert_called_once()
        assert type(tile._has_only_some_tissue()) == np.bool_

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
