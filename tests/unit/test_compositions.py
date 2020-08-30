import numpy as np
import pytest

from histolab.filters.compositions import FiltersComposition
from histolab.filters.image_filters import Compose
from histolab.tile import Tile

from ..unitutil import class_mock


def it_knows_tissue_areas_mask_filters_composition(
    RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_
):
    _enough_tissue_mask_filters_ = FiltersComposition(Tile).tissue_mask_filters

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


@pytest.fixture
def RgbToGrayscale_(request):
    return class_mock(request, "histolab.filters.image_filters.RgbToGrayscale")


@pytest.fixture
def OtsuThreshold_(request):
    return class_mock(request, "histolab.filters.image_filters.OtsuThreshold")


@pytest.fixture
def BinaryDilation_(request):
    return class_mock(request, "histolab.filters.morphological_filters.BinaryDilation")


@pytest.fixture
def BinaryFillHoles_(request):
    return class_mock(request, "histolab.filters.morphological_filters.BinaryFillHoles")
