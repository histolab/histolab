import numpy as np
import pytest

from histolab.exceptions import FilterCompositionError
from histolab.filters.compositions import (
    FiltersComposition,
    _SlideFiltersComposition,
    _TileFiltersComposition,
)
from histolab.filters.image_filters import Compose
from histolab.slide import Slide
from histolab.tile import Tile

from ..unitutil import class_mock, initializer_mock


class NotAllowedInFiltersComposition:
    """A class NOT allowed to be used with FiltersComposition."""


def it_knows_tissue_areas_mask_tile_filters_composition(
    RgbToGrayscale_, OtsuThreshold_, BinaryDilation_, BinaryFillHoles_
):
    _enough_tissue_mask_filters_ = FiltersComposition(Tile).tissue_mask_filters

    RgbToGrayscale_.assert_called_once()
    OtsuThreshold_.assert_called_once()
    BinaryDilation_.assert_called_once()
    assert BinaryDilation_.call_args_list[0][1]["disk_size"] == 2
    BinaryFillHoles_.assert_called_once()
    np.testing.assert_almost_equal(
        BinaryFillHoles_.call_args_list[0][1]["structure"], np.ones((20, 20))
    )
    assert _enough_tissue_mask_filters_.filters == [
        RgbToGrayscale_(),
        OtsuThreshold_(),
        BinaryDilation_(),
        BinaryFillHoles_(),
    ]
    assert type(_enough_tissue_mask_filters_) == Compose


def it_knows_tissue_areas_mask_slide_filters_composition(
    RgbToGrayscale_,
    OtsuThreshold_,
    BinaryDilation_,
    RemoveSmallHoles_,
    RemoveSmallObjects_,
):
    _enough_tissue_mask_filters_ = FiltersComposition(Slide).tissue_mask_filters

    RgbToGrayscale_.assert_called_once()
    OtsuThreshold_.assert_called_once()
    BinaryDilation_.assert_called_once()
    RemoveSmallHoles_.assert_called_once()
    RemoveSmallObjects_.assert_called_once()

    assert _enough_tissue_mask_filters_.filters == [
        RgbToGrayscale_(),
        OtsuThreshold_(),
        BinaryDilation_(),
        RemoveSmallHoles_(),
        RemoveSmallObjects_(),
    ]

    assert type(_enough_tissue_mask_filters_) == Compose


def it_knows_tissue_areas_mask_compose_filters_composition(
    RgbToGrayscale_,
    OtsuThreshold_,
    BinaryDilation_,
    RemoveSmallHoles_,
    RemoveSmallObjects_,
):
    filters = [
        RgbToGrayscale_(),
        OtsuThreshold_(),
        BinaryDilation_(),
        RemoveSmallHoles_(),
        RemoveSmallObjects_(),
    ]

    _enough_tissue_mask_filters_ = FiltersComposition(
        Compose, *filters
    ).tissue_mask_filters

    assert type(_enough_tissue_mask_filters_) == Compose
    assert list(_enough_tissue_mask_filters_.filters) == filters


@pytest.mark.parametrize(
    "_cls, subclass",
    ((Tile, _TileFiltersComposition), (Slide, _SlideFiltersComposition)),
)
def it_can_dispatch_subclass_according_class_type(request, _cls, subclass):
    _init_ = initializer_mock(request, FiltersComposition)

    filters_composition = FiltersComposition(_cls)

    _init_.assert_called_once_with(_cls)
    assert isinstance(filters_composition, subclass)


def it_raises_filtercompositionerror_if_class_not_allowed(request):
    _init_ = initializer_mock(request, FiltersComposition)
    cls_ = NotAllowedInFiltersComposition

    with pytest.raises(FilterCompositionError) as err:
        FiltersComposition(cls_)

    _init_.assert_not_called()
    assert isinstance(err.value, FilterCompositionError)
    assert (
        str(err.value)
        == "Filters composition for the class NotAllowedInFiltersComposition \
is not available"
    )


def it_raises_filtercompositionerror_if_class_is_none(request):
    _init_ = initializer_mock(request, FiltersComposition)
    cls_ = None

    with pytest.raises(FilterCompositionError) as err:
        FiltersComposition(cls_)

    _init_.assert_not_called()
    assert isinstance(err.value, FilterCompositionError)
    assert str(err.value) == "cls_ parameter cannot be None"


def it_raises_filtercompositionerror_if_compose_is_specified_without_filters(
    request,
):
    _init_ = initializer_mock(request, FiltersComposition)
    cls_ = Compose

    with pytest.raises(FilterCompositionError) as err:
        FiltersComposition(cls_)

    _init_.assert_not_called()
    assert isinstance(err.value, FilterCompositionError)
    assert str(err.value) == "Custom filter pipeline requires at least one filter"


# fixture components ---------------------------------------------


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


@pytest.fixture
def RemoveSmallHoles_(request):
    return class_mock(
        request, "histolab.filters.morphological_filters.RemoveSmallHoles"
    )


@pytest.fixture
def RemoveSmallObjects_(request):
    return class_mock(
        request, "histolab.filters.morphological_filters.RemoveSmallObjects"
    )
