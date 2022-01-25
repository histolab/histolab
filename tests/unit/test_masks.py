import numpy as np
import PIL
import pytest

from histolab.filters.compositions import _SlideFiltersComposition
from histolab.filters.image_filters import Compose, ImageFilter
from histolab.filters.morphological_filters import RemoveSmallObjects
from histolab.masks import BiggestTissueBoxMask, TissueMask
from histolab.tile import Tile
from histolab.types import CP, Region

from ..unitutil import (
    PILIMG,
    base_test_slide,
    class_mock,
    function_mock,
    method_mock,
    property_mock,
)


class CustomFilterForTest(ImageFilter):
    def __call__(self, _: PIL.Image.Image) -> np.ndarray:
        raise RuntimeError("This call needs mocking")


class DescribeBiggestTissueBoxMask:
    def it_knows_its_biggest_regions(self):
        regions = [
            Region(index=0, area=14, bbox=(0, 0, 2, 2), center=(0.5, 0.5), coords=None),
            Region(index=1, area=2, bbox=(0, 0, 2, 2), center=(0.5, 0.5), coords=None),
            Region(index=2, area=5, bbox=(0, 0, 2, 2), center=(0.5, 0.5), coords=None),
            Region(index=3, area=10, bbox=(0, 0, 2, 2), center=(0.5, 0.5), coords=None),
        ]
        binary_mask = BiggestTissueBoxMask

        biggest_regions = binary_mask._regions(regions, 2)
        assert biggest_regions == [regions[0], regions[3]]

    @pytest.mark.parametrize(
        "n, expected_message",
        (
            (0, "Number of regions must be greater than 0, got 0."),
            (3, "n should be smaller than the number of regions [0], got 3"),
        ),
    )
    def but_it_raises_exception_when_number_of_regions_is_wrong(
        self, n, expected_message
    ):
        binary_mask = BiggestTissueBoxMask()

        with pytest.raises(ValueError) as e:
            binary_mask._regions([], n)

        assert str(e.value) == expected_message

    def it_knows_its_mask(
        self,
        request,
        tmpdir,
        RgbToGrayscale_,
        OtsuThreshold_,
        BinaryDilation_,
        RemoveSmallHoles_,
        RemoveSmallObjects_,
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        regions = [
            Region(index=0, area=33, bbox=(0, 0, 2, 2), center=(0.5, 0.5), coords=None)
        ]
        main_tissue_areas_mask_filters_ = property_mock(
            request, _SlideFiltersComposition, "tissue_mask_filters"
        )
        main_tissue_areas_mask_filters_.return_value = Compose(
            [
                RgbToGrayscale_,
                OtsuThreshold_,
                BinaryDilation_,
                RemoveSmallHoles_,
                RemoveSmallObjects_,
            ]
        )
        regions_from_binary_mask = function_mock(
            request, "histolab.masks.regions_from_binary_mask"
        )
        regions_from_binary_mask.return_value = regions
        biggest_regions_ = method_mock(
            request, BiggestTissueBoxMask, "_regions", autospec=False
        )
        biggest_regions_.return_value = regions
        region_coordinates_ = function_mock(
            request, "histolab.masks.region_coordinates"
        )
        region_coordinates_.return_values = CP(0, 0, 2, 2)
        rectangle_to_mask_ = function_mock(request, "histolab.util.rectangle_to_mask")
        rectangle_to_mask_((1000, 1000), CP(0, 0, 2, 2)).return_value = [
            [True, True],
            [False, True],
        ]
        biggest_mask_tissue_box = BiggestTissueBoxMask()

        binary_mask = biggest_mask_tissue_box(slide)

        np.testing.assert_almost_equal(binary_mask, np.zeros((500, 500)))
        region_coordinates_.assert_called_once_with(regions[0])
        biggest_regions_.assert_called_once_with(regions, n=1)
        rectangle_to_mask_.assert_called_once_with(
            (1000, 1000), CP(x_ul=0, y_ul=0, x_br=2, y_br=2)
        )


class DescribeTissueMask:
    def it_knows_its_mask_slide(
        self,
        request,
        tmpdir,
        RgbToGrayscale_,
        OtsuThreshold_,
        BinaryDilation_,
        RemoveSmallHoles_,
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        main_tissue_areas_mask_filters_ = property_mock(
            request, _SlideFiltersComposition, "tissue_mask_filters"
        )
        expected_mask = [
            [True, True],
            [False, True],
        ]
        remove_small_objects = method_mock(request, RemoveSmallObjects, "__call__")
        remove_small_objects.return_value = expected_mask
        main_tissue_areas_mask_filters_.return_value = Compose(
            [
                RgbToGrayscale_,
                OtsuThreshold_,
                BinaryDilation_,
                RemoveSmallHoles_,
                RemoveSmallObjects(),
            ]
        )
        tissue_mask = TissueMask()
        binary_mask = tissue_mask(slide)

        assert binary_mask == expected_mask

    def it_knows_its_mask_slide_and_supports_custom_filters(self, request, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        custom_filters = [CustomFilterForTest()]
        custom_filter_call_ = method_mock(request, CustomFilterForTest, "__call__")
        expected_mask = [
            [True, True],
            [False, True],
        ]
        custom_filter_call_.return_value = expected_mask

        tissue_mask = TissueMask(*custom_filters)
        binary_mask = tissue_mask(slide)

        assert binary_mask == expected_mask
        # Check the filter has been called once
        custom_filter_call_.assert_called_once_with(custom_filters[0], slide.thumbnail)

    def it_knows_its_mask_tile(self, request):
        tile = Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, None, None)
        tissue_mask_tile_ = property_mock(request, Tile, "tissue_mask")
        expected_mask = [
            [True, True],
            [False, True],
        ]
        tissue_mask_tile_.return_value = expected_mask

        tissue_mask = TissueMask()
        binary_mask = tissue_mask(tile)

        assert binary_mask == expected_mask

    def it_knows_its_mask_tile_and_supports_custom_filters(self, request):
        tile = Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, None, None)
        custom_filters = [CustomFilterForTest()]
        custom_filter_call_ = method_mock(request, CustomFilterForTest, "__call__")
        # We need to account for the fact padding is added before filtering
        expected_mask = np.zeros((510, 510), dtype=np.bool)
        custom_filter_call_.return_value = expected_mask

        # Call the tissue mask with custom filters
        tissue_mask = TissueMask(*custom_filters)
        binary_mask = tissue_mask(tile)

        # Due to the padding in 'calculate_tissue_mask', a direct
        # check with 'assert_called_once_with' is not possible.
        assert np.array_equal(binary_mask, expected_mask[:500, :500])
        custom_filter_call_.assert_called_once()
        assert len(custom_filter_call_.call_args[0]) == 2
        assert custom_filter_call_.call_args[0][0] == custom_filters[0]
        assert type(custom_filter_call_.call_args[0][1]) == PIL.Image.Image


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
def RemoveSmallHoles_(request):
    return class_mock(
        request, "histolab.filters.morphological_filters.RemoveSmallHoles"
    )


@pytest.fixture
def RemoveSmallObjects_(request):
    return class_mock(
        request, "histolab.filters.morphological_filters.RemoveSmallObjects"
    )
