import numpy as np
import pytest

from histolab.filters.compositions import _SlideFiltersComposition
from histolab.filters.image_filters import Compose
from histolab.masks import BiggestTissueBoxMask
from histolab.types import CP, Region
from tests.unitutil import (
    PILIMG,
    base_test_slide,
    class_mock,
    function_mock,
    method_mock,
    property_mock,
)


class DescribeBiggestTissueBoxMask:
    def it_knows_its_biggest_regions(self):
        regions = [
            Region(index=0, area=14, bbox=(0, 0, 2, 2), center=(0.5, 0.5)),
            Region(index=1, area=2, bbox=(0, 0, 2, 2), center=(0.5, 0.5)),
            Region(index=2, area=5, bbox=(0, 0, 2, 2), center=(0.5, 0.5)),
            Region(index=3, area=10, bbox=(0, 0, 2, 2), center=(0.5, 0.5)),
        ]
        binary_mask = BiggestTissueBoxMask

        biggest_regions = binary_mask._regions(regions, 2)
        assert biggest_regions == [regions[0], regions[3]]

    def it_knows_its_biggest_tissue_box_mask(
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
        regions = [Region(index=0, area=33, bbox=(0, 0, 2, 2), center=(0.5, 0.5))]
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
        polygon_to_mask_array_ = function_mock(
            request, "histolab.util.polygon_to_mask_array"
        )
        polygon_to_mask_array_((1000, 1000), CP(0, 0, 2, 2)).return_value = [
            [True, True],
            [False, True],
        ]

        biggest_mask_tissue_box = BiggestTissueBoxMask()

        np.testing.assert_almost_equal(
            biggest_mask_tissue_box(slide), np.zeros((500, 500))
        )
        region_coordinates_.assert_called_once_with(regions[0])
        biggest_regions_.assert_called_once_with(regions, n=1)
        polygon_to_mask_array_.assert_called_once_with(
            (1000, 1000), CP(x_ul=0, y_ul=0, x_br=2, y_br=2)
        )

    # fixture components ---------------------------------------------

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
    def RemoveSmallHoles_(self, request):
        return class_mock(
            request, "histolab.filters.morphological_filters.RemoveSmallHoles"
        )

    @pytest.fixture
    def RemoveSmallObjects_(self, request):
        return class_mock(
            request, "histolab.filters.morphological_filters.RemoveSmallObjects"
        )
