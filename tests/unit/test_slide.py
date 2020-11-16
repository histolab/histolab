# encoding: utf-8

import errno
import math
import os
from collections import namedtuple
from unittest.mock import call

import numpy as np
import openslide
import PIL
import pytest
from PIL import ImageShow

from histolab.exceptions import LevelError
from histolab.filters.compositions import _SlideFiltersComposition
from histolab.filters.image_filters import Compose
from histolab.slide import Slide, SlideSet
from histolab.types import CP, Region
from histolab.util import regions_from_binary_mask

from ..unitutil import (
    ANY,
    PILIMG,
    class_mock,
    dict_list_eq,
    function_mock,
    initializer_mock,
    instance_mock,
    is_win32,
    method_mock,
    on_ci,
    property_mock,
)


class Describe_Slide:
    def it_constructs_from_args(self, request):
        _init_ = initializer_mock(request, Slide)
        _slide_path = "/foo/bar/myslide.svs"
        _processed_path = "/foo/bar/myslide/processed"

        slide = Slide(_slide_path, _processed_path)

        _init_.assert_called_once_with(ANY, _slide_path, _processed_path)
        assert isinstance(slide, Slide)

    def but_it_has_wrong_slide_path_type(self):
        """This test simulates a wrong user behaviour, using a None object instead of a
        str, or a path as slide_path param"""
        with pytest.raises(TypeError) as err:
            slide = Slide(None, "prc")
            slide.name

        assert isinstance(err.value, TypeError)
        assert (
            str(err.value) == "expected str, bytes or os.PathLike object, not NoneType"
        )

    def or_it_has_wrong_processed_path(self, request):
        """This test simulates a wrong user behaviour, using a None object instead of a
        str, or a path as processed_path param"""
        _resampled_dimensions = method_mock(request, Slide, "_resampled_dimensions")
        _resampled_dimensions.return_value = (1, 2, 3, 4)
        with pytest.raises(TypeError) as err:
            slide = Slide("path", None)
            slide.scaled_image_path(32)

        assert isinstance(err.value, TypeError)
        assert (
            str(err.value) == "expected str, bytes or os.PathLike object, not NoneType"
        )

    @pytest.mark.parametrize(
        "resampled_dims, dir_path, slide_path, proc_path, scale_factor, expected_path",
        (
            (
                (245, 123, 145, 99),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                os.path.join("/foo/bar/b/0/9", "myslide-64x-245x123-145x99.png"),
            ),
            (
                (245, 123, 145, 99),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                32,
                os.path.join("/foo/bar/b/0/9", "myslide-32x-245x123-145x99.png"),
            ),
            (
                (None, None, None, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                os.path.join("/foo/bar/b/0/9", "myslide*.png"),
            ),
            (
                (None, 234, 192, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                os.path.join("/foo/bar/b/0/9", "myslide-64x-Nonex234-192xNone.png"),
            ),
            (
                (123, 234, 192, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                os.path.join("/foo/bar/b/0/9", "myslide-64x-123x234-192xNone.png"),
            ),
            (
                (None, None, 192, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                os.path.join("/foo/bar/b/0/9", "myslide-64x-NonexNone-192xNone.png"),
            ),
        ),
    )
    def it_generates_the_correct_breadcrumb(
        self,
        request,
        resampled_dims,
        dir_path,
        slide_path,
        proc_path,
        scale_factor,
        expected_path,
    ):
        _resampled_dimensions = method_mock(request, Slide, "_resampled_dimensions")
        _resampled_dimensions.return_value = resampled_dims
        slide = Slide(slide_path, proc_path)

        _breadcrumb = slide._breadcrumb(dir_path, scale_factor)

        assert _breadcrumb == expected_path

    @pytest.mark.parametrize(
        "slide_path, expected_value",
        (("/foo/bar/myslide.svs", "myslide"), ("/foo/myslide.svs", "myslide")),
    )
    def it_knows_its_name(self, slide_path, expected_value):
        slide = Slide(slide_path, "processed/")

        name = slide.name

        assert name == expected_value

    @pytest.mark.parametrize(
        "slide_path, proc_path, slide_dims, expected_value",
        (
            (
                "1.svs",
                "1/p",
                (345, 111, 333, 444),
                os.path.join("1/p", "1-22x-345x111-333x444.png"),
            ),
            (
                "2.svs",
                "2/p",
                (345, 111, None, None),
                os.path.join("2/p", "2-22x-345x111-NonexNone.png"),
            ),
            (
                "2.svs",
                "2/p",
                (345, 111, 123, 123),
                os.path.join("2/p", "2-22x-345x111-123x123.png"),
            ),
            ("2.svs", "2/p", (None, None, None, None), os.path.join("2/p", "2*.png")),
        ),
    )
    def it_knows_its_scaled_image_path(
        self, resampled_dims_, slide_path, proc_path, slide_dims, expected_value
    ):
        resampled_dims_.return_value = slide_dims
        slide = Slide(slide_path, proc_path)

        scaled_img_path = slide.scaled_image_path(scale_factor=22)

        assert scaled_img_path == expected_value

    def it_knows_its_thumbnails_path(self, resampled_dims_):
        slide_path, proc_path, slide_dims, expected_value = (
            "/foo/bar/myslide.svs",
            "/foo/bar/myslide/processed",
            (345, 111, 333, 444),
            os.path.join("/foo/bar/myslide/processed", "thumbnails", "myslide.png"),
        )
        resampled_dims_.return_value = slide_dims
        slide = Slide(slide_path, proc_path)

        thumbnail_path = slide.thumbnail_path

        assert thumbnail_path == expected_value

    def it_knows_its_dimensions(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")

        slide_dims = slide.dimensions

        assert slide_dims == (500, 500)

    def it_knows_its_resampled_dimensions(self, dimensions_):
        """This test prove that given the dimensions (mock object here), it does
        the correct maths operations:
            `large_w, large_h = self.dimensions`
            `new_w = math.floor(large_w / self._scale_factor)`
            `new_h = math.floor(large_h / self._scale_factor)`
            `return large_w, large_h, new_w, new_h`
        """
        dimensions_.return_value = (300, 500)
        slide = Slide("/a/b/foo", "processed")

        _resampled_dims = slide._resampled_dimensions(scale_factor=32)

        assert _resampled_dims == (300, 500, 9, 15)

    def but_it_raises_zero_division_error_when_scalefactor_is_0(self, dimensions_):
        """Considering the teset above, this one prove that a wrong behaviour of the
        user can cause a zerodivision error. In this case the scale_factor=0 generates
        the ZeroDivision Exception
        """
        dimensions_.return_value = (300, 500)
        slide = Slide("/a/b/foo", "processed")
        with pytest.raises(ZeroDivisionError) as err:
            slide._resampled_dimensions(scale_factor=0)

        assert isinstance(err.value, ZeroDivisionError)
        assert str(err.value) == "division by zero"

    def it_knows_its_resampled_array(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        resampled_dims_.return_value = (100, 200, 300, 400)

        resampled_array = slide.resampled_array(scale_factor=32)

        assert type(resampled_array) == np.ndarray
        assert resampled_array.shape == (400, 300, 3)

    def it_knows_its_thumbnail_size(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")

        thumb_size = slide._thumbnail_size

        assert thumb_size == (500, 500)

    def it_creates_a_correct_slide_object(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_50X50_155_0_0
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")

        _wsi = slide._wsi

        assert type(_wsi) in (openslide.OpenSlide, openslide.ImageSlide)

    def but_it_raises_an_exception_if_file_not_found(self):
        with pytest.raises(FileNotFoundError) as err:
            slide = Slide("wrong/path/fake.wsi", "processed")
            slide._wsi

        assert isinstance(err.value, FileNotFoundError)
        assert str(err.value) == "The wsi path resource doesn't exist"

    def or_it_raises_an_PIL_exception(self, tmpdir):
        slide_path = tmpdir.mkdir("sub").join("hello.txt")
        slide_path.write("content")
        with pytest.raises(PIL.UnidentifiedImageError) as err:
            slide = Slide(os.path.join(slide_path), "processed")
            slide._wsi

        assert isinstance(err.value, PIL.UnidentifiedImageError)
        assert (
            str(err.value) == "Your wsi has something broken inside, a doctor is needed"
        )

    def it_can_resample_itself(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        resampled_dims_.return_value = (100, 200, 300, 400)

        _resample = slide._resample(32)

        # image array assertions
        assert type(_resample[1]) == np.ndarray
        # ---The np array shape should be (new_h X new_w X channels),---
        # ---in this case (look at resampled_dims mock) the new_h is 400---
        # ---the new_w is 300 and the color channels of the image are 3---
        assert _resample[1].shape == (400, 300, 3)
        # ---Here we prove that the 3 channels are compliant with the color---
        # ---definition and that each channel is a np.array (400x300) filled---
        # ---with the related color expressed during the image creation---
        np.testing.assert_almost_equal(_resample[1][:, :, 0], np.full((400, 300), 155))
        np.testing.assert_almost_equal(_resample[1][:, :, 1], np.full((400, 300), 249))
        np.testing.assert_almost_equal(_resample[1][:, :, 2], np.full((400, 300), 240))
        # PIL image assertions
        assert type(_resample[0]) == PIL.Image.Image
        assert _resample[0].size == (300, 400)
        assert _resample[0].width == 300
        assert _resample[0].height == 400
        assert _resample[0].mode == "RGB"

    def it_resamples_with_the_correct_scale_factor(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        resampled_dims_.return_value = (500, 500, 15, 15)

        _resample = slide._resample(32)

        assert _resample[1].shape == (math.floor(500 / 32), math.floor(500 / 32), 3)

    def it_can_save_scaled_image(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        resampled_dims_.return_value = (100, 200, 300, 400)

        slide.save_scaled_image(32)

        assert slide.scaled_image_path(32) == os.path.join(
            tmp_path_, "processed", "mywsi-32x-100x200-300x400.png"
        )
        assert os.path.exists(os.path.join(tmp_path_, slide.scaled_image_path(32)))

    def it_can_save_thumbnail(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        resampled_dims_.return_value = (100, 200, 300, 400)

        slide.save_thumbnail()

        assert slide.thumbnail_path == os.path.join(
            tmp_path_, "processed", "thumbnails", "mywsi.png"
        )
        assert os.path.exists(os.path.join(tmp_path_, slide.thumbnail_path))

    def it_knows_regions_from_binary_mask(self, request):
        binary_mask = np.array([[True, False], [True, True]])
        label = function_mock(request, "histolab.util.label")
        regionprops = function_mock(request, "histolab.util.regionprops")
        RegionProps = namedtuple("RegionProps", ("area", "bbox", "centroid"))
        regions_props = [
            RegionProps(3, (0, 0, 2, 2), (0.6666666666666666, 0.3333333333333333))
        ]
        regionprops.return_value = regions_props
        label(binary_mask).return_value = [[0, 1], [1, 1]]

        regions_from_binary_mask_ = regions_from_binary_mask(binary_mask)

        regionprops.assert_called_once_with(label(binary_mask))
        assert type(regions_from_binary_mask_) == list
        assert len(regions_from_binary_mask_) == 1
        assert type(regions_from_binary_mask_[0]) == Region
        assert regions_from_binary_mask_ == [
            Region(
                index=0,
                area=regions_props[0].area,
                bbox=regions_props[0].bbox,
                center=regions_props[0].centroid,
            )
        ]

    def it_knows_its_biggest_regions(self):
        regions = [
            Region(index=0, area=14, bbox=(0, 0, 2, 2), center=(0.5, 0.5)),
            Region(index=1, area=2, bbox=(0, 0, 2, 2), center=(0.5, 0.5)),
            Region(index=2, area=5, bbox=(0, 0, 2, 2), center=(0.5, 0.5)),
            Region(index=3, area=10, bbox=(0, 0, 2, 2), center=(0.5, 0.5)),
        ]
        slide = Slide("a/b", "c/d")

        biggest_regions = slide._biggest_regions(regions, 2)

        assert biggest_regions == [regions[0], regions[3]]

    @pytest.mark.parametrize("n", (0, 6))
    def but_it_raises_an_error_when_n_is_not_between_1_and_number_of_regions(self, n):
        regions = [Region(i, i + 1, (0, 0, 2, 2), (0.5, 0.5)) for i in range(4)]
        slide = Slide("a/b", "c/d")
        with pytest.raises(ValueError) as err:
            slide._biggest_regions(regions, n)

        assert str(err.value) == f"n should be between 1 and {len(regions)}, got {n}"

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
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
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
            request, "histolab.slide.regions_from_binary_mask"
        )
        regions_from_binary_mask.return_value = regions
        biggest_regions_ = method_mock(
            request, Slide, "_biggest_regions", autospec=False
        )
        biggest_regions_.return_value = regions
        region_coordinates_ = function_mock(
            request, "histolab.slide.region_coordinates"
        )
        region_coordinates_.return_values = CP(0, 0, 2, 2)
        polygon_to_mask_array_ = function_mock(
            request, "histolab.util.polygon_to_mask_array"
        )
        polygon_to_mask_array_((1000, 1000), CP(0, 0, 2, 2)).return_value = [
            [True, True],
            [False, True],
        ]

        biggest_mask_tissue_box = slide.biggest_tissue_box_mask

        region_coordinates_.assert_called_once_with(regions[0])
        biggest_regions_.assert_called_once_with(regions, n=1)
        polygon_to_mask_array_.assert_called_once_with(
            (1000, 1000), CP(x_ul=0, y_ul=0, x_br=2, y_br=2)
        )
        np.testing.assert_almost_equal(biggest_mask_tissue_box, np.zeros((500, 500)))

    @pytest.mark.skipif(
        not on_ci() or is_win32(), reason="Only run on CIs; hangs on Windows CIs"
    )
    def it_can_show_its_thumbnail(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")

        slide.save_thumbnail()

        assert ImageShow.show(PIL.Image.open(slide.thumbnail_path))

    def but_it_raises_error_when_it_doesnt_exist(self):
        slide = Slide("a/b", "processed")

        with pytest.raises(FileNotFoundError) as err:
            slide.show()

        assert (
            str(err.value)
            == "Cannot display the slide thumbnail:[Errno 2] No such file or "
            f"directory: {repr(os.path.join('processed', 'thumbnails', 'b.png'))}"
        )

    def it_knows_its_level_dimensions(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")

        level_dimensions = slide.level_dimensions(level=0)

        assert level_dimensions == (500, 500)

    def but_it_raises_expection_when_level_does_not_exist(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        with pytest.raises(LevelError) as err:
            slide.level_dimensions(level=3)

        assert isinstance(err.value, LevelError)
        assert str(err.value) == "Level 3 not available. Number of available levels: 1"

    @pytest.mark.parametrize(
        "coords, expected_result",
        (
            (CP(0, 128, 0, 128), True),
            (CP(800000, 90000, 8000010, 90010), False),
            (CP(800000, 90000, -1, 90010), False),
        ),
    )
    def it_knows_if_coords_are_valid(self, coords, expected_result, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")

        _are_valid = slide._has_valid_coords(coords)

        assert type(_are_valid) == bool
        assert _are_valid == expected_result

    def it_knows_its_levels(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")

        levels = slide.levels

        assert type(levels) == list
        assert levels == [0]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def resampled_dims_(self, request):
        return method_mock(request, Slide, "_resampled_dimensions")

    @pytest.fixture
    def dimensions_(self, request):
        return property_mock(request, Slide, "dimensions")

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


class Describe_Slideset:
    def it_constructs_from_args(self, request):
        _init_ = initializer_mock(request, SlideSet)
        _slides_path = "/foo/bar/"
        _processed_path = "/foo/bar/wsislides/processed"
        _valid_extensions = [".svs", ".tiff"]

        slideset = SlideSet(_slides_path, _processed_path, _valid_extensions)

        _init_.assert_called_once_with(
            ANY, _slides_path, _processed_path, _valid_extensions
        )
        assert isinstance(slideset, SlideSet)

    def it_can_constructs_slides(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        slides_ = tuple(instance_mock(request, Slide) for _ in range(10))
        _slides_ = property_mock(request, SlideSet, "slides")
        _slides_.return_value = slides_
        slideset = SlideSet(tmp_path_, os.path.join(tmp_path_, "b"), [".svs"])

        slides = slideset.slides

        _slides_.assert_called_once_with()
        assert len(slides) == 10

    def it_knows_its_slides(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, "proc", [".svs"])

        slides = slideset.slides

        assert len(slides) == 1

        slideset = SlideSet(None, "proc", [".svs"])

        slides = slideset.slides

        assert len(slides) == 0

        with pytest.raises(FileNotFoundError) as err:
            slideset = SlideSet("fake/path", "proc", [".svs"])
            slides = slideset.slides

        assert isinstance(err.value, FileNotFoundError)
        assert err.value.errno == errno.ENOENT

    def it_constructs_its_sequence_of_slides_to_help(self, request, Slide_, tmpdir):
        slides_path = tmpdir.mkdir("mypath")
        for i in range(4):
            open(os.path.join(slides_path, f"myfile{i}.svs"), "a")
        slides_ = tuple(instance_mock(request, Slide) for _ in range(4))
        Slide_.side_effect = iter(slides_)
        slide_set = SlideSet(
            slides_path=slides_path,
            processed_path=os.path.join(slides_path, "processed"),
            valid_extensions=[".svs"],
        )
        slides = tuple(slide_set.slides)

        assert sorted(Slide_.call_args_list) == sorted(
            [
                call(
                    os.path.join(slides_path, f"myfile{i}.svs"),
                    os.path.join(slides_path, "processed"),
                )
                for i in range(4)
            ]
        )
        assert slides == slides_

    def it_knows_the_slides_dimensions(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        image2 = PILIMG.RGBA_COLOR_50X50_155_0_0
        image2.save(os.path.join(tmp_path_, "mywsi2.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, "proc", [".svs"])

        slides_dimensions = slideset._slides_dimensions

        expected_value = [
            {"slide": "mywsi", "width": 500, "height": 500, "size": 250000},
            {"slide": "mywsi2", "width": 50, "height": 50, "size": 2500},
        ]
        assert dict_list_eq(slides_dimensions, expected_value) is True

    def it_knows_its_slides_dimensions_list(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        image2 = PILIMG.RGBA_COLOR_50X50_155_0_0
        image2.save(os.path.join(tmp_path_, "mywsi2.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, "proc", [".svs"])

        _slides_dimensions_list = slideset._slides_dimensions_list

        assert sorted(_slides_dimensions_list) == sorted([(500, 500), (50, 50)])

    def it_knows_its_total_slides(self, request, Slide_):
        slides = property_mock(request, SlideSet, "slides")
        slides.return_value = [Slide_ for _ in range(4)]
        slideset = SlideSet("the/path", "proc", [".svs"])

        total_slides = slideset.total_slides

        assert total_slides == 4

    def it_knows_its_avg_width_slide(self, _slides_dimensions_prop, total_slides_prop):
        total_slides_prop.return_value = 2
        _slides_dimensions_prop.return_value = [
            {"slide": "mywsi", "width": 500, "height": 500, "size": 250000},
            {"slide": "mywsi2", "width": 50, "height": 50, "size": 2500},
        ]
        slideset = SlideSet("fake/path", "proc", [".svs"])

        _avg_width_slide = slideset._avg_width_slide

        assert _avg_width_slide == 275.0
        assert (
            _avg_width_slide
            == sum(d["width"] for d in _slides_dimensions_prop.return_value) / 2
        )

    def it_knows_its_avg_height_slide(self, _slides_dimensions_prop, total_slides_prop):
        total_slides_prop.return_value = 2
        _slides_dimensions_prop.return_value = [
            {"slide": "mywsi", "width": 500, "height": 100, "size": 250000},
            {"slide": "mywsi2", "width": 50, "height": 50, "size": 2500},
        ]
        slideset = SlideSet("fake/path", "proc", [".svs"])

        _avg_height_slide = slideset._avg_height_slide

        assert _avg_height_slide == 75.0
        assert (
            _avg_height_slide
            == sum(d["height"] for d in _slides_dimensions_prop.return_value) / 2
        )

    def it_knows_its_avg_size_slide(self, _slides_dimensions_prop, total_slides_prop):
        total_slides_prop.return_value = 2
        _slides_dimensions_prop.return_value = [
            {"slide": "mywsi", "width": 500, "height": 100, "size": 250000},
            {"slide": "mywsi2", "width": 50, "height": 50, "size": 2500},
        ]
        slideset = SlideSet("fake/path", "proc", [".svs"])

        _avg_size_slide = slideset._avg_size_slide

        assert _avg_size_slide == 126250.0
        assert (
            _avg_size_slide
            == sum(d["size"] for d in _slides_dimensions_prop.return_value) / 2
        )

    def it_knows_its_max_height_slide(self, _slides_dimensions_prop):
        _slides_dimensions_prop.return_value = [
            {"slide": "mywsi", "width": 500, "height": 100, "size": 250000},
            {"slide": "mywsi2", "width": 50, "height": 50, "size": 2500},
        ]
        slideset = SlideSet("fake/path", "proc", [".svs"])

        _max_height_slide = slideset._max_height_slide

        assert _max_height_slide == {"slide": "mywsi", "height": 100}

    def it_knows_its_max_size_slide(self, _slides_dimensions_prop):
        _slides_dimensions_prop.return_value = [
            {"slide": "mywsi", "width": 500, "height": 100, "size": 250000},
            {"slide": "mywsi2", "width": 50, "height": 50, "size": 2500},
        ]
        slideset = SlideSet("fake/path", "proc", [".svs"])

        _max_size_slide = slideset._max_size_slide

        assert _max_size_slide == {"slide": "mywsi", "size": 250000}

    def it_knows_its_max_width_slide(self, _slides_dimensions_prop):
        _slides_dimensions_prop.return_value = [
            {"slide": "mywsi", "width": 500, "height": 100, "size": 250000},
            {"slide": "mywsi2", "width": 600, "height": 50, "size": 2500},
        ]
        slideset = SlideSet("fake/path", "proc", [".svs"])

        _max_width_slide = slideset._max_width_slide

        assert _max_width_slide == {"slide": "mywsi2", "width": 600}

    def it_knows_its_min_width_slide(self, _slides_dimensions_prop):
        _slides_dimensions_prop.return_value = [
            {"slide": "mywsi", "width": 500, "height": 100, "size": 250000},
            {"slide": "mywsi2", "width": 600, "height": 50, "size": 2500},
        ]
        slideset = SlideSet("fake/path", "proc", [".svs"])

        _min_width_slide = slideset._min_width_slide

        assert _min_width_slide == {"slide": "mywsi", "width": 500}

    def it_knows_its_min_height_slide(self, _slides_dimensions_prop):
        _slides_dimensions_prop.return_value = [
            {"slide": "mywsi", "width": 500, "height": 100, "size": 250000},
            {"slide": "mywsi2", "width": 600, "height": 50, "size": 2500},
        ]
        slideset = SlideSet("fake/path", "proc", [".svs"])

        _min_height_slide = slideset._min_height_slide

        assert _min_height_slide == {"slide": "mywsi2", "height": 50}

    def it_knows_its_min_size_slide(self, _slides_dimensions_prop):
        _slides_dimensions_prop.return_value = [
            {"slide": "mywsi", "width": 500, "height": 100, "size": 250000},
            {"slide": "mywsi2", "width": 600, "height": 50, "size": 2500},
        ]
        slideset = SlideSet("fake/path", "proc", [".svs"])

        _min_size_slide = slideset._min_size_slide

        assert _min_size_slide == {"slide": "mywsi2", "size": 2500}

    def it_can_save_scaled_slides(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        slide1 = instance_mock(request, Slide)
        slide2 = instance_mock(request, Slide)

        slideset = SlideSet(tmp_path_, os.path.join(tmp_path_, "processed"), [])
        slides = property_mock(request, SlideSet, "slides")
        slides.return_value = [slide1, slide2]
        slideset.save_scaled_slides(32, 2)

        slide1.save_scaled_image.assert_called_once_with(32)
        slide2.save_scaled_image.assert_called_once_with(32)

    def it_can_save_thumbnails(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        slide1 = instance_mock(request, Slide)
        slide2 = instance_mock(request, Slide)

        slideset = SlideSet(tmp_path_, os.path.join(tmp_path_, "processed"), [])
        slides = property_mock(request, SlideSet, "slides")
        slides.return_value = [slide1, slide2]
        slideset.save_thumbnails(2)

        slide1.save_thumbnail.assert_called_once_with()
        slide2.save_thumbnail.assert_called_once_with()

    def it_generates_slides_stats(self, total_slides_prop, tmpdir):
        total_slides_prop.return_value = 2
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        image2 = PILIMG.RGBA_COLOR_50X50_155_0_0
        image2.save(os.path.join(tmp_path_, "mywsi2.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, os.path.join("proc"), [".svs"])

        slides_stats = slideset.slides_stats

        assert slides_stats == {
            "no_of_slides": 2,
            "max_width": {"slide": "mywsi", "width": 500},
            "max_height": {"slide": "mywsi", "height": 500},
            "max_size": {"slide": "mywsi", "size": 250000},
            "min_width": {"slide": "mywsi2", "width": 50},
            "min_height": {"slide": "mywsi2", "height": 50},
            "min_size": {"slide": "mywsi2", "size": 2500},
            "avg_width": 275.0,
            "avg_height": 275.0,
            "avg_size": 126250.0,
        }

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _slides_dimensions_prop(self, request):
        return property_mock(request, SlideSet, "_slides_dimensions")

    @pytest.fixture
    def total_slides_prop(self, request):
        return property_mock(request, SlideSet, "total_slides")

    @pytest.fixture
    def _slides_dimensions_list_prop(self, request):
        return property_mock(request, SlideSet, "_slides_dimensions_list")

    @pytest.fixture
    def Slide_(self, request):
        return class_mock(request, "histolab.slide.Slide")
