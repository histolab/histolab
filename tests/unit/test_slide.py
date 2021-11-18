# encoding: utf-8

import errno
import math
import os
from pathlib import Path

import numpy as np
import openslide
import PIL
import pytest
from large_image.exceptions import TileSourceException
from PIL import ImageShow

from histolab.exceptions import LevelError, MayNeedLargeImageError, SlidePropertyError
from histolab.slide import Slide, SlideSet
from histolab.types import CP

from ..fixtures import SVS
from ..unitutil import (
    ANY,
    PILIMG,
    base_test_slide,
    call,
    class_mock,
    dict_list_eq,
    initializer_mock,
    instance_mock,
    is_win32,
    method_mock,
    on_ci,
    property_mock,
)


class Describe_Slide:
    @pytest.mark.parametrize(
        "slide_path, processed_path, use_largeimage",
        [
            ("/foo/bar/myslide.svs", "/foo/bar/myslide/processed", False),
            ("/foo/bar/myslide.svs", "/foo/bar/myslide/processed", True),
            (Path("/foo/bar/myslide.svs"), Path("/foo/bar/myslide/processed"), False),
        ],
    )
    def it_constructs_from_args(
        self, request, slide_path, processed_path, use_largeimage
    ):
        _init_ = initializer_mock(request, Slide)

        slide = Slide(slide_path, processed_path, use_largeimage=use_largeimage)

        _init_.assert_called_once_with(ANY, slide_path, processed_path, use_largeimage)
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
            Slide("path", None)

        assert isinstance(err.value, TypeError)
        assert str(err.value) == "processed_path cannot be None."

    @pytest.mark.parametrize("path_type_transform", [str, Path])
    def it_knows_its_wsi(self, tmpdir, path_type_transform):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = path_type_transform(os.path.join(tmp_path_, "mywsi.png"))
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))

        wsi = slide._wsi

        assert type(wsi) == openslide.ImageSlide

    @pytest.mark.parametrize(
        "slide_path, expected_value",
        (
            ("/foo/bar/myslide.svs", "myslide"),
            ("/foo/myslide.svs", "myslide"),
            ("/foo/name.has.dot.svs", "name.has.dot"),
        ),
    )
    def it_knows_its_name(self, slide_path, expected_value):
        slide = Slide(slide_path, "processed/")

        name = slide.name

        assert name == expected_value

    @pytest.mark.parametrize(
        "use_largeimage, properties, metadata, expected_value",
        (
            (True, {}, {"mm_x": 1}, 1000.0),
            (False, {"openslide.mpp-x": 33}, None, 33.0),
            (False, {"aperio.MPP": 33}, None, 33.0),
            (
                False,
                {"tiff.XResolution": 1000, "tiff.ResolutionUnit": "centimeter"},
                None,
                10.0,
            ),
        ),
    )
    def it_knows_its_base_mpp(
        self, request, use_largeimage, properties, metadata, expected_value
    ):
        slide = Slide("foo", "bar", use_largeimage=use_largeimage)
        property_mock(request, Slide, "properties", return_value=properties)
        property_mock(request, Slide, "_metadata", return_value=metadata)

        assert slide.base_mpp == expected_value

    def but_it_raises_large_image_error_with_unknown_mpp_without_largeimage(
        self, request
    ):
        slide = Slide("foo", "bar", use_largeimage=False)
        property_mock(request, Slide, "properties", return_value={})

        with pytest.raises(MayNeedLargeImageError) as err:
            slide.base_mpp

        assert str(err.value) == (
            "Unknown scan magnification! This slide format may be best "
            "handled using the large_image module. Consider setting "
            "use_largeimage to True when instantiating this Slide."
        )

    def and_it_raises_value_error_with_unknown_mpp_with_largeimage(self, request):
        slide = Slide("foo", "bar", use_largeimage=True)
        property_mock(request, Slide, "_metadata", return_value={})

        with pytest.raises(ValueError) as err:
            slide.base_mpp

        assert str(err.value) == (
            "Unknown scan resolution! This slide is missing metadata "
            "needed for calculating the scanning resolution. Without "
            "this information, you can only ask for a tile by level, "
            "not mpp resolution."
        )

    def it_has_largeimage_tilesource(self, tmpdir):
        slide, _ = base_test_slide(
            tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240, use_largeimage=True
        )

        assert slide._tile_source.name == "pilfile"

    def it_raises_error_if_tilesource_and_not_use_largeimage(self):
        slide = Slide("/a/b/foo", "processed")

        with pytest.raises(MayNeedLargeImageError) as err:
            slide._tile_source

        assert isinstance(err.value, MayNeedLargeImageError)
        assert str(err.value) == (
            "This property uses the large_image module. Please set "
            "use_largeimage to True when instantiating this Slide."
        )

    def it_knows_its_dimensions(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)

        slide_dims = slide.dimensions

        assert slide_dims == (500, 500)

    def it_extracts_tile_with_mpp_without_image_resizing(self, tmpdir, request):
        slide = Slide(SVS.CMU_1_SMALL_REGION, tmpdir, use_largeimage=True)
        property_mock(request, PIL.Image.Image, "size", return_value=(128, 128))

        tile = slide.extract_tile(CP(0, 10, 0, 10), (128, 128), level=None, mpp=0.25)

        assert tile.image.size == (128, 128)

    def but_it_raises_a_runtime_error_when_tile_size_and_mpp_are_not_compatible(
        self, tmpdir, request
    ):
        slide = Slide(SVS.CMU_1_SMALL_REGION, tmpdir, use_largeimage=True)
        property_mock(request, PIL.Image.Image, "size", return_value=(100, 100))

        with pytest.raises(RuntimeError) as err:
            slide.extract_tile(CP(0, 10, 0, 10), (128, 128), level=None, mpp=0.25)

        assert (
            str(err.value)
            == "The tile you requested at a resolution of 0.25 MPP has a size of "
            "(100, 100), yet you specified a final `tile_size` of (128, 128), "
            "which is a very different value. When you set `mpp`, "
            "the `tile_size` parameter is used to resize fetched tiles if they are "
            "off by just 5 pixels due to rounding differences etc. Please check if "
            "you requested the right `mpp` and/or `tile_size`."
        )

    def it_raises_error_if_bad_args_for_extract_tile(self):
        slide = Slide("foo", "bar", use_largeimage=True)

        with pytest.raises(ValueError) as err:
            slide.extract_tile(CP(0, 10, 0, 10), (10, 10), level=None, mpp=None)

        assert str(err.value) == "Either level or mpp must be provided!"

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
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        resampled_dims_.return_value = (100, 200, 300, 400)

        resampled_array = slide.resampled_array(scale_factor=32)

        assert type(resampled_array) == np.ndarray
        assert resampled_array.shape == (400, 300, 3)

    def it_knows_its_thumbnail_size(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)

        thumb_size = slide._thumbnail_size

        assert thumb_size == (500, 500)

    def it_raises_error_if_thumbnail_size_and_use_largeimage(self):
        slide = Slide("/a/b/foo", "processed", use_largeimage=True)

        with pytest.raises(TileSourceException) as err:
            slide._thumbnail_size

        assert str(err.value) == "No available tilesource for /a/b/foo"

    def it_creates_a_correct_slide_object(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_50X50_155_0_0)

        _wsi = slide._wsi

        assert type(_wsi) in (openslide.OpenSlide, openslide.ImageSlide)

    def but_it_raises_an_exception_if_file_not_found(self):
        with pytest.raises(FileNotFoundError) as err:
            slide = Slide("wrong/path/fake.wsi", "processed")
            slide._wsi

        assert isinstance(err.value, FileNotFoundError)
        assert (
            str(err.value) == "The wsi path resource doesn't exist: wrong/path/fake.wsi"
        )

    def or_it_raises_an_PIL_exception(self, tmpdir):
        slide_path = tmpdir.mkdir("sub").join("hello.txt")
        slide_path.write("content")
        with pytest.raises(PIL.UnidentifiedImageError) as err:
            slide = Slide(os.path.join(slide_path), "processed")
            slide._wsi

        assert isinstance(err.value, PIL.UnidentifiedImageError)
        assert str(err.value) == (
            "This slide may be corrupted or have a non-standard format not "
            "handled by the openslide and PIL libraries. Consider setting "
            "use_largeimage to True when instantiating this Slide."
        )

    @pytest.mark.parametrize(
        "use_largeimage",
        [
            (False,),
            (True,),
        ],
    )
    def it_can_resample_itself(self, tmpdir, resampled_dims_, use_largeimage):
        slide, _ = base_test_slide(
            tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240, use_largeimage=use_largeimage
        )
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
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        resampled_dims_.return_value = (500, 500, 15, 15)

        _resample = slide._resample(32)

        assert _resample[1].shape == (math.floor(500 / 32), math.floor(500 / 32), 3)

    def it_knows_its_scaled_image(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        resampled_dims_.return_value = (100, 200, 300, 400)

        scaled_image = slide.scaled_image(32)

        assert type(scaled_image) == PIL.Image.Image

    @pytest.mark.parametrize(
        "use_largeimage",
        [
            (False,),
            (True,),
        ],
    )
    def it_knows_its_thumbnail(self, tmpdir, resampled_dims_, use_largeimage):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(
            slide_path,
            os.path.join(tmp_path_, "processed"),
            use_largeimage=use_largeimage,
        )
        resampled_dims_.return_value = (100, 200, 300, 400)

        thumb = slide.thumbnail

        assert type(thumb) == PIL.Image.Image

    @pytest.mark.skipif(
        not on_ci() or is_win32(), reason="Only run on CIs; hangs on Windows CIs"
    )
    def it_can_show_its_thumbnail(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)

        assert ImageShow.show(slide.thumbnail)

    def but_it_raises_error_when_it_doesnt_exist(self):
        slide = Slide("a/b", "processed")

        with pytest.raises(FileNotFoundError) as err:
            slide.show()

        assert (
            str(err.value)
            == "Cannot display the slide thumbnail: The wsi path resource doesn't "
            "exist: a/b"
        )

    @pytest.mark.parametrize(
        "level, expected_value", ((0, (500, 500)), (-1, (500, 500)))
    )
    def it_knows_its_level_dimensions(self, level, expected_value, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)

        level_dimensions = slide.level_dimensions(level=level)

        assert level_dimensions == expected_value

    @pytest.mark.parametrize("level", (3, -3))
    def but_it_raises_exception_when_level_does_not_exist(self, level, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)

        with pytest.raises(LevelError) as err:
            slide.level_dimensions(level=level)

        assert isinstance(err.value, LevelError)
        assert (
            str(err.value)
            == f"Level {level} not available. Number of available levels: 1"
        )

    @pytest.mark.parametrize(
        "properties",
        (
            (
                {
                    "openslide.objective-power": "20",
                    "openslide.level[1].downsample": "4.003",
                    "openslide.level[2].downsample": "16",
                }
            ),
        ),
    )
    def it_knows_its_magnification(self, request, properties, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        property_mock(request, Slide, "properties", return_value=properties)
        property_mock(request, Slide, "levels", return_value=[0, 1, 2])

        assert slide.level_magnification_factor(1) == "5.0X"
        assert slide.level_magnification_factor(2) == "1.25X"

    @pytest.mark.parametrize(
        "properties, error",
        (
            (
                {
                    "openslide.objective-power": "20",
                },
                "Downsample factor for level 1 not available. Available slide "
                "properties: ['openslide.objective-power']",
            ),
            (
                {
                    "openslide.level[1].downsample": "4.003",
                    "openslide.level[2].downsample": "16",
                },
                "Native magnification not available. Available slide properties: "
                "['openslide.level[1].downsample', 'openslide.level[2].downsample']",
            ),
        ),
    )
    def but_it_raises_an_exception_if_metadata_are_unavailable(
        self, request, properties, error, tmpdir
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        property_mock(request, Slide, "properties", return_value=properties)
        property_mock(request, Slide, "levels", return_value=[0, 1, 2])
        with pytest.raises(SlidePropertyError) as err:
            slide.level_magnification_factor(1)

        assert isinstance(err.value, SlidePropertyError)
        assert str(err.value) == error

    @pytest.mark.parametrize(
        "properties",
        (
            (
                {
                    "openslide.objective-power": "20",
                    "openslide.level[1].downsample": "4.003",
                    "openslide.level[2].downsample": "16",
                },
            ),
        ),
    )
    def and_it_raises_an_exception_if_level_in_incorrect(
        self, request, properties, tmpdir
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)
        property_mock(request, Slide, "properties", return_value=properties)
        property_mock(request, Slide, "levels", return_value=[0, 1, 2])
        with pytest.raises(LevelError) as err:
            slide.level_magnification_factor(4)

        assert isinstance(err.value, LevelError)
        assert str(err.value) == "Level 4 not available. Number of available levels: 3"

    @pytest.mark.parametrize("level", (1, -3))
    def it_raises_an_exception_when_magnification_factor_is_unavailable(
        self, level, tmpdir
    ):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)

        with pytest.raises(LevelError) as err:
            slide.level_magnification_factor(level=level)

        assert isinstance(err.value, LevelError)
        assert (
            str(err.value)
            == f"Level {level} not available. Number of available levels: 1"
        )

    def it_raises_an_exception_when_native_magnification_in_unavailable(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)

        with pytest.raises(SlidePropertyError) as err:
            slide.level_magnification_factor()

        assert isinstance(err.value, SlidePropertyError)
        assert (
            str(err.value)
            == "Native magnification not available. Available slide properties: []"
        )

    @pytest.mark.parametrize(
        "coords, expected_result",
        (
            (CP(0, 40, 0, 40), True),  # point
            (CP(0, 0, 48, 50), True),  # valid box
            (CP(800000, 90000, 8000010, 90010), False),  # out of bounds box
            (CP(800000, 90000, -1, 90010), False),  # negative coordinates
        ),
    )
    def it_knows_if_coords_are_valid(self, coords, expected_result, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_49X51_155_0_0)

        _are_valid = slide._has_valid_coords(coords)

        assert type(_are_valid) == bool
        assert _are_valid == expected_result

    def it_knows_its_levels(self, tmpdir):
        slide, _ = base_test_slide(tmpdir, PILIMG.RGBA_COLOR_500X500_155_249_240)

        levels = slide.levels

        assert type(levels) == list
        assert levels == [0]

    def it_can_access_to_its_properties(self, request):
        slide = Slide("path", "processed")
        properties = property_mock(request, Slide, "properties")
        properties.return_value = {"foo": "bar"}

        assert slide.properties == {"foo": "bar"}

    @pytest.mark.parametrize("level, expected_value", ((-1, 8), (-2, 7), (-9, 0)))
    def it_can_remap_negative_level_indices(self, level, expected_value, levels_prop):
        levels_prop.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        slide = Slide("path", "processed")

        assert slide._remap_level(level) == expected_value

    def but_it_raises_a_level_error_when_it_cannot_be_mapped(self, tmpdir, levels_prop):
        levels_prop.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        slide, _ = base_test_slide(tmpdir, PILIMG.RGB_RANDOM_COLOR_500X500)

        with pytest.raises(LevelError) as err:
            slide._remap_level(-10)

        assert isinstance(err.value, LevelError)
        assert (
            str(err.value) == "Level -10 not available. Number of available levels: 1"
        )

    # fixture components ---------------------------------------------

    @pytest.fixture
    def levels_prop(self, request):
        return property_mock(request, Slide, "levels")

    @pytest.fixture
    def resampled_dims_(self, request):
        return method_mock(request, Slide, "_resampled_dimensions")

    @pytest.fixture
    def dimensions_(self, request):
        return property_mock(request, Slide, "dimensions")


class Describe_Slideset:
    def it_constructs_from_args(self, request):
        _init_ = initializer_mock(request, SlideSet)
        _slides_path = "/foo/bar/"
        _processed_path = "/foo/bar/wsislides/processed"
        _valid_extensions = [".svs", ".tiff"]
        _keep_slides = ["mywsi.svs"]
        _slide_kwargs = {"use_largeimage": True}

        slideset = SlideSet(
            _slides_path,
            _processed_path,
            _valid_extensions,
            _keep_slides,
            _slide_kwargs,
        )

        _init_.assert_called_once_with(
            ANY,
            _slides_path,
            _processed_path,
            _valid_extensions,
            _keep_slides,
            _slide_kwargs,
        )
        assert isinstance(slideset, SlideSet)

    def it_can_construct_slides(self, request, tmpdir, Slide_):
        tmp_path_ = tmpdir.mkdir("myslide")
        slides_ = method_mock(request, SlideSet, "__iter__")
        slides_.return_value = [Slide_ for _ in range(10)]
        slideset = SlideSet(tmp_path_, os.path.join(tmp_path_, "b"), [".svs"])

        slides = slideset.__iter__()

        slides_.assert_called_once_with(slideset)
        assert len(slides) == 10

    def it_knows_its_slides(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi1.svs"), "TIFF")
        image.save(os.path.join(tmp_path_, "mywsi2.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, "proc", [".svs"])

        assert len(slideset) == 2

        # it can keep a subset of slides
        slideset = SlideSet(tmp_path_, "proc", [".svs"], keep_slides=["mywsi1.svs"])

        assert len(slideset) == 1

        slideset = SlideSet(None, "proc", [".svs"])

        assert len(slideset) == 0

        with pytest.raises(FileNotFoundError) as err:
            slideset = SlideSet("fake/path", "proc", [".svs"])
            list(slideset)

        assert isinstance(err.value, FileNotFoundError)
        assert err.value.errno == errno.ENOENT

    @pytest.mark.parametrize("slide_kwargs", (({"use_largeimage": True}), ({})))
    def it_creates_its_slides_with_the_correct_parameters(
        self, tmpdir, request, slide_kwargs
    ):
        slide_init_ = initializer_mock(request, Slide)
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi1.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, "proc", [".svs"], slide_kwargs=slide_kwargs)

        slideset.__iter__()

        slide_init_.assert_called_once_with(
            ANY, os.path.join(tmp_path_, "mywsi1.svs"), "proc", **slide_kwargs
        )

    def it_can_access_directly_to_the_slides(self, request, Slide_):
        slideset = instance_mock(request, SlideSet)
        slideset.__iter__.side_effect = iter([Slide_])

        slideset[0]

        slideset.__getitem__.assert_called_once_with(0)

    def and_it_is_exaclty_what_expected(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, "proc", [".svs"])

        slide = slideset[0]

        np.testing.assert_array_almost_equal(
            slide.resampled_array(), slideset[0].resampled_array()
        )

    def it_constructs_its_sequence_of_slides_to_help(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        image2 = PILIMG.RGBA_COLOR_50X50_155_0_0
        image2.save(os.path.join(tmp_path_, "mywsi2.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, "proc", [".svs"])
        expected_slides = [
            Slide(os.path.join(tmp_path_, _path), "proc")
            for _path in os.listdir(tmp_path_)
        ]

        slides = slideset.__iter__()

        for i, slide in enumerate(slides):
            np.testing.assert_array_almost_equal(
                slide.resampled_array(), expected_slides[i].resampled_array()
            )

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
        slides = method_mock(request, SlideSet, "__iter__")
        slides.return_value = [Slide_ for _ in range(4)]
        slideset = SlideSet("the/path", "proc", [".svs"])

        total_slides = len(slideset)

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

    def it_knows_its_scaled_slides(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        slide1 = instance_mock(request, Slide)
        slide2 = instance_mock(request, Slide)

        slideset = SlideSet(tmp_path_, os.path.join(tmp_path_, "processed"), [])
        slides = method_mock(request, SlideSet, "__iter__")
        slides.return_value = [slide1, slide2]
        slideset.scaled_images(32, 2)

        slide1.scaled_image.assert_called_once_with(32)
        slide2.scaled_image.assert_called_once_with(32)

    def it_knows_its_thumbnails(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        thumbnail_ = property_mock(request, Slide, "thumbnail")
        slide1 = Slide("foo/bar", "proc")
        slide2 = Slide("foo/bar", "proc")

        slideset = SlideSet(tmp_path_, os.path.join(tmp_path_, "processed"), [])
        slides = method_mock(request, SlideSet, "__iter__")
        slides.return_value = [slide1, slide2]

        slideset.thumbnails()

        assert thumbnail_.call_args_list == [call(), call()]

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
