# encoding: utf-8

import os

import numpy as np
import openslide
import PIL
import pytest
from matplotlib.figure import Figure as matplotlib_figure


from src.histolab.slide import Slide, SlideSet

from ..unitutil import (
    dict_list_eq,
    property_mock,
    initializer_mock,
    class_mock,
    instance_mock,
    method_mock,
    PILImageMock,
    ANY,
)


class Describe_Slide(object):
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
            name = slide.name

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
            im_path = slide.scaled_image_path(32)

        assert isinstance(err.value, TypeError)
        assert (
            str(err.value) == "expected str, bytes or os.PathLike object, not NoneType"
        )

    def it_generates_the_correct_breadcumb(self, request, breadcumb_fixture):
        (
            resampled_dims,
            dir_path,
            slide_path,
            proc_path,
            scale_factor,
            expected_path,
        ) = breadcumb_fixture
        _resampled_dimensions = method_mock(request, Slide, "_resampled_dimensions")
        _resampled_dimensions.return_value = resampled_dims
        slide = Slide(slide_path, proc_path)

        _breadcumb = slide._breadcumb(dir_path, scale_factor)

        assert _breadcumb == expected_path

    def it_knows_its_name(self, slide_name_fixture):
        _slide_path, expected_value = slide_name_fixture
        slide = Slide(_slide_path, "processed/")

        name = slide.name

        assert name == expected_value

    def it_knows_its_scaled_image_path(self, scaled_img_path_fixture, resampled_dims_):
        slide_path, proc_path, slide_dims, expected_value = scaled_img_path_fixture
        resampled_dims_.return_value = slide_dims
        slide = Slide(slide_path, proc_path)

        scaled_img_path = slide.scaled_image_path(scale_factor=22)

        assert scaled_img_path == expected_value

    def it_knows_its_thumbnails_path(self, resampled_dims_):
        slide_path, proc_path, slide_dims, expected_value = (
            "/foo/bar/myslide.svs",
            "/foo/bar/myslide/processed",
            (345, 111, 333, 444),
            "/foo/bar/myslide/processed/thumbnails/myslide.png",
        )
        resampled_dims_.return_value = slide_dims
        slide = Slide(slide_path, proc_path)

        thumbnail_path = slide.thumbnail_path

        assert thumbnail_path == expected_value

    def it_knows_its_wsi_extension(self, slide_ext_fixture):
        slide_path, expected_value = slide_ext_fixture
        slide = Slide(slide_path, "processed")

        _ext = slide._extension

        assert _ext == expected_value

    def it_knows_its_dimensions(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
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
            _resampled_dims = slide._resampled_dimensions(scale_factor=0)

        assert isinstance(err.value, ZeroDivisionError)
        assert str(err.value) == "division by zero"

    def it_knows_its_resampled_array(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        resampled_dims_.return_value = (100, 200, 300, 400)

        resampled_array = slide.resampled_array(scale_factor=32)

        assert type(resampled_array) == np.ndarray
        assert resampled_array.shape == (400, 300, 3)

    def it_creates_a_correct_slide_object(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")

        _wsi = slide._wsi

        assert type(_wsi) in (openslide.OpenSlide, openslide.ImageSlide)

    def but_it_raises_an_exception_if_file_not_found(self):
        with pytest.raises(FileNotFoundError) as err:
            slide = Slide("wrong/path/fake.wsi", "processed")
            wsi = slide._wsi

        assert isinstance(err.value, FileNotFoundError)
        assert str(err.value) == "The wsi path resource doesn't exist"

    def or_it_raises_an_PIL_exception(self, tmpdir):
        slide_path = tmpdir.mkdir("sub").join("hello.txt")
        slide_path.write("content")
        with pytest.raises(PIL.UnidentifiedImageError) as err:
            slide = Slide(os.path.join(slide_path), "processed")
            wsi = slide._wsi

        assert isinstance(err.value, PIL.UnidentifiedImageError)
        assert (
            str(err.value) == f"cannot identify image file '{os.path.join(slide_path)}'"
        )

    def it_can_resample_itself(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
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

    def it_can_save_scaled_image(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
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
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        resampled_dims_.return_value = (100, 200, 300, 400)

        slide.save_thumbnail()

        assert slide.thumbnail_path == os.path.join(
            tmp_path_, "processed", "thumbnails", "mywsi.png"
        )
        assert os.path.exists(os.path.join(tmp_path_, slide.thumbnail_path))

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[("/a/b/mywsi.svs", ".svs"), ("/a/b/mywsi.34s", ".34s")])
    def slide_ext_fixture(self, request):
        slide_path, expected_value = request.param
        return slide_path, expected_value

    @pytest.fixture(
        params=[
            (
                "/foo/bar/myslide.svs",
                "/foo/bar/myslide/processed",
                (345, 111, 333, 444),
                "/foo/bar/myslide/processed/myslide-22x-345x111-333x444.png",
            ),
            (
                "/foo/bar/myslide2.svs",
                "/foo/bar/myslide/processed",
                (345, 111, None, None),
                "/foo/bar/myslide/processed/myslide2-22x-345x111-NonexNone.png",
            ),
            (
                "/foo/bar/myslide2.svs",
                "/foo/bar/myslide/processed",
                (345, 111, 123, 123),
                "/foo/bar/myslide/processed/myslide2-22x-345x111-123x123.png",
            ),
            (
                "/foo/bar/myslide2.svs",
                "/foo/bar/myslide/processed",
                (None, None, None, None),
                "/foo/bar/myslide/processed/myslide2*.png",
            ),
        ]
    )
    def scaled_img_path_fixture(self, request):
        slide_path, proc_path, slide_dims, expected_value = request.param
        return slide_path, proc_path, slide_dims, expected_value

    @pytest.fixture(
        params=[
            (
                (245, 123, 145, 99),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                "/foo/bar/b/0/9/myslide-64x-245x123-145x99.png",
            ),
            (
                (245, 123, 145, 99),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                32,
                "/foo/bar/b/0/9/myslide-32x-245x123-145x99.png",
            ),
            (
                (None, None, None, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                "/foo/bar/b/0/9/myslide*.png",
            ),
            (
                (None, 234, 192, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                "/foo/bar/b/0/9/myslide-64x-Nonex234-192xNone.png",
            ),
            (
                (123, 234, 192, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                "/foo/bar/b/0/9/myslide-64x-123x234-192xNone.png",
            ),
            (
                (None, None, 192, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                "processed",
                64,
                "/foo/bar/b/0/9/myslide-64x-NonexNone-192xNone.png",
            ),
        ]
    )
    def breadcumb_fixture(self, request):
        (
            resampled_dims,
            dir_path,
            slide_path,
            proc_path,
            scale_factor,
            expected_path,
        ) = request.param
        return (
            resampled_dims,
            dir_path,
            slide_path,
            proc_path,
            scale_factor,
            expected_path,
        )

    @pytest.fixture(
        params=[("/foo/bar/myslide.svs", "myslide"), ("/foo/myslide.svs", "myslide")]
    )
    def slide_name_fixture(self, request):
        slide_path, expceted_value = request.param
        return slide_path, expceted_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def resampled_dims_(self, request):
        return method_mock(request, Slide, "_resampled_dimensions")

    @pytest.fixture
    def dimensions_(self, request):
        return property_mock(request, Slide, "dimensions")


class Describe_Slideset(object):
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
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
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
        assert str(err.value) == "[Errno 2] No such file or directory: 'fake/path'"

    def it_knows_the_slides_dimensions(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        image2 = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
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
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        image2 = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        image2.save(os.path.join(tmp_path_, "mywsi2.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, "proc", [".svs"])

        _slides_dimensions_list = slideset._slides_dimensions_list

        assert sorted(_slides_dimensions_list) == sorted([(500, 500), (50, 50)])

    def it_knows_its_total_slides(self, request):
        slides = property_mock(request, SlideSet, "slides")
        slide_mock = class_mock(request, "src.histolab.slide.Slide")
        slides.return_value = [slide_mock for _ in range(4)]
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

    def it_knows_its_dimensions_stats(self, total_slides_prop, tmpdir):
        total_slides_prop.return_value = 2
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        image2 = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        image2.save(os.path.join(tmp_path_, "mywsi2.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, "proc", [".svs"])

        dimensions_stats = slideset._dimensions_stats

        expected_value = {
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
        assert dimensions_stats == expected_value

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
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.svs"), "TIFF")
        image2 = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        image2.save(os.path.join(tmp_path_, "mywsi2.svs"), "TIFF")
        slideset = SlideSet(tmp_path_, os.path.join("proc"), [".svs"])

        slides_stats = slideset.slides_stats

        assert slides_stats[0] == slideset._dimensions_stats
        assert type(slides_stats[1]) == matplotlib_figure

    @pytest.mark.mpl_image_compare(baseline_dir="../fixtures/mpl-baseline-images")
    def test_generates_a_correct_plot_figure(
        self, request, total_slides_prop, _slides_dimensions_list_prop
    ):
        dimensions_stats = property_mock(request, SlideSet, "_dimensions_stats")
        dimensions_stats.return_vaulue = {"a": 1}
        total_slides_prop.return_value = 2
        _slides_dimensions_list_prop.return_value = ((100, 200), (200, 300))

        slideset = SlideSet(None, None, [".svs"])

        slides_stats_chart = slideset.slides_stats[1]

        return slides_stats_chart

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
