# encoding: utf-8

import os

import numpy as np
import openslide
import PIL
import pytest

from histolab.slide import Slide

from .unitutil import ANY, PILImageMock, initializer_mock, method_mock, property_mock


class DescribeSlide(object):
    def it_constructs_from_args(self, request):
        _init_ = initializer_mock(request, Slide)
        _wsi_path = "/foo/bar/myslide.svs"
        _processed_path = "/foo/bar/myslide/processed"

        slide = Slide(_wsi_path, _processed_path)

        _init_.assert_called_once_with(
            ANY, _wsi_path, _processed_path,
        )
        assert isinstance(slide, Slide)

    def but_it_has_wrong_wsi_path_type(self):
        """This test simulates a wrong user behaviour, using a None object instead of a
        str, or a path as wsi_path param"""
        with pytest.raises(TypeError) as err:
            slide = Slide(None, "prc")
            wsiname = slide.wsi_name

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
            slide = Slide("wsipath", None)
            im_path = slide.scaled_image_path(32)

        assert isinstance(err.value, TypeError)
        assert (
            str(err.value) == "expected str, bytes or os.PathLike object, not NoneType"
        )

    def it_generates_the_correct_breadcumb(self, request, breadcumb_fixture):
        (
            resampled_dims,
            dir_path,
            wsi_path,
            proc_path,
            scale_factor,
            expected_path,
        ) = breadcumb_fixture
        _resampled_dimensions = method_mock(request, Slide, "_resampled_dimensions")
        _resampled_dimensions.return_value = resampled_dims
        slide = Slide(wsi_path, proc_path)

        _breadcumb = slide._breadcumb(dir_path, scale_factor)

        assert _breadcumb == expected_path

    def it_knows_its_wsi_name(self, wsi_name_fixture):
        _wsi_path, expected_value = wsi_name_fixture
        slide = Slide(_wsi_path, "processed/")

        wsi_name = slide.wsi_name

        assert wsi_name == expected_value

    def it_knows_its_scaled_image_path(self, scaled_img_path_fixture, resampled_dims_):
        wsi_path, proc_path, wsi_dims, expected_value = scaled_img_path_fixture
        resampled_dims_.return_value = wsi_dims
        slide = Slide(wsi_path, proc_path)

        scaled_img_path = slide.scaled_image_path(scale_factor=22)

        assert scaled_img_path == expected_value

    def it_knows_its_thumbnails_path(self, resampled_dims_):
        wsi_path, proc_path, wsi_dims, expected_value = (
            "/foo/bar/myslide.svs",
            "/foo/bar/myslide/processed",
            (345, 111, 333, 444),
            "/foo/bar/myslide/processed/thumbnails/myslide.png",
        )
        resampled_dims_.return_value = wsi_dims
        slide = Slide(wsi_path, proc_path)

        thumbnail_path = slide.thumbnail_path

        assert thumbnail_path == expected_value

    def it_knows_its_wsi_extension(self, wsi_ext_fixture):
        wsi_path, expected_value = wsi_ext_fixture
        slide = Slide(wsi_path, "processed")

        _wsi_ext = slide._wsi_extension

        assert _wsi_ext == expected_value

    def it_knows_its_wsi_dimensions(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        wsi_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(wsi_path, "processed")

        wsi_dims = slide.wsi_dimensions

        assert wsi_dims == (500, 500)

    def it_knows_its_resampled_dimensions(self, wsi_dimensions_):
        """This test prove that given the wsi_dimensions (mock object here), it does
        the correct maths operations:
            `large_w, large_h = self.wsi_dimensions`
            `new_w = math.floor(large_w / self._scale_factor)`
            `new_h = math.floor(large_h / self._scale_factor)`
            `return large_w, large_h, new_w, new_h`
        """
        wsi_dimensions_.return_value = (300, 500)
        slide = Slide("/a/b/foo", "processed")

        _resampled_dims = slide._resampled_dimensions(scale_factor=32)

        assert _resampled_dims == (300, 500, 9, 15)

    def but_it_raises_zero_division_error_when_scalefactor_is_0(self, wsi_dimensions_):
        """Considering the teset above, this one prove that a wrong behaviour of the
        user can cause a zerodivision error. In this case the scale_factor=0 generates
        the ZeroDivision Exception
        """
        wsi_dimensions_.return_value = (300, 500)
        slide = Slide("/a/b/foo", "processed")
        with pytest.raises(ZeroDivisionError) as err:
            _resampled_dims = slide._resampled_dimensions(scale_factor=0)

        assert isinstance(err.value, ZeroDivisionError)
        assert str(err.value) == "division by zero"

    def it_knows_its_resampled_array(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        wsi_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(wsi_path, "processed")
        resampled_dims_.return_value = (100, 200, 300, 400)

        resampled_array = slide.resampled_array(scale_factor=32)

        assert type(resampled_array) == np.ndarray
        assert resampled_array.shape == (400, 300, 3)

    def it_creates_a_correct_slide_object(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_50X50_RGBA_COLOR_155_0_0
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        wsi_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(wsi_path, "processed")

        _wsi = slide._wsi

        assert type(_wsi) in (openslide.OpenSlide, openslide.ImageSlide)

    def but_it_raises_an_exception_if_file_not_found(self):
        with pytest.raises(FileNotFoundError) as err:
            slide = Slide("wrong/path/fake.wsi", "processed")
            wsi = slide._wsi

        assert isinstance(err.value, FileNotFoundError)
        assert str(err.value) == "The wsi path resource doesn't exist"

    def or_it_raises_an_PIL_exception(self, tmpdir):
        wsi_path_ = tmpdir.mkdir("sub").join("hello.txt")
        wsi_path_.write("content")
        with pytest.raises(PIL.UnidentifiedImageError) as err:
            slide = Slide(os.path.join(wsi_path_), "processed")
            wsi = slide._wsi

        assert isinstance(err.value, PIL.UnidentifiedImageError)
        assert (
            str(err.value) == f"cannot identify image file '{os.path.join(wsi_path_)}'"
        )

    def it_can_resample_its_wsi(self, tmpdir, resampled_dims_):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_500X500_RGBA_COLOR_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        wsi_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(wsi_path, "processed")
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
        wsi_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(wsi_path, os.path.join(tmp_path_, "processed"))
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
        wsi_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(wsi_path, os.path.join(tmp_path_, "processed"))
        resampled_dims_.return_value = (100, 200, 300, 400)

        slide.save_thumbnail()

        assert slide.thumbnail_path == os.path.join(
            tmp_path_, "processed", "thumbnails", "mywsi.png"
        )
        assert os.path.exists(os.path.join(tmp_path_, slide.thumbnail_path))

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[("/a/b/mywsi.svs", ".svs"), ("/a/b/mywsi.34s", ".34s")])
    def wsi_ext_fixture(self, request):
        wsi_path, expected_value = request.param
        return wsi_path, expected_value

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
        wsi_path, proc_path, wsi_dims, expected_value = request.param
        return wsi_path, proc_path, wsi_dims, expected_value

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
            wsi_path,
            proc_path,
            scale_factor,
            expected_path,
        ) = request.param
        return (
            resampled_dims,
            dir_path,
            wsi_path,
            proc_path,
            scale_factor,
            expected_path,
        )

    @pytest.fixture(
        params=[("/foo/bar/myslide.svs", "myslide"), ("/foo/myslide.svs", "myslide")]
    )
    def wsi_name_fixture(self, request):
        wsi_path, expceted_value = request.param
        return wsi_path, expceted_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def resampled_dims_(self, request):
        return method_mock(request, Slide, "_resampled_dimensions")

    @pytest.fixture
    def wsi_dimensions_(self, request):
        return property_mock(request, Slide, "wsi_dimensions")
