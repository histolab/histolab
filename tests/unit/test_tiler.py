import os

import pytest

from src.histolab.slide import Slide
from src.histolab.tiler import RandomTiler, Tiler

from ..unitutil import ANY, PILImageMock, initializer_mock


class Describe_RandomTiler(object):
    def it_constructs_from_args(self, request):
        _init = initializer_mock(request, RandomTiler)

        random_tiler = RandomTiler((512, 512), 10, 2, 7, True, "", ".png", 1e4)

        _init.assert_called_once_with(ANY, (512, 512), 10, 2, 7, True, "", ".png", 1e4)
        assert isinstance(random_tiler, RandomTiler)
        assert isinstance(random_tiler, Tiler)

    def but_it_has_wrong_tile_size_value(self, request):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, -1), 10, 0)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Tile size must be greater than 0 ((512, -1))"

    def or_it_has_not_available_level_value(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILImageMock.DIMS_50X50_RGB_RANDOM_COLOR
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        random_tiler = RandomTiler((512, 512), 10, 3)

        with pytest.raises(IndexError) as err:
            random_tiler.extract(slide)

        assert isinstance(err.value, IndexError)
        assert str(err.value) == "tuple index out of range"

    def or_it_has_negative_level_value(self, request):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, 512), 10, -1)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Level cannot be negative (-1)"

    def or_it_has_wrong_max_iter(self, request):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, 512), 10, 0, max_iter=3)

        assert isinstance(err.value, ValueError)
        assert (
            str(err.value)
            == "The maximum number of iterations (3) must be grater than or equal to the maximum number of tiles (10)."
        )
