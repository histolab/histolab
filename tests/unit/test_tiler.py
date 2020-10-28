import csv
import os
from unittest.mock import call

import pytest

import numpy as np
from histolab.exceptions import LevelError
from histolab.scorer import RandomScorer
from histolab.slide import Slide
from histolab.tile import Tile
from histolab.tiler import GridTiler, RandomTiler, ScoreTiler, Tiler
from histolab.types import CP

from ..unitutil import (
    ANY,
    PILIMG,
    NpArrayMock,
    function_mock,
    initializer_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class Describe_RandomTiler:
    def it_constructs_from_args(self, request):
        _init = initializer_mock(request, RandomTiler)

        random_tiler = RandomTiler((512, 512), 10, 2, 7, True, "", ".png", int(1e4))

        _init.assert_called_once_with(
            ANY, (512, 512), 10, 2, 7, True, "", ".png", int(1e4)
        )
        assert isinstance(random_tiler, RandomTiler)
        assert isinstance(random_tiler, Tiler)

    def but_it_has_wrong_tile_size_value(self):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, -1), 10, 0)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Tile size must be greater than 0 ((512, -1))"

    def or_it_has_not_available_level_value(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGB_RANDOM_COLOR_500X500
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        random_tiler = RandomTiler((128, 128), 10, 3)

        with pytest.raises(LevelError) as err:
            random_tiler.extract(slide)

        assert isinstance(err.value, LevelError)
        assert str(err.value) == "Level 3 not available. Number of available levels: 1"

    def or_it_has_negative_level_value(self):
        with pytest.raises(LevelError) as err:
            RandomTiler((512, 512), 10, -1)

        assert isinstance(err.value, LevelError)
        assert str(err.value) == "Level cannot be negative (-1)"

    def or_it_has_wrong_max_iter(self):
        with pytest.raises(ValueError) as err:
            RandomTiler((512, 512), 10, 0, max_iter=3)

        assert isinstance(err.value, ValueError)
        assert (
            str(err.value)
            == "The maximum number of iterations (3) must be grater than or equal to "
            "the maximum number of tiles (10)."
        )

    def or_it_has_wrong_seed(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGB_RANDOM_COLOR_500X500
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        random_tiler = RandomTiler((128, 128), 10, 0, seed=-1)

        with pytest.raises(ValueError) as err:
            random_tiler.extract(slide)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Seed must be between 0 and 2**32 - 1"

    @pytest.mark.parametrize("tile_size", ((512, 512), (128, 128), (10, 10)))
    def it_knows_its_tile_size(self, tile_size):
        random_tiler = RandomTiler(tile_size, 10, 0)

        tile_size_ = random_tiler.tile_size

        assert type(tile_size_) == tuple
        assert tile_size_ == tile_size

    @pytest.mark.parametrize("max_iter", (1000, 10, 3000))
    def it_knows_its_max_iter(self, max_iter):
        random_tiler = RandomTiler((128, 128), 10, 0, max_iter=max_iter)

        max_iter_ = random_tiler.max_iter

        assert type(max_iter_) == int
        assert max_iter_ == max_iter

    @pytest.mark.parametrize(
        "level, prefix, suffix, tile_coords, tiles_counter, expected_filename",
        (
            (3, "", ".png", CP(0, 512, 0, 512), 3, "tile_3_level3_0-512-0-512.png"),
            (
                0,
                "folder/",
                ".png",
                CP(4, 127, 4, 127),
                10,
                "folder/tile_10_level0_4-127-4-127.png",
            ),
        ),
    )
    def it_knows_its_tile_filename(
        self, level, prefix, suffix, tile_coords, tiles_counter, expected_filename
    ):
        random_tiler = RandomTiler((512, 512), 10, level, 7, True, prefix, suffix)

        _filename = random_tiler._tile_filename(tile_coords, tiles_counter)

        assert type(_filename) == str
        assert _filename == expected_filename

    def it_can_generate_random_coordinates(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        _box_mask_thumb = method_mock(request, RandomTiler, "box_mask")
        _box_mask_thumb.return_value = NpArrayMock.ONES_500X500_BOOL
        _tile_size = property_mock(request, RandomTiler, "tile_size")
        _tile_size.return_value = (128, 128)
        _np_random_choice1 = function_mock(request, "numpy.random.choice")
        _np_random_choice1.return_value = 0
        _np_random_choice2 = function_mock(request, "numpy.random.choice")
        _np_random_choice2.return_value = 0
        _scale_coordinates = function_mock(request, "histolab.tiler.scale_coordinates")
        random_tiler = RandomTiler((128, 128), 10, 0)

        random_tiler._random_tile_coordinates(slide)

        _box_mask_thumb.assert_called_once_with(slide)
        _tile_size.assert_has_calls([call((128, 128))])
        _scale_coordinates.assert_called_once_with(
            reference_coords=CP(x_ul=0, y_ul=0, x_br=128, y_br=128),
            reference_size=(500, 500),
            target_size=(500, 500),
        )

    @pytest.mark.parametrize(
        "check_tissue, expected_box",
        (
            (False, NpArrayMock.RANDOM_500X500_BOOL),
            (True, NpArrayMock.RANDOM_500X500_BOOL),
        ),
    )
    def it_knows_its_box_mask(self, request, tmpdir, check_tissue, expected_box):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        _biggest_tissue_box_mask = property_mock(
            request, Slide, "biggest_tissue_box_mask"
        )
        _biggest_tissue_box_mask.return_value = expected_box
        random_tiler = RandomTiler((128, 128), 10, 0, check_tissue=check_tissue)

        box_mask = random_tiler.box_mask(slide)

        _biggest_tissue_box_mask.assert_called_once_with()
        assert type(box_mask) == np.ndarray
        np.testing.assert_array_almost_equal(box_mask, expected_box)

    @pytest.mark.parametrize(
        "tile1, tile2, check_tissue, has_enough_tissue, max_iter, expected_value",
        (
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                True,
                [True, True],
                10,
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                True,
                [True, False],
                2,
                1,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(5900, 6000, 5900, 6000)),
                True,
                [True, True],
                2,
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                False,
                [True, True],
                10,
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                False,
                [False, False],
                10,
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                True,
                [False, False],
                10,
                0,
            ),
        ),
    )
    def it_can_generate_random_tiles(
        self,
        request,
        tmpdir,
        tile1,
        tile2,
        check_tissue,
        has_enough_tissue,
        max_iter,
        expected_value,
    ):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        _extract_tile = method_mock(request, Slide, "extract_tile")
        _has_enough_tissue = method_mock(request, Tile, "has_enough_tissue")
        _has_enough_tissue.side_effect = has_enough_tissue * (max_iter // 2)
        _random_tile_coordinates = method_mock(
            request, RandomTiler, "_random_tile_coordinates"
        )
        tiles = [tile1, tile2]
        _extract_tile.side_effect = tiles * (max_iter // 2)
        random_tiler = RandomTiler(
            (10, 10), 2, level=0, max_iter=max_iter, check_tissue=check_tissue
        )

        generated_tiles = list(random_tiler._random_tiles_generator(slide))

        _random_tile_coordinates.assert_called_with(random_tiler, slide)
        assert _random_tile_coordinates.call_count <= random_tiler.max_iter
        assert len(generated_tiles) == expected_value
        for i, tile in enumerate(generated_tiles):
            assert tile[0] == tiles[i]

    def it_can_extract_random_tiles(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _random_tiles_generator = method_mock(
            request, RandomTiler, "_random_tiles_generator"
        )
        coords = CP(0, 10, 0, 10)
        tile = Tile(image, coords)
        _random_tiles_generator.return_value = [(tile, coords), (tile, coords)]
        _tile_filename = method_mock(request, RandomTiler, "_tile_filename")
        _tile_filename.side_effect = [
            f"tile_{i}_level2_0-10-0-10.png" for i in range(2)
        ]
        random_tiler = RandomTiler((10, 10), n_tiles=2, level=2)

        random_tiler.extract(slide)

        assert _tile_filename.call_args_list == [
            call(random_tiler, coords, 0),
            call(random_tiler, coords, 1),
        ]
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tiles", "tile_0_level2_0-10-0-10.png")
        )
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tiles", "tile_1_level2_0-10-0-10.png")
        )


class Describe_GridTiler:
    def it_constructs_from_args(self, request):
        _init = initializer_mock(request, GridTiler)

        grid_tiler = GridTiler((512, 512), 2, True, 0, "", ".png")

        _init.assert_called_once_with(ANY, (512, 512), 2, True, 0, "", ".png")
        assert isinstance(grid_tiler, GridTiler)
        assert isinstance(grid_tiler, Tiler)

    def but_it_has_wrong_tile_size_value(self):
        with pytest.raises(ValueError) as err:
            GridTiler((512, -1))

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Tile size must be greater than 0 ((512, -1))"

    def or_it_has_not_available_level_value(self, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGB_RANDOM_COLOR_500X500
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        grid_tiler = GridTiler((128, 128), 3)

        with pytest.raises(LevelError) as err:
            grid_tiler.extract(slide)

        assert isinstance(err.value, LevelError)
        assert str(err.value) == "Level 3 not available. Number of available levels: 1"

    def or_it_has_negative_level_value(self):
        with pytest.raises(LevelError) as err:
            GridTiler((512, 512), -1)

        assert isinstance(err.value, LevelError)
        assert str(err.value) == "Level cannot be negative (-1)"

    @pytest.mark.parametrize("tile_size", ((512, 512), (128, 128), (10, 10)))
    def it_knows_its_tile_size(self, tile_size):
        grid_tiler = GridTiler(tile_size, 10, True, 0)

        tile_size_ = grid_tiler.tile_size

        assert type(tile_size_) == tuple
        assert tile_size_ == tile_size

    @pytest.mark.parametrize(
        "level, pixel_overlap, prefix, tile_coords, tiles_counter, expected_filename",
        (
            (3, 0, "", CP(0, 512, 0, 512), 3, "tile_3_level3_0-512-0-512.png"),
            (
                0,
                0,
                "folder/",
                CP(4, 127, 4, 127),
                10,
                "folder/tile_10_level0_4-127-4-127.png",
            ),
        ),
    )
    def it_knows_its_tile_filename(
        self,
        level,
        pixel_overlap,
        prefix,
        tile_coords,
        tiles_counter,
        expected_filename,
    ):
        grid_tiler = GridTiler((512, 512), level, True, pixel_overlap, prefix, ".png")

        _filename = grid_tiler._tile_filename(tile_coords, tiles_counter)

        assert type(_filename) == str
        assert _filename == expected_filename

    @pytest.mark.parametrize(
        "check_tissue, expected_box",
        (
            (False, NpArrayMock.ONES_500X500_BOOL),
            (True, NpArrayMock.RANDOM_500X500_BOOL),
        ),
    )
    def it_knows_its_box_mask(self, request, tmpdir, check_tissue, expected_box):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        _biggest_tissue_box_mask = property_mock(
            request, Slide, "biggest_tissue_box_mask"
        )
        _biggest_tissue_box_mask.return_value = expected_box
        grid_tiler = GridTiler((128, 128), 0, check_tissue=check_tissue)

        box_mask = grid_tiler.box_mask(slide)

        _biggest_tissue_box_mask.assert_called_once_with()
        assert type(box_mask) == np.ndarray
        np.testing.assert_array_almost_equal(box_mask, expected_box)

    @pytest.mark.parametrize(
        "bbox_coordinates, pixel_overlap, expected_n_tiles_row",
        (
            (CP(x_ul=0, y_ul=0, x_br=6060, y_br=1917), 0, 11),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), 0, 3),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), 128, 5),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), -128, 3),
        ),
    )
    def it_can_calculate_n_tiles_row(
        self, bbox_coordinates, pixel_overlap, expected_n_tiles_row
    ):
        grid_tiler = GridTiler((512, 512), 2, True, pixel_overlap)

        n_tiles_row = grid_tiler._n_tiles_row(bbox_coordinates)

        assert type(n_tiles_row) == int
        assert n_tiles_row == expected_n_tiles_row

    @pytest.mark.parametrize(
        "bbox_coordinates, pixel_overlap, expected_n_tiles_column",
        (
            (CP(x_ul=0, y_ul=0, x_br=6060, y_br=1917), 0, 3),
            (CP(x_ul=0, y_ul=0, x_br=6060, y_br=1917), -1, 3),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), 0, 4),
            (CP(x_ul=0, y_ul=0, x_br=1921, y_br=2187), 128, 5),
        ),
    )
    def it_can_calculate_n_tiles_column(
        self, bbox_coordinates, pixel_overlap, expected_n_tiles_column
    ):
        grid_tiler = GridTiler((512, 512), 2, True, pixel_overlap)

        n_tiles_column = grid_tiler._n_tiles_column(bbox_coordinates)

        assert type(n_tiles_column) == int
        assert n_tiles_column == expected_n_tiles_column

    @pytest.mark.parametrize(
        "tile1, tile2, check_tissue, has_enough_tissue, expected_n_tiles",
        (
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                True,
                [True, True],
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                False,
                [True, True],
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                False,
                [False, False],
                2,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                True,
                [False, False],
                0,
            ),
            (
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                Tile(PILIMG.RGBA_COLOR_500X500_155_249_240, CP(0, 10, 0, 10)),
                True,
                [True, False],
                1,
            ),
        ),
    )
    def it_can_generate_grid_tiles(
        self,
        request,
        tmpdir,
        tile1,
        tile2,
        check_tissue,
        has_enough_tissue,
        expected_n_tiles,
    ):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        _extract_tile = method_mock(request, Slide, "extract_tile")
        _has_enough_tissue = method_mock(request, Tile, "has_enough_tissue")
        _has_enough_tissue.side_effect = has_enough_tissue
        _grid_coordinates_generator = method_mock(
            request, GridTiler, "_grid_coordinates_generator"
        )
        _grid_coordinates_generator.return_value = [CP(0, 10, 0, 10), CP(0, 10, 0, 10)]
        _extract_tile.side_effect = [tile1, tile2]
        grid_tiler = GridTiler((10, 10), level=0, check_tissue=check_tissue)
        tiles = [tile1, tile2]

        generated_tiles = list(grid_tiler._grid_tiles_generator(slide))

        _grid_coordinates_generator.assert_called_once_with(grid_tiler, slide)
        assert _extract_tile.call_args_list == (
            [call(slide, CP(0, 10, 0, 10), 0), call(slide, CP(0, 10, 0, 10), 0)]
        )
        assert len(generated_tiles) == expected_n_tiles
        for i, tile in enumerate(generated_tiles):
            assert tile[0] == tiles[i]

    def but_with_wrong_coordinates(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        _has_enough_tissue = method_mock(request, Tile, "has_enough_tissue")
        _has_enough_tissue.return_value = False
        _grid_coordinates_generator = method_mock(
            request, GridTiler, "_grid_coordinates_generator"
        )
        coords1 = CP(600, 610, 600, 610)
        coords2 = CP(0, 10, 0, 10)
        _grid_coordinates_generator.return_value = [coords1, coords2]
        grid_tiler = GridTiler((10, 10), level=0, check_tissue=False)

        generated_tiles = list(grid_tiler._grid_tiles_generator(slide))

        _grid_coordinates_generator.assert_called_once_with(grid_tiler, slide)
        assert len(generated_tiles) == 1
        # generated_tiles[0][0] is a Tile object but we don't know what object it is
        # because Slide.extract_tile is not mocked (for the exception to happen inside)
        assert isinstance(generated_tiles[0][0], Tile)
        assert generated_tiles[0][1] == coords2

    def and_doesnt_raise_error_with_wrong_coordinates(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, "processed")
        coords = CP(5800, 6000, 5800, 6000)
        _grid_coordinates_generator = method_mock(
            request, GridTiler, "_grid_coordinates_generator"
        )
        _grid_coordinates_generator.return_value = [coords]
        grid_tiler = GridTiler((10, 10))
        generated_tiles = list(grid_tiler._grid_tiles_generator(slide))

        assert len(generated_tiles) == 0
        _grid_coordinates_generator.assert_called_once_with(grid_tiler, slide)

    def it_can_extract_grid_tiles(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _grid_tiles_generator = method_mock(request, GridTiler, "_grid_tiles_generator")
        coords = CP(0, 10, 0, 10)
        tile = Tile(image, coords)
        _grid_tiles_generator.return_value = [(tile, coords), (tile, coords)]
        _tile_filename = method_mock(request, GridTiler, "_tile_filename")
        _tile_filename.side_effect = [
            os.path.join(
                tmp_path_, "processed", "tiles", f"tile_{i}_level2_0-10-0-10.png"
            )
            for i in range(2)
        ]
        grid_tiler = GridTiler((10, 10), level=0)

        grid_tiler.extract(slide)

        assert _tile_filename.call_args_list == [
            call(grid_tiler, coords, 0),
            call(grid_tiler, coords, 1),
        ]
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tiles", "tile_0_level2_0-10-0-10.png")
        )
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tiles", "tile_1_level2_0-10-0-10.png")
        )


class Describe_ScoreTiler:
    def it_constructs_from_args(self, request):
        _init = initializer_mock(request, ScoreTiler)
        rs = RandomScorer()
        grid_tiler = ScoreTiler(rs, (512, 512), 4, 2, True, 0, "", ".png")

        _init.assert_called_once_with(ANY, rs, (512, 512), 4, 2, True, 0, "", ".png")

        assert isinstance(grid_tiler, ScoreTiler)
        assert isinstance(grid_tiler, GridTiler)
        assert isinstance(grid_tiler, Tiler)

    def it_knows_its_scorer(self):
        random_scorer = RandomScorer()
        score_tiler = ScoreTiler(random_scorer, (512, 512), 4, 0)

        scorer_ = score_tiler.scorer

        assert callable(scorer_)
        assert isinstance(scorer_, RandomScorer)

    def it_knows_its_n_tiles(self):
        n_tiles = 4
        score_tiler = ScoreTiler(RandomScorer(), (512, 512), n_tiles, 0)

        n_tiles_ = score_tiler.n_tiles

        assert type(n_tiles_) == int
        assert n_tiles_ == n_tiles

    def it_can_calculate_scores(self, request):
        slide = instance_mock(request, Slide)
        coords = CP(0, 10, 0, 10)
        image = PILIMG.RGB_RANDOM_COLOR_500X500
        tile = Tile(image, coords)
        _grid_tiles_generator = method_mock(
            request, ScoreTiler, "_grid_tiles_generator"
        )
        # it needs to be a generator
        _grid_tiles_generator.return_value = ((tile, coords) for i in range(3))
        _scorer = instance_mock(request, RandomScorer)
        _scorer.side_effect = [0.5, 0.7]
        score_tiler = ScoreTiler(_scorer, (10, 10), 2, 0)

        scores = score_tiler._scores(slide)

        assert _grid_tiles_generator.call_args_list == [
            call(score_tiler, slide),
            call(score_tiler, slide),
        ]
        assert _scorer.call_args_list == [call(tile), call(tile)]
        assert type(scores) == list
        assert type(scores[0]) == tuple
        assert type(scores[0][0]) == float
        assert type(scores[0][1]) == CP
        assert scores == [(0.5, coords), (0.7, coords)]

    def but_it_raises_runtimeerror_if_no_tiles_are_extracted(self, request):
        slide = instance_mock(request, Slide)
        _grid_tiles_generator = method_mock(
            request, ScoreTiler, "_grid_tiles_generator"
        )
        # it needs to be an empty generator
        _grid_tiles_generator.return_value = (n for n in [])
        score_tiler = ScoreTiler(None, (10, 10), 2, 0)

        with pytest.raises(RuntimeError) as err:
            score_tiler._scores(slide)

        _grid_tiles_generator.assert_called_once_with(score_tiler, slide)
        assert isinstance(err.value, RuntimeError)
        assert (
            str(err.value)
            == "No tiles have been generated. This could happen if `check_tissue=True`"
        )

    def it_can_scale_scores(self):
        coords = [CP(0, 10 * i, 0, 10) for i in range(3)]
        scores = [0.3, 0.4, 0.7]
        scores_ = list(zip(scores, coords))
        score_tiler = ScoreTiler(None, (10, 10), 2, 0)
        expected_scaled_coords = list(zip([0.0, 0.2500000000000001, 1.0], coords))

        scaled_scores = score_tiler._scale_scores(scores_)

        for (score, coords_), (expected_score, expected_coords) in zip(
            scaled_scores, expected_scaled_coords
        ):
            assert round(score, 5) == round(expected_score, 5)
            assert coords_ == expected_coords

    @pytest.mark.parametrize(
        "n_tiles, expected_value",
        (
            (
                0,
                (
                    [
                        (0.8, CP(0, 10, 0, 10)),
                        (0.7, CP(0, 10, 0, 10)),
                        (0.5, CP(0, 10, 0, 10)),
                        (0.2, CP(0, 10, 0, 10)),
                        (0.1, CP(0, 10, 0, 10)),
                    ],
                    [
                        (1.0, CP(0, 10, 0, 10)),
                        (0.857142857142857, CP(0, 10, 0, 10)),
                        (0.5714285714285714, CP(0, 10, 0, 10)),
                        (0.14285714285714285, CP(0, 10, 0, 10)),
                        (0.0, CP(0, 10, 0, 10)),
                    ],
                ),
            ),
            (
                2,
                (
                    [(0.8, CP(0, 10, 0, 10)), (0.7, CP(0, 10, 0, 10))],
                    [(1.0, CP(0, 10, 0, 10)), (0.857142857142857, CP(0, 10, 0, 10))],
                ),
            ),
            (
                3,
                (
                    [
                        (0.8, CP(0, 10, 0, 10)),
                        (0.7, CP(0, 10, 0, 10)),
                        (0.5, CP(0, 10, 0, 10)),
                    ],
                    [
                        (1.0, CP(0, 10, 0, 10)),
                        (0.857142857142857, CP(0, 10, 0, 10)),
                        (0.5714285714285714, CP(0, 10, 0, 10)),
                    ],
                ),
            ),
        ),
    )
    def it_can_calculate_highest_score_tiles(self, request, n_tiles, expected_value):
        slide = instance_mock(request, Slide)
        _scores = method_mock(request, ScoreTiler, "_scores")
        coords = CP(0, 10, 0, 10)
        _scores.return_value = [
            (0.7, coords),
            (0.5, coords),
            (0.2, coords),
            (0.8, coords),
            (0.1, coords),
        ]
        _scorer = instance_mock(request, RandomScorer)
        score_tiler = ScoreTiler(_scorer, (10, 10), n_tiles, 0)

        highest_score_tiles = score_tiler._highest_score_tiles(slide)

        _scores.assert_called_once_with(score_tiler, slide)
        assert highest_score_tiles == expected_value

    def but_it_raises_error_with_negative_n_tiles_value(self, request):
        slide = instance_mock(request, Slide)
        _scores = method_mock(request, ScoreTiler, "_scores")
        coords = CP(0, 10, 0, 10)
        _scores.return_value = [
            (0.7, coords),
            (0.5, coords),
            (0.2, coords),
            (0.8, coords),
            (0.1, coords),
        ]
        _scorer = instance_mock(request, RandomScorer)
        score_tiler = ScoreTiler(_scorer, (10, 10), -1, 0)

        with pytest.raises(ValueError) as err:
            score_tiler.extract(slide)

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "'n_tiles' cannot be negative (-1)"

    def it_can_extract_score_tiles(self, request, tmpdir):
        _extract_tile = method_mock(request, Slide, "extract_tile")
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _highest_score_tiles = method_mock(request, ScoreTiler, "_highest_score_tiles")
        coords = CP(0, 10, 0, 10)
        tile = Tile(image, coords)
        _extract_tile.return_value = tile
        _highest_score_tiles.return_value = (
            [(0.8, coords), (0.7, coords)],
            [(0.8, coords), (0.7, coords)],
        )
        _tile_filename = method_mock(request, GridTiler, "_tile_filename")
        _tile_filename.side_effect = [
            f"tile_{i}_level2_0-10-0-10.png" for i in range(2)
        ]
        _save_report = method_mock(request, ScoreTiler, "_save_report")
        random_scorer = RandomScorer()
        score_tiler = ScoreTiler(random_scorer, (10, 10), 2, 2)

        score_tiler.extract(slide)

        assert _extract_tile.call_args_list == [
            call(slide, coords, 2),
            call(slide, coords, 2),
        ]
        _highest_score_tiles.assert_called_once_with(score_tiler, slide)
        assert _tile_filename.call_args_list == [
            call(score_tiler, coords, 0),
            call(score_tiler, coords, 1),
        ]
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tiles", "tile_0_level2_0-10-0-10.png")
        )
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tiles", "tile_1_level2_0-10-0-10.png")
        )
        _save_report.assert_not_called()

    def it_can_save_report(self, request, tmpdir):
        tmp_path_ = tmpdir.mkdir("path")
        coords = CP(0, 10, 0, 10)
        highest_score_tiles = [(0.8, coords), (0.7, coords)]
        highest_scaled_score_tiles = [(0.1, coords), (0.0, coords)]
        filenames = ["tile0.png", "tile1.png"]
        random_scorer_ = instance_mock(request, RandomScorer)
        score_tiler = ScoreTiler(random_scorer_, (10, 10), 2, 2)
        report_ = [
            "filename,score,scaled_score",
            "tile0.png,0.8,0.1",
            "tile1.png,0.7,0.0",
        ]

        score_tiler._save_report(
            os.path.join(tmp_path_, "report.csv"),
            highest_score_tiles,
            highest_scaled_score_tiles,
            filenames,
        )

        assert os.path.exists(os.path.join(tmp_path_, "report.csv"))
        with open(os.path.join(tmp_path_, "report.csv"), newline="") as f:
            reader = csv.reader(f)
            report = [",".join(row) for row in reader]
            assert report == report_

    def it_can_extract_score_tiles_and_save_report(self, request, tmpdir):
        _extract_tile = method_mock(request, Slide, "extract_tile")
        tmp_path_ = tmpdir.mkdir("myslide")
        image = PILIMG.RGBA_COLOR_500X500_155_249_240
        image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
        slide_path = os.path.join(tmp_path_, "mywsi.png")
        slide = Slide(slide_path, os.path.join(tmp_path_, "processed"))
        _highest_score_tiles = method_mock(request, ScoreTiler, "_highest_score_tiles")
        coords = CP(0, 10, 0, 10)
        tile = Tile(image, coords)
        _extract_tile.return_value = tile
        _highest_score_tiles.return_value = (
            [(0.8, coords), (0.7, coords)],
            [(0.8, coords), (0.7, coords)],
        )
        _tile_filename = method_mock(request, GridTiler, "_tile_filename")
        _tile_filename.side_effect = [
            f"tile_{i}_level2_0-10-0-10.png" for i in range(2)
        ]
        _save_report = method_mock(request, ScoreTiler, "_save_report", autospec=False)
        random_scorer = RandomScorer()
        score_tiler = ScoreTiler(random_scorer, (10, 10), 2, 2)

        score_tiler.extract(slide, "report.csv")

        assert _extract_tile.call_args_list == [
            call(slide, coords, 2),
            call(slide, coords, 2),
        ]
        _highest_score_tiles.assert_called_once_with(score_tiler, slide)
        assert _tile_filename.call_args_list == [
            call(score_tiler, coords, 0),
            call(score_tiler, coords, 1),
        ]
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tiles", "tile_0_level2_0-10-0-10.png")
        )
        assert os.path.exists(
            os.path.join(tmp_path_, "processed", "tiles", "tile_1_level2_0-10-0-10.png")
        )
        _save_report.assert_called_once_with(
            "report.csv",
            [(0.8, coords), (0.7, coords)],
            [(0.8, coords), (0.7, coords)],
            [f"tile_{i}_level2_0-10-0-10.png" for i in range(2)],
        )
