# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2020 All Histolab Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import csv
import os
from abc import abstractmethod
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from .exceptions import LevelError
from .scorer import Scorer
from .slide import Slide
from .tile import Tile
from .types import CoordinatePair
from .util import region_coordinates, regions_from_binary_mask, scale_coordinates

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Tiler(Protocol):
    """General tiler object"""

    level: int
    tile_size: int

    @lru_cache(maxsize=100)
    def box_mask(self, slide: Slide) -> np.ndarray:
        """Return binary mask, at thumbnail level, of the box for tiles extraction.

        The mask pixels set to True correspond to the tissue box.

        Parameters
        ----------
        slide : Slide
            The Slide from which to extract the extraction mask

        Returns
        -------
        np.ndarray
            Extraction mask at thumbnail level
        """

        return slide.biggest_tissue_box_mask

    @abstractmethod
    def extract(self, slide: Slide):
        raise NotImplementedError

    # ------- implementation helpers -------

    def _tile_filename(
        self, tile_wsi_coords: CoordinatePair, tiles_counter: int
    ) -> str:
        """Return the tile filename according to its 0-level coordinates and a counter.

        Parameters
        ----------
        tile_wsi_coords : CoordinatePair
            0-level coordinates of the slide the tile has been extracted from.
        tiles_counter : int
            Counter of extracted tiles.

        Returns
        -------
        str
            Tile filename, according to the format
            `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}"
            "-{y_br_wsi}{suffix}`
        """

        x_ul_wsi, y_ul_wsi, x_br_wsi, y_br_wsi = tile_wsi_coords
        tile_filename = (
            f"{self.prefix}tile_{tiles_counter}_level{self.level}_{x_ul_wsi}-{y_ul_wsi}"
            f"-{x_br_wsi}-{y_br_wsi}{self.suffix}"
        )

        return tile_filename


class GridTiler(Tiler):
    """Extractor of tiles arranged in a grid, at the given level, with the given size.

    Arguments
    ---------
    tile_size : Tuple[int, int]
        (width, height) of the extracted tiles.
    level : int, optional
        Level from which extract the tiles. Default is 0.
    check_tissue : bool, optional
        Whether to check if the tile has enough tissue to be saved. Default is True.
    pixel_overlap : int, optional
       Number of overlapping pixels (for both height and width) between two adjacent
       tiles. If negative, two adjacent tiles will be strided by the absolute value of
       ``pixel_overlap``. Default is 0.
    prefix : str, optional
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str, optional
        Suffix to be added to the tile filename. Default is '.png'
    """

    def __init__(
        self,
        tile_size: Tuple[int, int],
        level: int = 0,
        check_tissue: bool = True,
        pixel_overlap: int = 0,
        prefix: str = "",
        suffix: str = ".png",
    ):
        self.tile_size = tile_size
        self.level = level
        self.check_tissue = check_tissue
        self.pixel_overlap = pixel_overlap
        self.prefix = prefix
        self.suffix = suffix

    def extract(self, slide: Slide):
        """Extract tiles arranged in a grid and save them to disk, following this
        filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        """
        if self.level not in slide.levels:
            raise LevelError(
                f"Level {self.level} not available. Number of available levels: "
                f"{len(slide.levels)}"
            )

        grid_tiles = self._grid_tiles_generator(slide)

        tiles_counter = 0

        for tiles_counter, (tile, tile_wsi_coords) in enumerate(grid_tiles):
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            full_tile_path = os.path.join(slide.processed_path, "tiles", tile_filename)
            tile.save(full_tile_path)
            print(f"\t Tile {tiles_counter} saved: {tile_filename}")

        print(f"{tiles_counter} Grid Tiles have been saved.")

    @property
    def level(self) -> int:
        return self._valid_level

    @level.setter
    def level(self, level_: int):
        if level_ < 0:
            raise LevelError(f"Level cannot be negative ({level_})")
        self._valid_level = level_

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self._valid_tile_size

    @tile_size.setter
    def tile_size(self, tile_size_: Tuple[int, int]):
        if tile_size_[0] < 1 or tile_size_[1] < 1:
            raise ValueError(f"Tile size must be greater than 0 ({tile_size_})")
        self._valid_tile_size = tile_size_

    # ------- implementation helpers -------

    def _grid_coordinates_from_bbox_coordinates(
        self, bbox_coordinates: CoordinatePair, slide: Slide
    ) -> CoordinatePair:
        """Generate Coordinates at level 0 of grid tiles within a tissue box.

        Parameters
        ----------
        bbox_coordinates: CoordinatePair
            Coordinates of the tissue box from which to calculate the coordinates.
        slide : Slide
            Slide from which to calculate the coordinates.

        Yields
        -------
        Iterator[CoordinatePair]
            Iterator of tiles' CoordinatePair
        """
        tile_w_lvl, tile_h_lvl = self.tile_size

        n_tiles_row = self._n_tiles_row(bbox_coordinates)
        n_tiles_column = self._n_tiles_column(bbox_coordinates)

        for i in range(n_tiles_row):
            for j in range(n_tiles_column):
                x_ul_lvl = bbox_coordinates.x_ul + tile_w_lvl * j - self.pixel_overlap
                y_ul_lvl = bbox_coordinates.y_ul + tile_h_lvl * i - self.pixel_overlap

                x_ul_lvl = np.clip(x_ul_lvl, bbox_coordinates.x_ul, None)
                y_ul_lvl = np.clip(y_ul_lvl, bbox_coordinates.y_ul, None)

                x_br_lvl = x_ul_lvl + tile_w_lvl
                y_br_lvl = y_ul_lvl + tile_h_lvl

                tile_wsi_coords = scale_coordinates(
                    reference_coords=CoordinatePair(
                        x_ul_lvl, y_ul_lvl, x_br_lvl, y_br_lvl
                    ),
                    reference_size=slide.level_dimensions(level=self.level),
                    target_size=slide.level_dimensions(level=0),
                )
                yield tile_wsi_coords

    def _grid_coordinates_generator(self, slide: Slide) -> CoordinatePair:
        """Generate Coordinates at level 0 of grid tiles within the tissue.

        Parameters
        ----------
        slide : Slide
            Slide from which to calculate the coordinates. Needed to calculate the
            tissue area.

        Yields
        -------
        Iterator[CoordinatePair]
            Iterator of tiles' CoordinatePair
        """
        box_mask = self.box_mask(slide)

        regions = regions_from_binary_mask(box_mask)
        # ----at the moment there is only one region----
        for region in regions:
            bbox_coordinates_thumb = region_coordinates(region)
            bbox_coordinates = scale_coordinates(
                bbox_coordinates_thumb,
                box_mask.shape[::-1],
                slide.level_dimensions(self.level),
            )
            yield from self._grid_coordinates_from_bbox_coordinates(
                bbox_coordinates, slide
            )

    def _grid_tiles_generator(self, slide: Slide) -> Tuple[Tile, CoordinatePair]:
        """Generator of tiles arranged in a grid.

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles

        Yields
        -------
        Tile
            Extracted tile
        CoordinatePair
            Coordinates of the slide at level 0 from which the tile has been extracted
        """

        grid_coordinates_generator = self._grid_coordinates_generator(slide)
        for coords in grid_coordinates_generator:
            try:
                tile = slide.extract_tile(coords, self.level)
            except ValueError:
                continue

            if not self.check_tissue or tile.has_enough_tissue():
                yield tile, coords

    def _n_tiles_column(self, bbox_coordinates: CoordinatePair) -> int:
        """Return the number of tiles which can be extracted in a column.

        Parameters
        ----------
        bbox_coordinates : CoordinatePair
            Coordinates of the tissue box

        Returns
        -------
        int
            Number of tiles which can be extracted in a column.
        """
        return (bbox_coordinates.y_br - bbox_coordinates.y_ul) // (
            self.tile_size[1] - self.pixel_overlap
        )

    def _n_tiles_row(self, bbox_coordinates: CoordinatePair) -> int:
        """Return the number of tiles which can be extracted in a row.

        Parameters
        ----------
        bbox_coordinates : CoordinatePair
            Coordinates of the tissue box

        Returns
        -------
        int
            Number of tiles which can be extracted in a row.
        """
        return (bbox_coordinates.x_br - bbox_coordinates.x_ul) // (
            self.tile_size[0] - self.pixel_overlap
        )


class RandomTiler(Tiler):
    """Extractor of random tiles from a Slide, at the given level, with the given size.

    Arguments
    ---------
    tile_size : Tuple[int, int]
        (width, height) of the extracted tiles.
    n_tiles : int
        Maximum number of tiles to extract.
    level : int, optional
        Level from which extract the tiles. Default is 0.
    seed : int, optional
        Seed for RandomState. Must be convertible to 32 bit unsigned integers. Default
        is 7.
    check_tissue : bool, optional
        Whether to check if the tile has enough tissue to be saved. Default is True.
    prefix : str, optional
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str, optional
        Suffix to be added to the tile filename. Default is '.png'
    max_iter : int, optional
        Maximum number of iterations performed when searching for eligible (if
        ``check_tissue=True``) tiles. Must be grater than or equal to ``n_tiles``.
    """

    def __init__(
        self,
        tile_size: Tuple[int, int],
        n_tiles: int,
        level: int = 0,
        seed: int = 7,
        check_tissue: bool = True,
        prefix: str = "",
        suffix: str = ".png",
        max_iter: int = int(1e4),
    ):

        super().__init__()

        self.tile_size = tile_size
        self.n_tiles = n_tiles
        self.max_iter = max_iter
        self.level = level
        self.seed = seed
        self.check_tissue = check_tissue
        self.prefix = prefix
        self.suffix = suffix

    def extract(self, slide: Slide):
        """Extract random tiles and save them to disk, following this filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        """

        np.random.seed(self.seed)

        random_tiles = self._random_tiles_generator(slide)

        tiles_counter = 0
        for tiles_counter, (tile, tile_wsi_coords) in enumerate(random_tiles):
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            tile.save(os.path.join(slide.processed_path, "tiles", tile_filename))
            print(f"\t Tile {tiles_counter} saved: {tile_filename}")
        print(f"{tiles_counter+1} Random Tiles have been saved.")

    @property
    def level(self) -> int:
        return self._valid_level

    @level.setter
    def level(self, level_: int):
        if level_ < 0:
            raise LevelError(f"Level cannot be negative ({level_})")
        self._valid_level = level_

    @property
    def max_iter(self) -> int:
        return self._valid_max_iter

    @max_iter.setter
    def max_iter(self, max_iter_: int = int(1e4)):
        if max_iter_ < self.n_tiles:
            raise ValueError(
                f"The maximum number of iterations ({max_iter_}) must be grater than or"
                f" equal to the maximum number of tiles ({self.n_tiles})."
            )
        self._valid_max_iter = max_iter_

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self._valid_tile_size

    @tile_size.setter
    def tile_size(self, tile_size_: Tuple[int, int]):
        if tile_size_[0] < 1 or tile_size_[1] < 1:
            raise ValueError(f"Tile size must be greater than 0 ({tile_size_})")
        self._valid_tile_size = tile_size_

    # ------- implementation helpers -------

    def _random_tile_coordinates(self, slide: Slide) -> CoordinatePair:
        """Return 0-level Coordinates of a tile picked at random within the box.

        Parameters
        ----------
        slide : Slide
            Slide from which calculate the coordinates. Needed to calculate the box.

        Returns
        -------
        CoordinatePair
            Random tile Coordinates at level 0
        """
        box_mask = self.box_mask(slide)
        tile_w_lvl, tile_h_lvl = self.tile_size

        x_ul_lvl = np.random.choice(np.where(box_mask)[1])
        y_ul_lvl = np.random.choice(np.where(box_mask)[0])

        # Scale tile dimensions to thumbnail dimensions
        tile_w_thumb = (
            tile_w_lvl * box_mask.shape[1] / slide.level_dimensions(self.level)[0]
        )
        tile_h_thumb = (
            tile_h_lvl * box_mask.shape[0] / slide.level_dimensions(self.level)[1]
        )

        x_br_lvl = x_ul_lvl + tile_w_thumb
        y_br_lvl = y_ul_lvl + tile_h_thumb

        tile_wsi_coords = scale_coordinates(
            reference_coords=CoordinatePair(x_ul_lvl, y_ul_lvl, x_br_lvl, y_br_lvl),
            reference_size=box_mask.shape[::-1],
            target_size=slide.dimensions,
        )

        return tile_wsi_coords

    def _random_tiles_generator(self, slide: Slide) -> Tuple[Tile, CoordinatePair]:
        """Generate Random Tiles within a slide box.

        Stops if:
        * the number of extracted tiles is equal to ``n_tiles`` OR
        * the maximum number of iterations ``max_iter`` is reached

        Parameters
        ----------
        slide : Slide
            The Whole Slide Image from which to extract the tiles.

        Yields
        ------
        tile : Tile
            The extracted Tile
        coords : CoordinatePair
            The level-0 coordinates of the extracted tile
        """

        iteration = valid_tile_counter = 0

        while True:

            tile_wsi_coords = self._random_tile_coordinates(slide)
            try:
                tile = slide.extract_tile(tile_wsi_coords, self.level)
            except ValueError:
                iteration -= 1
                continue

            if not self.check_tissue or tile.has_enough_tissue():
                yield tile, tile_wsi_coords
                valid_tile_counter += 1
            iteration += 1

            if self.max_iter and iteration >= self.max_iter:
                break

            if valid_tile_counter >= self.n_tiles:
                break


class ScoreTiler(GridTiler):
    """Extractor of tiles arranged in a grid according to a scoring function.

    The extraction procedure is the same as the ``GridTiler`` extractor, but only the
    first ``n_tiles`` tiles with the highest score are saved.

    Arguments
    ---------
    scorer : Scorer
        Scoring function used to score the tiles.
    tile_size : Tuple[int, int]
        (width, height) of the extracted tiles.
    n_tiles : int, optional
        The number of tiles to be saved. Default is 0, which means that all the tiles
        will be saved (same exact behaviour of a GridTiler). Cannot be negative.
    level : int, optional
        Level from which extract the tiles. Default is 0.
    check_tissue : bool, optional
        Whether to check if the tile has enough tissue to be saved. Default is True.
    pixel_overlap : int, optional
       Number of overlapping pixels (for both height and width) between two adjacent
       tiles. If negative, two adjacent tiles will be strided by the absolute value of
       ``pixel_overlap``. Default is 0.
    prefix : str, optional
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str, optional
        Suffix to be added to the tile filename. Default is '.png'
    """

    def __init__(
        self,
        scorer: Scorer,
        tile_size: Tuple[int, int],
        n_tiles: int = 0,
        level: int = 0,
        check_tissue: bool = True,
        pixel_overlap: int = 0,
        prefix: str = "",
        suffix: str = ".png",
    ):
        self.scorer = scorer
        self.n_tiles = n_tiles

        super().__init__(tile_size, level, check_tissue, pixel_overlap, prefix, suffix)

    def extract(self, slide: Slide, report_path: str = None):
        """Extract grid tiles and save them to disk, according to a scoring function and
        following this filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Save a CSV report file with the saved tiles and the associated score.

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        report_path : str, optional
            Path to the CSV report. If None, no report will be saved
        """
        highest_score_tiles, highest_scaled_score_tiles = self._highest_score_tiles(
            slide
        )

        tiles_counter = 0
        filenames = []

        for tiles_counter, (score, tile_wsi_coords) in enumerate(highest_score_tiles):
            tile = slide.extract_tile(tile_wsi_coords, self.level)
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            tile.save(os.path.join(slide.processed_path, "tiles", tile_filename))
            filenames.append(tile_filename)
            print(f"\t Tile {tiles_counter} - score: {score} saved: {tile_filename}")

        if report_path:
            self._save_report(
                report_path, highest_score_tiles, highest_scaled_score_tiles, filenames
            )

        print(f"{tiles_counter+1} Grid Tiles have been saved.")

    # ------- implementation helpers -------

    def _highest_score_tiles(self, slide: Slide) -> List[Tuple[float, CoordinatePair]]:
        """Calculate the tiles with the highest scores and their extraction coordinates.

        Parameters
        ----------
        slide : Slide
            The slide to extract the tiles from.

        Returns
        -------
        List[Tuple[float, CoordinatePair]]
            List of tuples containing the score and the extraction coordinates for the
            tiles with the highest score. Each tuple represents a tile.
        List[Tuple[float, CoordinatePair]]
            List of tuples containing the scaled score between 0 and 1 and the
            extraction coordinates for the tiles with the highest score. Each tuple
            represents a tile.

        Raises
        ------
        ValueError
            If ``n_tiles`` is negative.
        """
        all_scores = self._scores(slide)
        scaled_scores = self._scale_scores(all_scores)

        sorted_tiles_by_score = sorted(all_scores, key=lambda x: x[0], reverse=True)
        sorted_tiles_by_scaled_score = sorted(
            scaled_scores, key=lambda x: x[0], reverse=True
        )
        if self.n_tiles < 0:
            raise ValueError(f"'n_tiles' cannot be negative ({self.n_tiles})")

        if self.n_tiles > 0:
            highest_score_tiles = sorted_tiles_by_score[: self.n_tiles]
            highest_scaled_score_tiles = sorted_tiles_by_scaled_score[: self.n_tiles]
        else:
            highest_score_tiles = sorted_tiles_by_score
            highest_scaled_score_tiles = sorted_tiles_by_scaled_score

        return highest_score_tiles, highest_scaled_score_tiles

    @staticmethod
    def _save_report(
        report_path: str,
        highest_score_tiles: List[Tuple[float, CoordinatePair]],
        highest_scaled_score_tiles: List[Tuple[float, CoordinatePair]],
        filenames: List[str],
    ) -> None:
        """Save to ``filename`` the report of the saved tiles with the associated score.

        The CSV file

        Parameters
        ----------
        report_path : str
            Path to the report
        highest_score_tiles : List[Tuple[float, CoordinatePair]]
            List of tuples containing the score and the extraction coordinates for the
            tiles with the highest score. Each tuple represents a tile.
        List[Tuple[float, CoordinatePair]]
            List of tuples containing the scaled score between 0 and 1 and the
            extraction coordinates for the tiles with the highest score. Each tuple
            represents a tile.
        filenames : List[str]
            List of the tiles' filename
        """

        header = ["filename", "score", "scaled_score"]
        rows = [
            dict(zip(header, values))
            for values in zip(
                filenames,
                np.array(highest_score_tiles, dtype=object)[:, 0],
                np.array(highest_scaled_score_tiles, dtype=object)[:, 0],
            )
        ]

        with open(report_path, "w+", newline="") as filename:
            writer = csv.DictWriter(
                filename, fieldnames=header, lineterminator=os.linesep
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _scale_scores(
        scores: List[Tuple[float, CoordinatePair]]
    ) -> List[Tuple[float, CoordinatePair]]:
        """Scale scores between 0 and 1.

        Parameters
        ----------
        scores : List[Tuple[float, CoordinatePair]]
            Scores to be scaled

        Returns
        -------
        List[Tuple[float, CoordinatePair]])
            Scaled scores
        """
        scores_ = np.array(scores, dtype=object)[:, 0]
        coords = np.array(scores, dtype=object)[:, 1]
        scores_scaled = (scores_ - np.min(scores_)) / (
            np.max(scores_) - np.min(scores_)
        )

        return list(zip(scores_scaled, coords))

    def _scores(self, slide: Slide) -> List[Tuple[float, CoordinatePair]]:
        """Calculate the scores for all the tiles extracted from the ``slide``.

        Parameters
        ----------
        slide : Slide
            The slide to extract the tiles from.

        Returns
        -------
        List[Tuple[float, CoordinatePair]]
            List of tuples containing the score and the extraction coordinates for each
            tile. Each tuple represents a tile.
        """
        if next(self._grid_tiles_generator(slide), None) is None:
            raise RuntimeError(
                "No tiles have been generated. This could happen if `check_tissue=True`"
            )

        grid_tiles = self._grid_tiles_generator(slide)
        scores = []

        for tile, tile_wsi_coords in grid_tiles:
            score = self.scorer(tile)
            scores.append((score, tile_wsi_coords))

        return scores
