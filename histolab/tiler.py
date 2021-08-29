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
import logging
import os
from abc import abstractmethod
from itertools import zip_longest
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL

from .exceptions import LevelError, TileSizeOrCoordinatesError
from .masks import BiggestTissueBoxMask, BinaryMask
from .scorer import Scorer
from .slide import Slide
from .tile import Tile
from .types import CoordinatePair
from .util import (
    random_choice_true_mask2d,
    rectangle_to_mask,
    region_coordinates,
    regions_from_binary_mask,
    regions_to_binary_mask,
    scale_coordinates,
)

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


logger = logging.getLogger("tiler")

COORDS_WITHIN_EXTRACTION_MASK_THRESHOLD = 0.8


@runtime_checkable
class Tiler(Protocol):
    """General tiler object"""

    level: int
    mpp: float  # if provided, always takes precedence over level
    tile_size: Tuple[int, int]

    @abstractmethod
    def extract(
        self,
        slide: Slide,
        extraction_mask: BinaryMask = BiggestTissueBoxMask(),
        log_level: str = "INFO",
    ) -> None:
        pass  # pragma: no cover

    def locate_tiles(
        self,
        slide: Slide,
        extraction_mask: BinaryMask = BiggestTissueBoxMask(),
        scale_factor: int = 32,
        alpha: int = 128,
        outline: Union[str, Iterable[str], Iterable[Tuple[int]]] = "red",
        linewidth: int = 1,
        tiles: Optional[Iterable[Tile]] = None,
    ) -> PIL.Image.Image:
        """Draw tile box references on a rescaled version of the slide

        Parameters
        ----------
        slide : Slide
            Slide reference where placing the tiles
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`
        scale_factor : int, optional
            Scaling factor for the returned image. Default is 32.
        alpha : int, optional
            The alpha level to be applied to the rescaled slide. Default is 128.
        outline : Union[str, Iterable[str], Iterable[Tuple[int]]], optional
            The outline color for the tile annotations. Default is 'red'.
            You can provide this as a string compatible with matplotlib, or
            you can provide a list of the same length as the tiles, where
            each color is your assigned color for the corresponding individual
            tile. This list can be a list of matplotlib-style string colors, or
            a list of tuples of ints in the [0, 255] range, each of
            length 3, representing the red, green and blue color for each tile.
            For example, if you have two tiles that you want to be colored
            yellow, you can pass this argument as any of the following ..
            - 'yellow'
            - ['yellow', 'yellow']
            - [(255, 255, 0), (255, 255, 0)]
        linewidth : int, optional
            Thickness of line used to draw tiles. Default is 1.
        tiles : Optional[Iterable[Tile]], optional
            Tiles to visualize. Will be extracted if None. Default is None.
            You may decide to provide this argument if you do not want the
            tiles to be re-extracted for visualization if you already have
            the tiles in hand.

        Returns
        -------
        PIL.Image.Image
            PIL Image of the rescaled slide with the extracted tiles outlined
        """
        img = slide.scaled_image(scale_factor)
        img.putalpha(alpha)
        draw = PIL.ImageDraw.Draw(img)

        if tiles is None:
            tiles = (
                self._tiles_generator(slide, extraction_mask)[0]
                if isinstance(self, ScoreTiler)
                else self._tiles_generator(slide, extraction_mask)
            )
        tiles_coords = (tile[1] for tile in tiles)

        for coords, one_outline in self._tile_coords_and_outline_generator(
            tiles_coords, outline
        ):
            rescaled = scale_coordinates(coords, slide.dimensions, img.size)
            draw.rectangle(tuple(rescaled), outline=one_outline, width=linewidth)
        return img

    # ------- implementation helpers -------

    def _has_valid_tile_size(self, slide: Slide) -> bool:
        """Return True if the tile size is smaller or equal than the ``slide`` size.

        Parameters
        ----------
        slide : Slide
            The slide to check the tile size against.

        Returns
        -------
        bool
            True if the tile size is smaller or equal than the ``slide`` size at
            extraction level, False otherwise
        """
        return (
            self.tile_size[0] <= slide.level_dimensions(self.level)[0]
            and self.tile_size[1] <= slide.level_dimensions(self.level)[1]
        )

    def _scale_factor(self, slide: Slide) -> float:
        """Retrieve the scale factor that maps the original tile_size to proper one.

        Parameters
        ----------
        slide : Slide
            The slide to tile.

        Returns
        -------
        float
            Scale factor that maps the original self.tile_size to proper one.
        """
        if self.mpp is None:
            return 1.0
        return self.mpp / slide.base_mpp

    @staticmethod
    def _tile_coords_and_outline_generator(
        tiles_coords: Iterable[CoordinatePair],
        outlines: Union[str, List[str], List[Tuple[int]]],
    ) -> Union[str, Tuple[int]]:
        """Zip tile coordinates and outlines from tile and outline iterators.

        Parameters
        ----------
        tiles_coords : Iterable[CoordinatePair]
            Coordinates referring to the tiles' upper left and lower right corners.
        outlines : Union[str, Iterable[str], Iterable[Tuple[int]]]
            See docstring for ``locate_tiles`` for details.

        Yields
        -------
        CoordinatePair
            Coordinates referring to the tiles' upper left and lower right corners.
        Union[str, Tuple[int]]
            Fixed outline depending on user input to used by method ``locate_tiles``.
        """
        if isinstance(outlines, str):
            for coords in tiles_coords:
                yield coords, outlines

        elif hasattr(outlines, "__iter__"):
            for coords, one_outline in zip_longest(tiles_coords, outlines):
                if None in (coords, one_outline):
                    raise ValueError(
                        "There should be as many outlines as there are tiles!"
                    )
                yield coords, one_outline

        else:
            raise ValueError(
                "The parameter ``outline`` should be of type: "
                "str, Iterable[str], or Iterable[List[int]]"
            )

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

    def _tiles_generator(
        self, slide: Slide, extraction_mask: BinaryMask = BiggestTissueBoxMask()
    ) -> Tuple[Tile, CoordinatePair]:
        pass  # pragma: no cover

    def _tile_size(self, slide: Slide) -> Tuple[int, int]:
        """Get the proper tile size for level or mpp requested.

        Parameters
        ----------
        slide : Slide
            The slide to tile.

        Returns
        -------
        Tuple[int, int]
            Proper tile size at desired level or MPP resolution.
        """
        if self.mpp is None:
            return self.tile_size
        return tuple(int(j * self._scale_factor(slide)) for j in self.tile_size)

    def _validate_level(self, slide: Slide) -> None:
        """Validate the Tiler's level according to the Slide.

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles

        Raises
        ------
        LevelError
            If the level is not available for the slide
        """
        if len(slide.levels) - abs(self.level) < 0:
            raise LevelError(
                f"Level {self.level} not available. Number of available levels: "
                f"{len(slide.levels)}"
            )

    def _validate_tile_size(self, slide: Slide) -> None:
        """Validate the tile size according to the Slide.

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles

        Raises
        ------
        TileSizeError
            If the tile size is larger than the slide size
        """
        if not self._has_valid_tile_size(slide):
            raise TileSizeOrCoordinatesError(
                f"Tile size {self.tile_size} is larger than slide size "
                f"{slide.level_dimensions(self.level)} at level {self.level}"
            )


class GridTiler(Tiler):
    """Extractor of tiles arranged in a grid, at the given level, with the given size.

    Arguments
    ---------
    tile_size : Tuple[int, int]
        (width, height) of the extracted tiles.
    level : int, optional
        Level from which extract the tiles. Default is 0.
        Superceded by mpp if the mpp argument is provided.
    check_tissue : bool, optional
        Whether to check if the tile has enough tissue to be saved. Default is True.
    tissue_percent : float, optional
        Number between 0.0 and 100.0 representing the minimum required percentage of
        tissue over the total area of the image, default is 80.0. This is considered
        only if ``check_tissue`` equals to True.
    pixel_overlap : int, optional
       Number of overlapping pixels (for both height and width) between two adjacent
       tiles. If negative, two adjacent tiles will be strided by the absolute value of
       ``pixel_overlap``. Default is 0.
    prefix : str, optional
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str, optional
        Suffix to be added to the tile filename. Default is '.png'
    mpp : float, optional
        Micron per pixel resolution of extracted tiles. Takes precedence over level.
        Default is None.
    """

    def __init__(
        self,
        tile_size: Tuple[int, int],
        level: int = 0,
        check_tissue: bool = True,
        tissue_percent: float = 80.0,
        pixel_overlap: int = 0,
        prefix: str = "",
        suffix: str = ".png",
        mpp: float = None,
    ):
        self.tile_size = tile_size
        self.final_tile_size = tile_size
        self.level = level if mpp is None else 0
        self.mpp = mpp
        self.check_tissue = check_tissue
        self.tissue_percent = tissue_percent
        self.pixel_overlap = pixel_overlap
        self.prefix = prefix
        self.suffix = suffix

    def extract(
        self,
        slide: Slide,
        extraction_mask: BinaryMask = BiggestTissueBoxMask(),
        log_level: str = "INFO",
    ) -> None:
        """Extract tiles arranged in a grid and save them to disk, following this
        filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`.
        log_level : str, {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            Threshold level for the log messages. Default "INFO"

        Raises
        ------
        TileSizeError
            If the tile size is larger than the slide size
        LevelError
            If the level is not available for the slide
        """
        level = logging.getLevelName(log_level)
        logger.setLevel(level)
        self._validate_level(slide)
        self.tile_size = self._tile_size(slide)
        self.pixel_overlap = int(self._scale_factor(slide) * self.pixel_overlap)
        self._validate_tile_size(slide)

        grid_tiles = self._tiles_generator(slide, extraction_mask)
        tiles_counter = 0
        for tiles_counter, (tile, tile_wsi_coords) in enumerate(grid_tiles):
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            full_tile_path = os.path.join(slide.processed_path, tile_filename)
            tile.save(full_tile_path)
            logger.info(f"\t Tile {tiles_counter} saved: {tile_filename}")
        logger.info(f"{tiles_counter} Grid Tiles have been saved.")

    @property
    def tile_size(self) -> Tuple[int, int]:
        """(width, height) of the extracted tiles."""
        return self._valid_tile_size

    @tile_size.setter
    def tile_size(self, tile_size_: Tuple[int, int]):
        if tile_size_[0] < 1 or tile_size_[1] < 1:
            raise ValueError(f"Tile size must be greater than 0 ({tile_size_})")
        self._valid_tile_size = tile_size_

    # ------- implementation helpers -------

    @staticmethod
    def _are_coordinates_within_extraction_mask(
        tile_thumb_coords: CoordinatePair,
        binary_mask_region: np.ndarray,
    ) -> bool:
        """Chack whether the ``tile_thumb_coords`` are inside of ``binary_mask_region``.

        Return True if 80% of the tile area defined by tile_thumb_coords is inside the
        area of the ``binary_mask_region.

        Parameters
        ----------
        tile_thumb_coords : CoordinatePair
            Coordinates of the tile at thumbnail dimension.
        binary_mask_region : np.ndarray
            Binary mask with True inside of the tissue region considered.

        Returns
        -------
        bool
            Whether the 80% of the tile area defined by tile_thumb_coords is inside the
            area of the ``binary_mask_region.
        """

        tile_thumb_mask = rectangle_to_mask(
            dims=binary_mask_region.shape, vertices=tile_thumb_coords
        )

        tile_in_binary_mask = binary_mask_region & tile_thumb_mask

        tile_area = np.count_nonzero(tile_thumb_mask)
        tile_in_binary_mask_area = np.count_nonzero(tile_in_binary_mask)

        return tile_area > 0 and (
            tile_in_binary_mask_area / tile_area
            > COORDS_WITHIN_EXTRACTION_MASK_THRESHOLD
        )

    def _grid_coordinates_from_bbox_coordinates(
        self,
        bbox_coordinates_lvl: CoordinatePair,
        slide: Slide,
        binary_mask_region: np.ndarray,
    ) -> CoordinatePair:
        """Generate Coordinates at level 0 of grid tiles within a tissue box.

        Parameters
        ----------
        bbox_coordinates_lvl : CoordinatePair
            Coordinates of the tissue box from which to calculate the coordinates of the
            tiles.
        slide : Slide
            Slide from which to calculate the coordinates.
        binary_mask_region : np.ndarray
            Binary mask corresponding to the connected component (region) considered.

        Notes
        -----
        This method needs to be called for every connected component (region) within the
        extraction mask.

        Yields
        -------
        Iterator[CoordinatePair]
            Iterator of tiles' CoordinatePair
        """
        tile_w_lvl, tile_h_lvl = self.tile_size

        n_tiles_row = self._n_tiles_row(bbox_coordinates_lvl)
        n_tiles_column = self._n_tiles_column(bbox_coordinates_lvl)

        for i in range(n_tiles_row):
            for j in range(n_tiles_column):
                x_ul_lvl = (
                    bbox_coordinates_lvl.x_ul + tile_w_lvl * i - self.pixel_overlap
                )
                y_ul_lvl = (
                    bbox_coordinates_lvl.y_ul + tile_h_lvl * j - self.pixel_overlap
                )

                x_ul_lvl = np.clip(x_ul_lvl, bbox_coordinates_lvl.x_ul, None)
                y_ul_lvl = np.clip(y_ul_lvl, bbox_coordinates_lvl.y_ul, None)

                x_br_lvl = x_ul_lvl + tile_w_lvl
                y_br_lvl = y_ul_lvl + tile_h_lvl

                tile_lvl_coords = CoordinatePair(x_ul_lvl, y_ul_lvl, x_br_lvl, y_br_lvl)
                tile_thumb_coords = scale_coordinates(
                    reference_coords=tile_lvl_coords,
                    reference_size=slide.level_dimensions(level=self.level),
                    target_size=binary_mask_region.shape[::-1],
                )

                if self._are_coordinates_within_extraction_mask(
                    tile_thumb_coords, binary_mask_region
                ):
                    tile_wsi_coords = scale_coordinates(
                        reference_coords=tile_lvl_coords,
                        reference_size=slide.level_dimensions(level=self.level),
                        target_size=slide.level_dimensions(level=0),
                    )
                    yield tile_wsi_coords

    def _grid_coordinates_generator(
        self, slide: Slide, extraction_mask: BinaryMask = BiggestTissueBoxMask()
    ) -> CoordinatePair:
        """Generate Coordinates at level 0 of grid tiles within the tissue.

        Parameters
        ----------
        slide : Slide
            Slide from which to calculate the coordinates. Needed to calculate the
            tissue area.
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`.

        Yields
        -------
        Iterator[CoordinatePair]
            Iterator of tiles' CoordinatePair
        """
        binary_mask = extraction_mask(slide)

        regions = regions_from_binary_mask(binary_mask)
        for region in regions:
            bbox_coordinates_thumb = region_coordinates(region)  # coords of the bbox
            bbox_coordinates_lvl = scale_coordinates(
                bbox_coordinates_thumb,
                binary_mask.shape[::-1],
                slide.level_dimensions(self.level),
            )

            binary_mask_region = regions_to_binary_mask([region], binary_mask.shape)

            yield from self._grid_coordinates_from_bbox_coordinates(
                bbox_coordinates_lvl, slide, binary_mask_region
            )

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

    def _tiles_generator(
        self, slide: Slide, extraction_mask: BinaryMask = BiggestTissueBoxMask()
    ) -> Tuple[Tile, CoordinatePair]:
        """Generator of tiles arranged in a grid.

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`.

        Yields
        -------
        Tile
            Extracted tile
        CoordinatePair
            Coordinates of the slide at level 0 from which the tile has been extracted
        """
        grid_coordinates_generator = self._grid_coordinates_generator(
            slide, extraction_mask
        )
        for coords in grid_coordinates_generator:
            try:
                tile = slide.extract_tile(
                    coords,
                    tile_size=self.final_tile_size,
                    mpp=self.mpp,
                    level=self.level if self.mpp is None else None,
                )
            except TileSizeOrCoordinatesError:
                continue

            if not self.check_tissue or tile.has_enough_tissue(self.tissue_percent):
                yield tile, coords


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
        Superceded by mpp if the mpp argument is provided.
    seed : int, optional
        Seed for RandomState. Must be convertible to 32 bit unsigned integers. Default
        is 7.
    check_tissue : bool, optional
        Whether to check if the tile has enough tissue to be saved. Default is True.
    tissue_percent : float, optional
        Number between 0.0 and 100.0 representing the minimum required percentage of
        tissue over the total area of the image, default is 80.0. This is considered
        only if ``check_tissue`` equals to True.
    prefix : str, optional
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str, optional
        Suffix to be added to the tile filename. Default is '.png'
    max_iter : int, optional
        Maximum number of iterations performed when searching for eligible (if
        ``check_tissue=True``) tiles. Must be grater than or equal to ``n_tiles``.
    mpp : float, optional
        Micron per pixel resolution. If provided, takes precedence over level.
        Default is None.
    """

    def __init__(
        self,
        tile_size: Tuple[int, int],
        n_tiles: int,
        level: int = 0,
        seed: int = 7,
        check_tissue: bool = True,
        tissue_percent: float = 80.0,
        prefix: str = "",
        suffix: str = ".png",
        max_iter: int = int(1e4),
        mpp: float = None,
    ):
        self.tile_size = tile_size
        self.final_tile_size = tile_size
        self.n_tiles = n_tiles
        self.max_iter = max_iter
        self.level = level if mpp is None else 0
        self.mpp = mpp
        self.seed = seed
        self.check_tissue = check_tissue
        self.tissue_percent = tissue_percent
        self.prefix = prefix
        self.suffix = suffix

    def extract(
        self,
        slide: Slide,
        extraction_mask: BinaryMask = BiggestTissueBoxMask(),
        log_level: str = "INFO",
    ) -> None:
        """Extract random tiles and save them to disk, following this filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`.
        log_level: str, {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            Threshold level for the log messages. Default "INFO"

        Raises
        ------
        TileSizeError
            If the tile size is larger than the slide size
        LevelError
            If the level is not available for the slide
        """
        level = logging.getLevelName(log_level)
        logger.setLevel(level)
        self._validate_level(slide)
        self.tile_size = self._tile_size(slide)
        self._validate_tile_size(slide)

        random_tiles = self._tiles_generator(slide, extraction_mask)

        tiles_counter = 0
        for tiles_counter, (tile, tile_wsi_coords) in enumerate(random_tiles):
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            full_tile_path = os.path.join(slide.processed_path, tile_filename)
            tile.save(full_tile_path)
            logger.info(f"\t Tile {tiles_counter} saved: {tile_filename}")
        logger.info(f"{tiles_counter+1} Random Tiles have been saved.")

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

    def _random_tile_coordinates(
        self, slide: Slide, extraction_mask: BinaryMask = BiggestTissueBoxMask()
    ) -> CoordinatePair:
        """Return 0-level Coordinates of a tile picked at random within the box.

        Parameters
        ----------
        slide : Slide
            Slide from which calculate the coordinates. Needed to calculate the box.
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`.

        Returns
        -------
        CoordinatePair
            Random tile Coordinates at level 0
        """
        binary_mask = extraction_mask(slide)
        tile_w_lvl, tile_h_lvl = self.tile_size

        x_ul_lvl, y_ul_lvl = random_choice_true_mask2d(binary_mask)

        # Scale tile dimensions to extraction mask dimensions
        tile_w_thumb = (
            tile_w_lvl * binary_mask.shape[1] / slide.level_dimensions(self.level)[0]
        )
        tile_h_thumb = (
            tile_h_lvl * binary_mask.shape[0] / slide.level_dimensions(self.level)[1]
        )

        x_br_lvl = x_ul_lvl + tile_w_thumb
        y_br_lvl = y_ul_lvl + tile_h_thumb

        tile_wsi_coords = scale_coordinates(
            reference_coords=CoordinatePair(x_ul_lvl, y_ul_lvl, x_br_lvl, y_br_lvl),
            reference_size=binary_mask.shape[::-1],
            target_size=slide.dimensions,
        )

        return tile_wsi_coords

    def _tiles_generator(
        self, slide: Slide, extraction_mask: BinaryMask = BiggestTissueBoxMask()
    ) -> Tuple[Tile, CoordinatePair]:
        """Generate Random Tiles within a slide box.

        Stops if:
        * the number of extracted tiles is equal to ``n_tiles`` OR
        * the maximum number of iterations ``max_iter`` is reached

        Parameters
        ----------
        slide : Slide
            The Whole Slide Image from which to extract the tiles.
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`.

        Yields
        ------
        tile : Tile
            The extracted Tile
        coords : CoordinatePair
            The level-0 coordinates of the extracted tile
        """
        np.random.seed(self.seed)
        iteration = valid_tile_counter = 0

        while True:
            tile_wsi_coords = self._random_tile_coordinates(slide, extraction_mask)
            try:
                tile = slide.extract_tile(
                    tile_wsi_coords,
                    tile_size=self.final_tile_size,
                    mpp=self.mpp,
                    level=self.level if self.mpp is None else None,
                )
            except TileSizeOrCoordinatesError:
                iteration -= 1
                continue

            if not self.check_tissue or tile.has_enough_tissue(self.tissue_percent):
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
        Superceded by mpp if the mpp argument is provided.
    check_tissue : bool, optional
        Whether to check if the tile has enough tissue to be saved. Default is True.
    tissue_percent : float, optional
        Number between 0.0 and 100.0 representing the minimum required percentage of
        tissue over the total area of the image, default is 80.0. This is considered
        only if ``check_tissue`` equals to True.
    pixel_overlap : int, optional
       Number of overlapping pixels (for both height and width) between two adjacent
       tiles. If negative, two adjacent tiles will be strided by the absolute value of
       ``pixel_overlap``. Default is 0.
    prefix : str, optional
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str, optional
        Suffix to be added to the tile filename. Default is '.png'
    mpp : float, optional.
        Micron per pixel resolution. If provided, takes precedence over level.
        Default is None.
    """

    def __init__(
        self,
        scorer: Scorer,
        tile_size: Tuple[int, int],
        n_tiles: int = 0,
        level: int = 0,
        check_tissue: bool = True,
        tissue_percent: float = 80.0,
        pixel_overlap: int = 0,
        prefix: str = "",
        suffix: str = ".png",
        mpp: float = None,
    ):
        self.scorer = scorer
        self.n_tiles = n_tiles

        super().__init__(
            tile_size,
            level,
            check_tissue,
            tissue_percent,
            pixel_overlap,
            prefix,
            suffix,
            mpp=mpp,
        )

    def extract(
        self,
        slide: Slide,
        extraction_mask: BinaryMask = BiggestTissueBoxMask(),
        report_path: str = None,
        log_level: str = "INFO",
    ) -> None:
        """Extract grid tiles and save them to disk, according to a scoring function and
        following this filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Save a CSV report file with the saved tiles and the associated score.

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`.
        report_path : str, optional
            Path to the CSV report. If None, no report will be saved
        log_level: str, {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            Threshold level for the log messages. Default "INFO"

        Raises
        ------
        TileSizeError
            If the tile size is larger than the slide size
        LevelError
            If the level is not available for the slide
        """
        level = logging.getLevelName(log_level)
        logger.setLevel(level)
        self._validate_level(slide)
        self.tile_size = self._tile_size(slide)
        self.pixel_overlap = int(self._scale_factor(slide) * self.pixel_overlap)
        self._validate_tile_size(slide)

        highest_score_tiles, highest_scaled_score_tiles = self._tiles_generator(
            slide, extraction_mask
        )

        tiles_counter = 0
        filenames = []

        for tiles_counter, (score, tile_wsi_coords) in enumerate(highest_score_tiles):
            tile = slide.extract_tile(
                tile_wsi_coords,
                tile_size=self.final_tile_size,
                mpp=self.mpp,
                level=self.level if self.mpp is None else None,
            )
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            tile.save(os.path.join(slide.processed_path, tile_filename))
            filenames.append(tile_filename)
            logger.info(
                f"\t Tile {tiles_counter} - score: {score} saved: {tile_filename}"
            )

        if report_path:
            self._save_report(
                report_path, highest_score_tiles, highest_scaled_score_tiles, filenames
            )

        logger.info(f"{tiles_counter+1} Grid Tiles have been saved.")

    # ------- implementation helpers -------

    def _tiles_generator(
        self, slide: Slide, extraction_mask: BinaryMask = BiggestTissueBoxMask()
    ) -> Tuple[List[Tuple[float, CoordinatePair]], List[Tuple[float, CoordinatePair]]]:
        r"""Calculate the tiles with the highest scores and their extraction coordinates

        Parameters
        ----------
        slide : Slide
            The slide to extract the tiles from.
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`.

        Returns
        -------
        Tuple[List[Tuple[float, CoordinatePair]], List[Tuple[float, CoordinatePair]]]
            List of tuples containing the scores and the extraction coordinates
            for the tiles with the highest scores. If scaled=True, each score `s_i` of
            the i-th tile is normalized as

            .. math::

                s_{\hat{i}}=\frac{s_i-\min_{j\in T}{s_j}}{\max_{j\in T}{s_j}-\min_{j\in T}{s_j}}

            where `T` is the set of all the retrieved tiles. Notice that the normalized
            scores range between 0 and 1. This could be useful to have a more intuitive
            comparison between the scores. Each tuple represents a tile.

        Raises
        ------
        ValueError
            If ``n_tiles`` is negative.
        """  # noqa
        all_scores = self._scores(slide, extraction_mask)
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
        highest_scaled_score_tiles : List[Tuple[float, CoordinatePair]]
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

    def _scores(
        self, slide: Slide, extraction_mask: BinaryMask = BiggestTissueBoxMask()
    ) -> List[Tuple[float, CoordinatePair]]:
        """Calculate the scores for all the tiles extracted from the ``slide``.

        Parameters
        ----------
        slide : Slide
            The slide to extract the tiles from.
        extraction_mask : BinaryMask, optional
            BinaryMask object defining how to compute a binary mask from a Slide.
            Default `BiggestTissueBoxMask`.

        Returns
        -------
        List[Tuple[float, CoordinatePair]]
            List of tuples containing the score and the extraction coordinates for each
            tile. Each tuple represents a tile.
        """
        if next(super()._tiles_generator(slide, extraction_mask), None) is None:
            raise RuntimeError(
                "No tiles have been generated. This could happen if `check_tissue=True`"
            )

        grid_tiles = super()._tiles_generator(slide, extraction_mask)
        scores = []

        for tile, tile_wsi_coords in grid_tiles:
            score = self.scorer(tile)
            scores.append((score, tile_wsi_coords))

        return scores
