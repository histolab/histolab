from abc import abstractmethod
from typing import Tuple

import numpy as np
import sparse

from .slide import Slide
from .tile import Tile
from .types import CoordinatePair
from .util import (
    lru_cache,
    region_coordinates,
    regions_from_binary_mask,
    resize_mask,
    scale_coordinates,
)

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Tiler(Protocol):

    level: int
    tile_size: int

    @lru_cache(maxsize=100)
    def box_mask(self, slide: Slide) -> sparse._coo.core.COO:
        """Return binary mask at level 0 of the box to consider for tiles extraction.

        The mask pixels set to True will be the ones corresponding to the tissue box.

        Parameters
        ----------
        slide : Slide
            The Slide from which to extract the extraction mask

        Returns
        -------
        sparse._coo.core.COO
            Extraction mask at level 0
        """

        return slide.biggest_tissue_box_mask

    @lru_cache(maxsize=100)
    def box_mask_lvl(self, slide: Slide) -> sparse._coo.core.COO:
        """Return binary mask at target level of the box to consider for the extraction.

        The mask pixels set to True will be the ones corresponding to the tissue box.

        Parameters
        ----------
        slide : Slide
            The Slide from which to extract the extraction mask

        Returns
        -------
        sparse._coo.core.COO
            Extraction mask at target level
        """

        box_mask_wsi = self.box_mask(slide)

        if self.level != 0:
            return resize_mask(
                box_mask_wsi, target_dimensions=slide.level_dimensions(self.level),
            )
        else:
            return box_mask_wsi

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

    @abstractmethod
    def extract(self, slide: Slide):
        raise NotImplementedError


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

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self._valid_tile_size

    @tile_size.setter
    def tile_size(self, tile_size_: Tuple[int, int]):
        if tile_size_[0] < 1 or tile_size_[1] < 1:
            raise ValueError(f"Tile size must be greater than 0 ({tile_size_})")
        self._valid_tile_size = tile_size_

    @property
    def level(self) -> int:
        return self._valid_level

    @level.setter
    def level(self, level_: int):
        if level_ < 0:
            raise ValueError(f"Level cannot be negative ({level_})")
        self._valid_level = level_

    def extract(self, slide: Slide):
        """Extract tiles arranged in a grid and save them to disk, following this
        filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        """
        grid_tiles = self._grid_tiles_generator(slide)

        tiles_counter = 0

        for tiles_counter, (tile, tile_wsi_coords) in enumerate(grid_tiles):
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            tile.save(tile_filename)
            print(f"\t Tile {tiles_counter} saved: {tile_filename}")

        print(f"{tiles_counter+1} Grid Tiles have been saved.")

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

        x_ul_lvl_offset = bbox_coordinates.x_ul
        y_ul_lvl_offset = bbox_coordinates.y_ul

        for i in range(n_tiles_row):
            for j in range(n_tiles_column):
                x_ul_lvl = x_ul_lvl_offset + tile_w_lvl * j - self.pixel_overlap
                y_ul_lvl = y_ul_lvl_offset + tile_h_lvl * i - self.pixel_overlap

                x_ul_lvl = np.clip(x_ul_lvl, x_ul_lvl_offset, None)
                y_ul_lvl = np.clip(y_ul_lvl, y_ul_lvl_offset, None)

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
        box_mask_lvl = self.box_mask_lvl(slide)

        regions = regions_from_binary_mask(box_mask_lvl.todense())
        for region in regions:  # at the moment there is only one region
            bbox_coordinates = region_coordinates(region)
            yield from self._grid_coordinates_from_bbox_coordinates(
                bbox_coordinates, slide
            )

    def _grid_tiles_generator(self, slide: Slide) -> (Tile, CoordinatePair):
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
        max_iter: int = 1e4,
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

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self._valid_tile_size

    @tile_size.setter
    def tile_size(self, tile_size_: Tuple[int, int]):
        if tile_size_[0] < 1 or tile_size_[1] < 1:
            raise ValueError(f"Tile size must be greater than 0 ({tile_size_})")
        self._valid_tile_size = tile_size_

    @property
    def level(self) -> int:
        return self._valid_level

    @level.setter
    def level(self, level_: int):
        if level_ < 0:
            raise ValueError(f"Level cannot be negative ({level_})")
        self._valid_level = level_

    @property
    def max_iter(self) -> int:
        return self._valid_max_iter

    @max_iter.setter
    def max_iter(self, max_iter_: int = 1e4):
        if max_iter_ < self.n_tiles:
            raise ValueError(
                f"The maximum number of iterations ({max_iter_}) must be grater than or "
                f"equal to the maximum number of tiles ({self.n_tiles})."
            )
        self._valid_max_iter = max_iter_

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
            tile.save(tile_filename)
            print(f"\t Tile {tiles_counter} saved: {tile_filename}")
        print(f"{tiles_counter+1} Random Tiles have been saved.")

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
        box_mask_lvl = self.box_mask_lvl(slide)
        tile_w_lvl, tile_h_lvl = self.tile_size

        x_ul_lvl = np.random.choice(sparse.where(box_mask_lvl)[1])
        y_ul_lvl = np.random.choice(sparse.where(box_mask_lvl)[0])

        x_br_lvl = x_ul_lvl + tile_w_lvl
        y_br_lvl = y_ul_lvl + tile_h_lvl

        tile_wsi_coords = scale_coordinates(
            reference_coords=CoordinatePair(x_ul_lvl, y_ul_lvl, x_br_lvl, y_br_lvl),
            reference_size=slide.level_dimensions(level=self.level),
            target_size=slide.level_dimensions(level=0),
        )

        return tile_wsi_coords

    def _random_tiles_generator(self, slide: Slide) -> (Tile, CoordinatePair):
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
