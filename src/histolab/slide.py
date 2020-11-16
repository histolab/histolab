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

"""Provides the Slide class.

Slide is the main API class for manipulating slide objects.
"""

import math
import os
import pathlib
from functools import lru_cache
from typing import List, Tuple, Union

import ntpath
import numpy as np
import openslide
import PIL

from .exceptions import LevelError
from .filters.compositions import FiltersComposition
from .tile import Tile
from .types import CoordinatePair, Region
from .util import (
    lazyproperty,
    polygon_to_mask_array,
    region_coordinates,
    regions_from_binary_mask,
    scale_coordinates,
)

IMG_EXT = "png"


class Slide:
    """Provide Slide objects and expose property and methods.

    Arguments
    ---------
    path : str
        Path where the WSI is saved.
    processed_path : str
        Path where thumbnails and scaled images will be saved to.
    """

    def __init__(self, path: str, processed_path: str) -> None:
        self._path = path
        self._processed_path = processed_path

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(path={self._path}, processed_path={self._processed_path})"
        )

    # ---public interface methods and properties---

    @lazyproperty
    @lru_cache(maxsize=100)
    def biggest_tissue_box_mask(self) -> np.ndarray:
        """Return the thumbnail binary mask of the box containing the max tissue area.

        Returns
        -------
        mask: np.ndarray
            Binary mask of the box containing the max area of tissue. The dimensions are
            those of the thumbnail.

        """
        thumb = self._wsi.get_thumbnail(self._thumbnail_size)
        filters = FiltersComposition(Slide).tissue_mask_filters

        thumb_mask = filters(thumb)
        regions = regions_from_binary_mask(thumb_mask)
        biggest_region = self._biggest_regions(regions, n=1)[0]
        biggest_region_coordinates = region_coordinates(biggest_region)
        thumb_bbox_mask = polygon_to_mask_array(
            self._thumbnail_size, biggest_region_coordinates
        )
        return thumb_bbox_mask

    @lazyproperty
    def dimensions(self) -> Tuple[int, int]:
        """Return the slide dimensions (w,h) at level 0.

        Returns
        -------
        dimensions : tuple(width, height)
        """
        return self._wsi.dimensions

    def extract_tile(self, coords: CoordinatePair, level: int) -> Tile:
        """Extract a tile of the image at the selected level.

        Parameters
        ----------
        coords : CoordinatePair
            Coordinates at level 0 from which to extract the tile.
        level : int
            Level from which to extract the tile.

        Returns
        -------
        tile : Tile
            Image containing the selected tile.
        """

        if not self._has_valid_coords(coords):
            # OpenSlide doesn't complain if the coordinates for extraction are wrong,
            # but it returns an odd image.
            raise ValueError(
                f"Extraction Coordinates {coords} not valid for slide with dimensions "
                f"{self.dimensions}"
            )

        coords_level = scale_coordinates(
            reference_coords=coords,
            reference_size=self.level_dimensions(level=0),
            target_size=self.level_dimensions(level=level),
        )

        h_l = coords_level.y_br - coords_level.y_ul
        w_l = coords_level.x_br - coords_level.x_ul

        image = self._wsi.read_region(
            location=(coords.x_ul, coords.y_ul), level=level, size=(w_l, h_l)
        )
        tile = Tile(image, coords, level)
        return tile

    def level_dimensions(self, level: int = 0) -> Tuple[int, int]:
        """Return the slide dimensions (w,h) at the specified level

        Parameters
        ---------
        level : int
            The level which dimensions are requested, default is 0

        Returns
        -------
        dimensions : tuple (width, height)
        """
        try:
            return self._wsi.level_dimensions[level]
        except IndexError:
            raise LevelError(
                f"Level {level} not available. Number of available levels: "
                f"{len(self._wsi.level_dimensions)}"
            )

    @lazyproperty
    def levels(self) -> List[int]:
        """Return the slide's available levels

        Returns
        -------
        List[int]
            The levels available
        """
        return list(range(len(self._wsi.level_dimensions)))

    @lazyproperty
    def name(self) -> str:
        """Retrieve the slide name without extension.

        Returns
        -------
        name : str
        """
        return ntpath.basename(self._path).split(".")[0]

    @lazyproperty
    def processed_path(self) -> str:
        """Retrieve the path to store processed files generated from the slide.

        Returns
        -------
        str
            Path to store processed files generated from the slide
        """
        return self._processed_path

    def resampled_array(self, scale_factor: int = 32) -> np.array:
        """Retrieve the resampled array from the original slide

        Parameters
        ----------
        scale_factor : int, optional
            Image scaling factor. Default is 32.
        Returns
        ----------
        resampled_array: np.ndarray
            Resampled array
        """
        return self._resample(scale_factor)[1]

    def save_scaled_image(self, scale_factor: int = 32) -> None:
        """Save a scaled image in the correct path

        Parameters
        ----------
        scale_factor : int, optional
            Image scaling factor. Default is 32.
        """
        os.makedirs(self._processed_path, exist_ok=True)
        img = self._resample(scale_factor)[0]
        img.save(self.scaled_image_path(scale_factor))

    def save_thumbnail(self) -> None:
        """Save a thumbnail in the correct path"""
        os.makedirs(self._processed_path, exist_ok=True)

        img = self._wsi.get_thumbnail(self._thumbnail_size)

        folder = os.path.dirname(self.thumbnail_path)
        pathlib.Path(folder).mkdir(exist_ok=True)
        img.save(self.thumbnail_path)

    def scaled_image_path(self, scale_factor: int = 32) -> str:
        """Return slide image path.

        Parameters
        ----------
        scale_factor : int, optional
            Image scaling factor. Default is 32.

        Returns
        -------
        img_path : str
        """
        img_path = self._breadcrumb(self._processed_path, scale_factor)
        return img_path

    def show(self) -> None:
        """Display the slide thumbnail.

        NOTE: A new window of your OS image viewer will be opened.
        """
        try:
            thumbnail = PIL.Image.open(self.thumbnail_path)
            thumbnail.show()  # pragma: no cover
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Cannot display the slide thumbnail:{error}")

    @lazyproperty
    def thumbnail_path(self) -> str:
        """Return thumbnail image path.

        Returns
        -------
        thumb_path : str
        """
        thumb_path = os.path.join(
            self._processed_path, "thumbnails", f"{self.name}.{IMG_EXT}"
        )
        return thumb_path

    # ------- implementation helpers -------

    @staticmethod
    def _biggest_regions(regions: List[Region], n: int = 1) -> List[Region]:
        """Return the biggest ``n`` regions.

        Parameters
        ----------
        regions : List[Region]
            List of regions
        n : int, optional
            Number of regions to return, by default 1

        Returns
        -------
        List[Region]
            List of ``n`` biggest regions

        Raises
        ------
        ValueError
            If ``n`` is not between 1 and the number of elements of ``regions``
        """

        if not 1 <= n <= len(regions):
            raise ValueError(f"n should be between 1 and {len(regions)}, got {n}")

        sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
        return sorted_regions[:n]

    def _breadcrumb(self, directory_path: str, scale_factor: int = 32) -> str:
        """Returns a complete path according to the give directory path

        Parameters
        ----------
        directory_path: str
        scale_factor : int, 32 by default
            Image scaling factor

        Returns
        -------
        final_path: str, a real and complete path starting from the dir path
                    e.g. /processed_path/my-image-name-x32/
                         /thumb_path/my-image-name-x32/thumbs
        """
        large_w, large_h, new_w, new_h = self._resampled_dimensions(scale_factor)
        if {large_w, large_h, new_w, new_h} == {None}:
            final_path = os.path.join(directory_path, f"{self.name}*.{IMG_EXT}")
        else:
            final_path = os.path.join(
                directory_path,
                f"{self.name}-{scale_factor}x-{large_w}x{large_h}-{new_w}x"
                f"{new_h}.{IMG_EXT}",
            )
        return final_path

    def _has_valid_coords(self, coords: CoordinatePair) -> bool:
        """Check if ``coords`` are valid 0-level coordinates.

        Parameters
        ----------
        coords : CoordinatePair
            Coordinates at level 0 to check

        Returns
        -------
        bool
            True if the coordinates are valid, False otherwise
        """
        return (
            0 <= coords.x_ul < self.dimensions[0]
            and 0 <= coords.x_br < self.dimensions[0]
            and 0 <= coords.y_ul < self.dimensions[1]
            and 0 <= coords.y_br < self.dimensions[1]
        )

    def _resample(self, scale_factor: int = 32) -> Tuple[PIL.Image.Image, np.array]:
        """Converts a slide to a scaled-down PIL image.

        The PIL image is also converted to array.
        image is the scaled-down PIL image, original width and original height
        are the width and height of the slide, new width and new height are the
        dimensions of the PIL image.

        Parameters
        ----------
        scale_factor : int, 32 by default
            Image scaling factor

        Returns
        -------
        img, arr_img
        """

        _, _, new_w, new_h = self._resampled_dimensions(scale_factor)
        level = self._wsi.get_best_level_for_downsample(scale_factor)
        whole_slide_image = self._wsi.read_region(
            (0, 0), level, self._wsi.level_dimensions[level]
        )
        # ---converts openslide read_region to an actual RGBA image---
        whole_slide_image = whole_slide_image.convert("RGB")
        img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
        arr_img = np.asarray(img)
        return img, arr_img

    def _resampled_dimensions(
        self, scale_factor: int = 32
    ) -> Tuple[int, int, int, int]:
        """Scale the slide dimensions of a specified factor.

        Parameters
        ---------
        scale_factor : int, 32 by default
            Image scaling factor

        Returns
        -------
        tuple
            Original slide dimensions and scaled dimensions
        """
        large_w, large_h = self.dimensions
        new_w = math.floor(large_w / scale_factor)
        new_h = math.floor(large_h / scale_factor)
        return large_w, large_h, new_w, new_h

    @lazyproperty
    def _thumbnail_size(self) -> Tuple:
        r"""Compute the thumbnail size proportionally to the slide dimensions.

        If the size of the slide is (v, m) where v has magnitude w and m has magnitude
        n, that is,

        .. math::

            \left\lceil{\\log_{10}(v)}\right\rceil = w

        and

        .. math::

            \left\lceil{\log_{10}(m)}\right\rceil = n

        then the thumbnail size is computed as:

        .. math::

            \big(\frac{v}{10^{w-2}},\frac{v}{10^{n-2}}\big)

        Returns
        -------
        Tuple
            Thumbnail size
        """
        return tuple(
            [
                int(s / np.power(10, math.ceil(math.log10(s)) - 3))
                for s in self.dimensions
            ]
        )

    @lazyproperty
    def _wsi(self) -> Union[openslide.OpenSlide, openslide.ImageSlide]:
        """Open the slide and returns an openslide object

        Returns
        -------
        slide : OpenSlide object
            An OpenSlide object representing a whole-slide image.
        """
        try:
            slide = openslide.open_slide(self._path)
        except PIL.UnidentifiedImageError:
            raise PIL.UnidentifiedImageError(
                "Your wsi has something broken inside, a doctor is needed"
            )
        except FileNotFoundError:
            raise FileNotFoundError("The wsi path resource doesn't exist")
        return slide


class SlideSet:
    """Slideset object. It is considered a collection of Slides."""

    def __init__(
        self, slides_path: str, processed_path: str, valid_extensions: list
    ) -> None:
        self._slides_path = slides_path
        self._processed_path = processed_path
        self._valid_extensions = valid_extensions

    # ---public interface methods and properties---

    def save_scaled_slides(self, scale_factor: int = 32, n: int = 0) -> None:
        """Save rescaled images

        Parameters
        ----------
        scale_factor: int, optional
            Image scaling factor. Default is 32.
        n: int, optional
            First n slides in dataset folder to rescale and save. Default is 0, meaning
            that all the slides will be saved.
        """
        os.makedirs(self._processed_path, exist_ok=True)
        n = self.total_slides if (n > self.total_slides or n == 0) else n
        for slide in self.slides[:n]:
            slide.save_scaled_image(scale_factor)

    def save_thumbnails(self, n: int = 0) -> None:
        """Save thumbnails

        Parameters
        ----------
        n: int. optional
            First n slides in dataset folder. Default is 0, meaning that the thumbnails
            of all the slides will be saved.
        """
        os.makedirs(self._processed_path, exist_ok=True)
        n = self.total_slides if (n > self.total_slides or n == 0) else n
        for slide in self.slides[:n]:
            slide.save_thumbnail()

    @lazyproperty
    def slides(self) -> List[Slide]:
        """Retrieve all the slides of the slideset

        Returns
        ----------
        slides: list, list of `Slide` objects
        """
        return [
            Slide(os.path.join(self._slides_path, _path), self._processed_path)
            for _path in os.listdir(self._slides_path)
            if os.path.splitext(_path)[1] in self._valid_extensions
        ]

    @lazyproperty
    def slides_stats(self) -> dict:
        """Retrieve statistic/graphs of slides files contained in the dataset.

        Returns
        ----------
        basic_stats: dict of slides stats e.g. min_size, avg_size, etc...
        """
        return {
            "no_of_slides": self.total_slides,
            "max_width": self._max_width_slide,
            "max_height": self._max_height_slide,
            "max_size": self._max_size_slide,
            "min_width": self._min_width_slide,
            "min_height": self._min_height_slide,
            "min_size": self._min_size_slide,
            "avg_width": self._avg_width_slide,
            "avg_height": self._avg_height_slide,
            "avg_size": self._avg_size_slide,
        }

    @lazyproperty
    def total_slides(self) -> int:
        """Number of slides within the slideset.

        Returns
        ----------
        n: int
            Number of slides.
        """
        return len(self.slides)

    # ---private interface methods and properties---

    @lazyproperty
    def _avg_width_slide(self) -> float:
        return sum(d["width"] for d in self._slides_dimensions) / self.total_slides

    @lazyproperty
    def _avg_height_slide(self) -> float:
        return sum(d["height"] for d in self._slides_dimensions) / self.total_slides

    @lazyproperty
    def _avg_size_slide(self) -> float:
        return sum(d["size"] for d in self._slides_dimensions) / self.total_slides

    @lazyproperty
    def _max_height_slide(self) -> dict:
        max_height = max(self._slides_dimensions, key=lambda x: x["height"])
        return {"slide": max_height["slide"], "height": max_height["height"]}

    @lazyproperty
    def _max_size_slide(self) -> dict:
        max_size = max(self._slides_dimensions, key=lambda x: x["size"])
        return {"slide": max_size["slide"], "size": max_size["size"]}

    @lazyproperty
    def _max_width_slide(self) -> dict:
        max_width = max(self._slides_dimensions, key=lambda x: x["width"])
        return {"slide": max_width["slide"], "width": max_width["width"]}

    @lazyproperty
    def _min_width_slide(self) -> dict:
        min_width = min(self._slides_dimensions, key=lambda x: x["width"])
        return {"slide": min_width["slide"], "width": min_width["width"]}

    @lazyproperty
    def _min_height_slide(self) -> dict:
        min_height = min(self._slides_dimensions, key=lambda x: x["height"])
        return {"slide": min_height["slide"], "height": min_height["height"]}

    @lazyproperty
    def _min_size_slide(self) -> dict:
        min_size = min(self._slides_dimensions, key=lambda x: x["size"])
        return {"slide": min_size["slide"], "size": min_size["size"]}

    @lazyproperty
    def _slides_dimensions(self) -> List[dict]:
        return [
            {
                "slide": slide.name,
                "width": slide.dimensions[0],
                "height": slide.dimensions[1],
                "size": slide.dimensions[0] * slide.dimensions[1],
            }
            for slide in self.slides
        ]

    @lazyproperty
    def _slides_dimensions_list(self):
        return [slide.dimensions for slide in self.slides]
