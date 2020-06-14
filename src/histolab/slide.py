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
import ntpath
import os
import pathlib
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import openslide
import PIL
from matplotlib.figure import Figure as matplotlib_figure
from skimage.measure import label, regionprops

import src.histolab.filters.image_filters as imf
import src.histolab.filters.morphological_filters as mof

from .tile import Tile
from .types import CoordinatePair, Region
from .util import (
    lazyproperty,
    lru_cache,
    polygon_to_mask_array,
    resize_mask,
    scale_coordinates,
)

IMG_EXT = "png"
THUMBNAIL_SIZE = 1000

# needed for matplotlib
# TODO: can we get rid of this shit?
plt.ioff()


class Slide(object):
    """Provides Slide objects and expose property and methods."""

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
    def dimensions(self) -> Tuple[int, int]:
        """Returns the slide dimensions (w,h)

        Returns
        -------
        dimensions : tuple(width, height)
        """
        return self._wsi.dimensions

    def level_dimensions(self, level: int = 0) -> Tuple[int, int]:
        """Returns the slide dimensions (w,h) at the specified level

        Parameters
        ---------
        level : int
            The level which dimensions are requested, default is 0

        Returns
        -------
        dimensions : tuple(width, height)
        """
        return self._wsi.level_dimensions[level]

    @lazyproperty
    @lru_cache(maxsize=100)
    def biggest_tissue_box_mask(self) -> np.ndarray:
        """Returns the binary mask of the box containing the max area of tissue.

        Returns
        -------
        mask: np.ndarray
            Binary mask of the box containing the max area of tissue.

        """

        thumb = self._wsi.get_thumbnail((1000, 1000))
        filters = self._main_tissue_areas_mask_filters

        thumb_mask = filters(thumb)
        regions = self._regions_from_binary_mask(thumb_mask)
        biggest_region = self._biggest_regions(regions, n=1)[0]
        biggest_region_coordinates = self._region_coordinates(biggest_region)
        thumb_bbox_mask = polygon_to_mask_array(
            (1000, 1000), biggest_region_coordinates
        )
        return resize_mask(thumb_bbox_mask, self.dimensions)

    def extract_tile(self, coords: CoordinatePair, level: int) -> Tile:
        """Extract a tile of the image at the selected level.

        Parameters
        ----------
        coords : CoordinatePair
            Coordinates in the first level (0)
        level : int
            Level from which to extract the tile

        Returns
        -------
        tile : Tile
            Image containing the selected tile
        """

        if (
            coords.x_ul >= self.dimensions[0]
            or coords.x_br >= self.dimensions[0]
            or coords.y_ul >= self.dimensions[1]
            or coords.y_br >= self.dimensions[1]
        ):
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

    @lazyproperty
    def name(self) -> str:
        """Retrieves the slide name without extension.

        Returns
        -------
        name : str
        """
        return ntpath.basename(self._path).split(".")[0]

    def resampled_array(self, scale_factor: int = 32) -> np.array:
        """Retrieves the resampled array from the original slide

        Parameters
        ----------
        scale_factor : int, 32 by default
            Image scaling factor
        Returns
        ----------
        resampled_array: np.array
        """
        return self._resample(scale_factor)[1]

    def save_scaled_image(self, scale_factor: int = 32) -> None:
        """Save a scaled image in the correct path

        Parameters
        ----------
        scale_factor : int, 32 by default
            Image scaling factor
        """
        os.makedirs(self._processed_path, exist_ok=True)
        img = self._resample(scale_factor)[0]
        img.save(self.scaled_image_path(scale_factor))

    def save_thumbnail(self) -> None:
        """Save a thumbnail in the correct path"""
        os.makedirs(self._processed_path, exist_ok=True)

        img = self._wsi.get_thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))

        folder = os.path.dirname(self.thumbnail_path)
        pathlib.Path(folder).mkdir(exist_ok=True)
        img.save(self.thumbnail_path)

    def scaled_image_path(self, scale_factor: int = 32) -> str:
        """Returns slide image path.

        Parameters
        ----------
        scale_factor : int, 32 by default
            Image scaling factor

        Returns
        -------
        img_path : str
        """
        img_path = self._breadcumb(self._processed_path, scale_factor)
        return img_path

    def show(self) -> None:
        """Display the slide thumbnail.

        NOTE: A new window of your OS image viewer will be opened.
        """
        try:
            thumbnail = PIL.Image.open(self.thumbnail_path)
            thumbnail.show()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cannot display the slide thumbnail:{e}")

    @lazyproperty
    def thumbnail_path(self) -> str:
        """Returns thumbnail image path.

        Returns
        -------
        thumb_path : str
        """
        thumb_path = os.path.join(
            self._processed_path, "thumbnails", f"{self.name}.{IMG_EXT}"
        )
        return thumb_path

    # ---private interface methods and properties---

    def _biggest_regions(self, regions: List[Region], n: int = 1) -> List[Region]:
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

    def _breadcumb(self, directory_path: str, scale_factor: int = 32) -> str:
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

    @lazyproperty
    def _main_tissue_areas_mask_filters(self) -> imf.Compose:
        """Return a filters composition to get a binary mask of the main tissue regions.

        Returns
        -------
        imf.Compose
            Filters composition
        """
        filters = imf.Compose(
            [
                imf.RgbToGrayscale(),
                imf.OtsuThreshold(),
                mof.BinaryDilation(),
                mof.RemoveSmallHoles(),
                mof.RemoveSmallObjects(),
            ]
        )
        return filters

    def _region_coordinates(self, region: Region) -> CoordinatePair:
        """Extract bbox coordinates from the region.

        Parameters
        ----------
        region : Region
            Region from which to extract the coordinates of the bbox

        Returns
        -------
        CoordinatePair
            Coordinates of the bbox
        """
        y_ul, x_ul, y_br, x_br = region.bbox
        return CoordinatePair(x_ul, y_ul, x_br, y_br)

    def _regions_from_binary_mask(self, binary_mask: np.ndarray) -> List[Region]:
        """Calculate regions properties from a binary mask.

        Parameters
        ----------
        binary_mask : np.ndarray
            Binary mask from which to extract the regions

        Returns
        -------
        List[Region]
            Properties for all the regions present in the binary mask
        """

        thumb_labeled_regions = label(binary_mask)
        regions = [
            Region(index=i, area=rp.area, bbox=rp.bbox, center=rp.centroid)
            for i, rp in enumerate(regionprops(thumb_labeled_regions))
        ]
        return regions

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
        img, arr_img, large_w, large_h, new_w, new_h: tuple
        """
        # TODO: use logger instead of print(f"Opening Slide {slide_filepath}")

        large_w, large_h, new_w, new_h = self._resampled_dimensions(scale_factor)
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
        large_w, large_h = self.dimensions
        new_w = math.floor(large_w / scale_factor)
        new_h = math.floor(large_h / scale_factor)
        return large_w, large_h, new_w, new_h

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
        except openslide.OpenSlideError:
            raise openslide.OpenSlideError(
                "Your wsi has something broken inside, a doctor is needed"
            )
        except FileNotFoundError:
            raise FileNotFoundError("The wsi path resource doesn't exist")
        return slide

    @lazyproperty
    def _extension(self) -> str:
        return os.path.splitext(self._path)[1]


class SlideSet(object):
    """Slideset object. It is considered a collection of slides."""

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
        scale_factor: int, 32 by default
            Image scaling factor
        n: int,
            first n slides in dataset folder to rescale and save
        """
        os.makedirs(self._processed_path, exist_ok=True)
        n = self.total_slides if (n > self.total_slides or n == 0) else n
        for slide in self.slides[:n]:
            slide.save_scaled_image(scale_factor)

    def save_thumbnails(self, n: int = 0) -> None:
        """Save thumbnails

        Parameters
        ----------
        n: int
            first n slides in dataset folder
        scale_factor : int, 32 by default
            Image scaling factor
        """
        # TODO: add logger n>total_slide and log thumbnails names
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
    def slides_stats(self) -> Tuple[dict, matplotlib_figure]:
        """Retrieve statistic/graphs of slides files contained in the dataset

        Returns
        ----------
        basic_stats: dict of slides stats e.g. min_size, avg_size, etc...
        figure: matplotlib.figure.Figure
        """
        basic_stats = self._dimensions_stats

        x, y = zip(*self._slides_dimensions_list)
        colors = np.random.rand(self.total_slides)
        sizes = 8 * self.total_slides

        figure, axs = plt.subplots(ncols=2, nrows=2, constrained_layout=True)

        axs[0, 0].scatter(x, y, s=sizes, c=colors, alpha=0.7, cmap="prism")
        axs[0, 0].set_title("SVS Image Sizes (Labeled with slide name)")
        axs[0, 0].set_xlabel("width (pixels)")
        axs[0, 0].set_ylabel("height (pixels)")
        for i, s in enumerate(self.slides):
            axs[0, 1].annotate(s.name, (x[i], y[i]))

        area = [w * h / 1e6 for (w, h) in self._slides_dimensions_list]
        axs[0, 1].hist(area, bins=64)
        axs[0, 1].set_title("Distribution of image sizes in millions of pixels")
        axs[0, 1].set_xlabel("width x height (M of pixels)")
        axs[0, 1].set_ylabel("# images")

        whratio = [w / h for (w, h) in self._slides_dimensions_list]
        axs[1, 0].hist(whratio, bins=64)
        axs[1, 0].set_title("Image shapes (width to height)")
        axs[1, 0].set_xlabel("width to height ratio")
        axs[1, 0].set_ylabel("# images")

        hwratio = [h / w for (w, h) in self._slides_dimensions_list]
        axs[1, 1].hist(hwratio, bins=64)
        axs[1, 1].set_title("Image shapes (height to width)")
        axs[1, 1].set_xlabel("height to width ratio")
        axs[1, 1].set_ylabel("# images")

        return basic_stats, figure

    @lazyproperty
    def total_slides(self) -> int:
        """Number of slides within the slideset

        Returns
        ----------
        n: int, number of slides
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
    def _dimensions_stats(self) -> dict:
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
