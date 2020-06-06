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
from collections import namedtuple
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import ntpath
import numpy as np
import openslide
import PIL
from matplotlib.figure import Figure as matplotlib_figure
from skimage.measure import label, regionprops

import src.histolab.filters.image_filters as imf
import src.histolab.filters.morphological_filters as mof

from .util import CoordinatePair, lazyproperty, polygon_to_mask_array, resize_mask

IMG_EXT = "png"
THUMBNAIL_SIZE = 1000

# needed for matplotlib
# TODO: can we get rid of this shit?
plt.ioff()


class Slide(object):
    """Provides Slide objects and expose property and methods.

    HERE-> expand the docstring
    """

    def __init__(self, path: str, processed_path: str) -> None:
        self._path = path
        self._processed_path = processed_path

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
    def mask_biggest_tissue_box(self):
        """Returns the coordinates of the box containing the max area of tissue.

        Returns
        -------
        box_coords: Coordinates
            [x_ul, y_ul, x_br, y_br] coordinates of the box containing the
            max area of tissue.

        """
        Region = namedtuple("Region", ("index", "area", "bbox", "center"))

        thumb = self._wsi.get_thumbnail((1000, 1000))
        filters = imf.Compose(
            [
                imf.RgbToGrayscale(),
                imf.OtsuThreshold(),
                mof.BinaryDilation(),
                mof.RemoveSmallHoles(),
                mof.RemoveSmallObjects(),
            ]
        )

        thumb_mask = filters(thumb)
        thumb_labeled_regions = label(thumb_mask)
        labeled_region_properties = regionprops(thumb_labeled_regions)

        regions = [
            Region(index=i, area=rp.area, bbox=rp.bbox, center=rp.centroid)
            for i, rp in enumerate(labeled_region_properties)
        ]
        biggest_region = max(regions, key=lambda r: r.area)
        y_ul, x_ul, y_br, x_br = biggest_region.bbox

        thumb_bbox_coords = CoordinatePair(x_ul, y_ul, x_br, y_br)
        thumb_bbox_mask = polygon_to_mask_array((1000, 1000), thumb_bbox_coords)
        return resize_mask(thumb_bbox_mask, self.dimensions)

    @lazyproperty
    def name(self) -> str:
        """Retrieves the slide name without extension.

        Returns
        -------
        name : str
        """
        return ntpath.basename(self._path).split(".")[0]

    def resampled_array(self, scale_factor: int = 32) -> np.array:
        return self._resample(scale_factor)[1]

    def save_scaled_image(self, scale_factor: int = 32) -> None:
        """Save a scaled image in the correct path

        Parameters
        ----------
        scale_factor : int, default is 32
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
        scale_factor : int, default is 32
            Image scaling factor

        Returns
        -------
        img_path : str
        """
        img_path = self._breadcumb(self._processed_path, scale_factor)
        return img_path

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

    def _breadcumb(self, directory_path: str, scale_factor: int = 32) -> str:
        """Returns a complete path according to the give directory path

        Parameters
        ----------
        directory_path: str
        scale_factor : int, default is 32
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

    def _resample(self, scale_factor: int = 32) -> Tuple[PIL.Image.Image, np.array]:
        """Converts a slide to a scaled-down PIL image.

        The PIL image is also converted to array.
        image is the scaled-down PIL image, original width and original height
        are the width and height of the slide, new width and new height are the
        dimensions of the PIL image.

        Parameters
        ----------
        scale_factor : int, default is 32
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
        n: int
           first n slides in dataset folder to rescale and save
        """
        # TODO: add logger if n>total_slide and log saved images names
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
        scale_factor : int, default is 32
            Image scaling factor
        """
        # TODO: add logger n>total_slide and log thumbnails names
        os.makedirs(self._processed_path, exist_ok=True)
        n = self.total_slides if (n > self.total_slides or n == 0) else n
        for slide in self.slides[:n]:
            slide.save_thumbnail()

    @lazyproperty
    def slides(self) -> List[Slide]:
        return [
            Slide(os.path.join(self._slides_path, _path), self._processed_path)
            for _path in os.listdir(self._slides_path)
            if os.path.splitext(_path)[1] in self._valid_extensions
        ]

    @lazyproperty
    def slides_stats(self) -> Tuple[dict, matplotlib_figure]:
        """Retrieve statistic/graphs of slides files contained in the dataset"""

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
