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
from typing import Iterator, List, Tuple, Union

import numpy as np
import openslide
import PIL
from skimage.measure import find_contours

from .exceptions import LevelError
from .filters.compositions import FiltersComposition
from .masks import BinaryMask
from .tile import Tile
from .types import CoordinatePair
from .util import lazyproperty

IMG_EXT = "png"


class Slide:
    """Provide Slide objects and expose property and methods.

    Arguments
    ---------
    path : Union[str, pathlib.Path]
        Path where the WSI is saved.
    processed_path : Union[str, pathlib.Path]
        Path where thumbnails and scaled images will be saved to.
    """

    def __init__(
        self, path: Union[str, pathlib.Path], processed_path: Union[str, pathlib.Path]
    ) -> None:
        self._path = str(path) if isinstance(path, pathlib.Path) else path

        if processed_path is None:
            raise TypeError("processed_path cannot be None.")
        self._processed_path = processed_path

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(path={self._path}, processed_path={self._processed_path})"
        )

    # ---public interface methods and properties---

    @lazyproperty
    def dimensions(self) -> Tuple[int, int]:
        """Return the slide dimensions (w,h) at level 0.

        Returns
        -------
        dimensions : Tuple[int, int]
            Slide dimensions (width, height)
        """
        return self._wsi.dimensions

    def extract_tile(
        self, coords: CoordinatePair, level: int, tile_size: Tuple
    ) -> Tile:
        """Extract a tile of the image at the selected level.

        Parameters
        ----------
        coords : CoordinatePair
            Coordinates at level 0 from which to extract the tile.
        level : int
            Level from which to extract the tile.
        tile_size: tuple
            Final size of the tile (x,y).

        Returns
        -------
        tile : Tile
            Image containing the selected tile.
        """

        level = level if level >= 0 else self._remap_level(level)
        if not self._has_valid_coords(coords):
            # OpenSlide doesn't complain if the coordinates for extraction are wrong,
            # but it returns an odd image.
            raise ValueError(
                f"Extraction Coordinates {coords} not valid for slide with dimensions "
                f"{self.dimensions}"
            )

        image = self._wsi.read_region(
            location=(coords.x_ul, coords.y_ul), level=level, size=tile_size
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
        dimensions : Tuple[int, int]
            Slide dimensions at the specified level (width, height)
        """
        level = level if level >= 0 else self._remap_level(level)
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

    def locate_mask(
        self,
        binary_mask: BinaryMask,
        scale_factor: int = 32,
        tissue_mask: bool = False,
        alpha: int = 128,
        outline: str = "red",
    ) -> PIL.Image.Image:
        """Draw binary mask contours on a rescaled version of the slide

        Parameters
        ----------
        binary_mask : BinaryMask
            Binary Mask object
        scale_factor : int
            Scaling factor for the returned image. Default is 32.
        tissue_mask : bool, optional
            Whether to draw the contours on the binary tissue mask instead of
            the rescaled version of the slide. Default is False.
        alpha : int
            The alpha level to be applied to the rescaled slide, default to 128.
        outline : str
            The outline color for the annotation, default to 'red'.

        Returns
        -------
        PIL.Image.Image
            PIL Image of the rescaled slide with the binary mask contours outlined.
        """
        img = self.scaled_image(scale_factor)
        mask = binary_mask(self)
        resized_mask = np.array(
            PIL.Image.fromarray(mask).resize(img.size, PIL.Image.ANTIALIAS)
        )

        if tissue_mask:
            filters = FiltersComposition(Slide).tissue_mask_filters
            img_tissue_mask = filters(img)
            img = PIL.Image.fromarray(img_tissue_mask).convert("RGB")
        else:
            img.putalpha(alpha)

        # pad the mask to have closed contours along the edges
        padded_mask = np.pad(resized_mask, pad_width=1, mode="constant")
        contours = [
            cont - 1 for cont in find_contours(padded_mask, 0.5)
        ]  # unpad countours

        for contour in contours:
            contour = np.ceil(contour)
            contour = np.vstack([contour[:, 1], contour[:, 0]]).T
            PIL.ImageDraw.Draw(img).polygon(contour.ravel().tolist(), outline=outline)

        return img

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

    @lazyproperty
    def properties(self) -> dict:
        """Whole Slide Image properties.

        Returns
        -------
        dict
            WSI complete properties.
        """
        return dict(self._wsi.properties)

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

    def scaled_image(self, scale_factor: int = 32) -> PIL.Image.Image:
        """Return a scaled image of the slide.

        Parameters
        ----------
        scale_factor : int, optional
            Image scaling factor. Default is 32.

        Returns
        -------
        PIL.Image.Image
            A scaled image of the slide.
        """
        return self._resample(scale_factor)[0]

    def show(self) -> None:
        """Display the slide thumbnail.

        NOTE: A new window of your OS image viewer will be opened.
        """
        try:
            thumbnail = self.thumbnail
            thumbnail.show()  # pragma: no cover
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Cannot display the slide thumbnail: {error}")

    @lazyproperty
    def thumbnail(self) -> PIL.Image.Image:
        """Return the slide thumbnail.

        Returns
        -------
        PIL.Image.Image
            The slide thumbnail.
        """
        return self._wsi.get_thumbnail(self._thumbnail_size)

    # ------- implementation helpers -------

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

    def _remap_level(self, level: int) -> int:
        """Remap negative index for the given level onto a positive one.

        Parameters
        ----------
        level : int
            The level index to remap

        Returns
        -------
        level : int
           Positive level index

        Raises
        ------
        LevelError
            when the abs(level) is greater than the number of the levels.
        """
        if len(self.levels) - abs(level) < 0:
            raise LevelError(
                f"Level {level} not available. Number of available levels: "
                f"{len(self._wsi.level_dimensions)}"
            )
        return len(self.levels) - abs(level)

    def _resample(self, scale_factor: int = 32) -> Tuple[PIL.Image.Image, np.array]:
        """Convert a slide to a scaled-down PIL image.

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
        PIL.Image.Image
            The resampled image
        np.ndarray
            The resampled image converted to array
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
        Tuple[int, int, int, int]
            Original slide dimensions and scaled dimensions (original w, original h,
            resampled w, resampled h).
        """
        large_w, large_h = self.dimensions
        new_w = math.floor(large_w / scale_factor)
        new_h = math.floor(large_h / scale_factor)
        return large_w, large_h, new_w, new_h

    @lazyproperty
    def _thumbnail_size(self) -> Tuple[int, int]:
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
        Tuple[int, int]
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
            raise FileNotFoundError(
                f"The wsi path resource doesn't exist: {self._path}"
            )
        return slide


class SlideSet:
    """Slideset object. It is considered a collection of Slides."""

    def __init__(
        self, slides_path: str, processed_path: str, valid_extensions: list
    ) -> None:
        self._slides_path = slides_path
        self._processed_path = processed_path
        self._valid_extensions = valid_extensions

    def __iter__(self) -> Iterator[Slide]:
        """Slides of the slideset

        Returns
        -------
        generator of `Slide` objects.
        """
        return iter(
            [
                Slide(os.path.join(self._slides_path, _path), self._processed_path)
                for _path in os.listdir(self._slides_path)
                if os.path.splitext(_path)[1] in self._valid_extensions
            ]
        )

    def __getitem__(self, slide_id: int) -> Slide:
        """Slide object given the correspondent id"""
        return list(self.__iter__())[slide_id]

    def __len__(self) -> int:
        """Total number of the slides of this Slideset

        Returns
        -------
        int
            number of the Slides.
        """
        return len(list(self.__iter__()))

    # ---public interface methods and properties---

    def scaled_images(
        self, scale_factor: int = 32, n: int = 0
    ) -> List[PIL.Image.Image]:
        """Return rescaled images of the slides.

        Parameters
        ----------
        scale_factor : int, optional
            Image scaling factor. Default is 32.
        n : int, optional
            First n slides in dataset folder to rescale. Default is 0, meaning that all
            the slides will be returned.

        Returns
        -------
        List[PIL.Image.Image]
            List of rescaled images of the slides.
        """
        n = self.total_slides if (n > self.total_slides or n == 0) else n
        rescaled_imgs = []
        for slide in list(self.__iter__())[:n]:
            rescaled_imgs.append(slide.scaled_image(scale_factor))
        return rescaled_imgs

    def thumbnails(self, n: int = 0) -> List[PIL.Image.Image]:
        """Return slides thumbnails

        Parameters
        ----------
        n : int, optional
            First n slides in dataset folder. Default is 0, meaning that the thumbnails
            of all the slides will be returned.

        Returns
        -------
        List[PIL.Image.Image]
            List of slides thumbnails
        """
        n = self.total_slides if (n > self.total_slides or n == 0) else n
        thumbnails = []
        for slide in list(self.__iter__())[:n]:
            thumbnails.append(slide.thumbnail)
        return thumbnails

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
        return self.__len__()

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
            for slide in list(self.__iter__())
        ]

    @lazyproperty
    def _slides_dimensions_list(self):
        return [slide.dimensions for slide in list(self.__iter__())]
