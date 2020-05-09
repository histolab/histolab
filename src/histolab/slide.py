# encoding: utf-8

# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

"""Provides the Slide class.

Slide is the main API class for manipulating WSI slides images.
"""

import math
import ntpath
import numpy as np
import openslide
import os
import PIL
import pathlib
import matplotlib.pyplot as plt

from typing import Tuple, Union
from .util import Time

IMG_EXT = "png"
THUMBNAIL_SIZE = 300


class Slide(object):
    """Provides Slide objects and expose property and methods.

    HERE-> expand the docstring
    """

    def __init__(
        self, wsi_path: str, processed_path: str, scale_factor: int = 32
    ) -> None:
        self._wsi_path = wsi_path
        self._processed_path = processed_path
        self._scale_factor = scale_factor

    # ---public interface methods and properties---

    @property
    def resampled_array(self) -> np.array:
        return self._resample[1]

    def save_scaled_image(self) -> None:
        """Save a scaled image in the correct path"""
        os.makedirs(self._processed_path, exist_ok=True)
        img = self._resample[0]
        img.save(self.scaled_image_path)

    def save_thumbnail(self) -> None:
        """Save a thumbnail in the correct path"""
        os.makedirs(self._processed_path, exist_ok=True)
        img = self._resample[0]
        max_size = tuple(
            round(THUMBNAIL_SIZE * size / max(img.size)) for size in img.size
        )
        resized_img = img.resize(max_size, PIL.Image.BILINEAR)
        folder = os.path.dirname(self.thumbnail_path)
        pathlib.Path(folder).mkdir(exist_ok=True)
        resized_img.save(self.thumbnail_path)

    @property
    def scaled_image_path(self) -> str:
        """Returns slide image path.

        Returns
        -------
        img_path : str
        """
        img_path = self._breadcumb(self._processed_path)
        return img_path

    @property
    def thumbnail_path(self) -> str:
        """Returns thumbnail image path.

        Returns
        -------
        thumb_path : str
        """
        thumb_path = self._breadcumb(
            os.path.join(self._processed_path, f"thumbnails_{IMG_EXT}")
        )
        return thumb_path

    @property
    def wsi_dimensions(self) -> Tuple[int, int]:
        """Returns the wsi dimensions (w,h)

        Returns
        -------
        wsi_dimensions : tuple(width, height)
        """
        return self._wsi.dimensions

    @property
    def wsi_name(self) -> str:
        """Retrieves the WSI name without extension.

        Returns
        -------
        wsi_name : str
        """
        return ntpath.basename(self._wsi_path).split(".")[0]

    # ---private interface methods and properties---

    def _breadcumb(self, directory_path) -> str:
        """Returns a complete path according to the give directory path

        Parameters
        ----------
        directory_path: str

        Returns
        -------
        final_path: str, a real and complete path starting from the dir path
                    e.g. /processed_path/my-image-name-x32/
                         /thumb_path/my-image-name-x32/thumbs
        """
        large_w, large_h, new_w, new_h = self._resampled_dimensions
        if {large_w, large_h, new_w, new_h} == {None}:
            final_path = os.path.join(directory_path, f"{self.wsi_name}*.{IMG_EXT}")
        else:
            final_path = os.path.join(
                directory_path,
                f"{self.wsi_name}-{self._scale_factor}x-{large_w}x{large_h}-{new_w}x"
                f"{new_h}.{IMG_EXT}",
            )
        return final_path

    @property
    def _resample(self) -> Tuple[PIL.Image.Image, np.array]:
        """Converts a WSI slide to a scaled-down PIL image.

        The PIL image is also converted to array.
        image is the scaled-down PIL image, original width and original height
        are the width and height of the WSI, new width and new height are the
        dimensions of the PIL image.

        Returns
        -------
        img, arr_img, large_w, large_h, new_w, new_h: tuple
        """
        # TODO: use logger instead of print(f"Opening Slide {slide_filepath}")

        large_w, large_h, new_w, new_h = self._resampled_dimensions
        level = self._wsi.get_best_level_for_downsample(self._scale_factor)
        whole_slide_image = self._wsi.read_region(
            (0, 0), level, self._wsi.level_dimensions[level]
        )
        # ---converts openslide read_region to an actual RGBA image---
        whole_slide_image = whole_slide_image.convert("RGB")
        img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
        arr_img = np.asarray(img)
        return img, arr_img

    @property
    def _resampled_dimensions(self) -> Tuple[int, int, int, int]:
        large_w, large_h = self.wsi_dimensions
        new_w = math.floor(large_w / self._scale_factor)
        new_h = math.floor(large_h / self._scale_factor)
        return large_w, large_h, new_w, new_h

    @property
    def _wsi(self) -> Union[openslide.OpenSlide, openslide.ImageSlide]:
        """Open the slide and returns an openslide object

        Returns
        -------
        slide : OpenSlide object
                An OpenSlide object representing a whole-slide image.
        """
        try:
            slide = openslide.open_slide(self._wsi_path)
        except openslide.OpenSlideError:
            raise openslide.OpenSlideError(
                "Your wsi has something broken inside, a doctor is needed"
            )
        except FileNotFoundError:
            raise FileNotFoundError("The wsi path resource doesn't exist")
        return slide

    @property
    def _wsi_extension(self) -> str:
        return os.path.splitext(self._wsi_path)[1]


class SlideSet(object):
    def __init__(self, slides_path, processed_path, valid_wsi_extensions):
        self._slides_path = slides_path
        self._processed_path = processed_path
        self._valid_wsi_extensions = valid_wsi_extensions

    def save_rescaled_slides(self, n=0):
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
            slide.save_scaled_image()

    def save_thumbnails(self, n=0):
        """Save thumbnails

        Parameters
        ----------
        n: int
            first n slides in dataset folder
        """
        # TODO: add logger n>total_slide and log thumbnails names
        os.makedirs(self._processed_path, exist_ok=True)
        n = self.total_slides if (n > self.total_slides or n == 0) else n
        for slide in self.slides[:n]:
            slide.save_thumbnail()

    @property
    def slides(self):
        return [
            Slide(os.path.join(self._slides_path, wsi_path), self._processed_path)
            for wsi_path in os.listdir(self._slides_path)
            if os.path.splitext(wsi_path)[1] in self._valid_wsi_extensions
        ]

    @property
    def slides_dimensions(self):
        return [
            {
                "wsi": slide.wsi_name,
                "width": slide.wsi_dimensions[0],
                "height": slide.wsi_dimensions[1],
                "size": slide.wsi_dimensions[0] * slide.wsi_dimensions[1],
            }
            for slide in self.slides
        ]

    @property
    def slides_stats(self):
        """Retrieve statistic/graphs of wsi files contained in the dataset"""
        t = Time()

        basic_stats = {
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

        t.elapsed_display()

        x, y = zip(*[slide.wsi_dimensions for slide in self.slides])
        colors = np.random.rand(self.total_slides)
        sizes = 8 * self.total_slides

        plt.ioff()

        fig, ax = plt.subplots()
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.7)
        plt.xlabel("width (pixels)")
        plt.ylabel("height (pixels)")
        plt.title("WSI sizes")
        plt.set_cmap("prism")

        # plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
        # plt.xlabel("width (pixels)")
        # plt.ylabel("height (pixels)")
        # plt.title("SVS Image Sizes (Labeled with slide numbers)")
        # plt.set_cmap("prism")
        # for i in range(num_images):
        #     snum = i + 1
        #     plt.annotate(str(snum), (x[i], y[i]))
        # plt.tight_layout()

        # area = [w * h / 1e6 for (w, h) in slide_stats]
        # plt.hist(area, bins=64)
        # plt.xlabel("width x height (M of pixels)")
        # plt.ylabel("# images")
        # plt.title("Distribution of image sizes in millions of pixels")
        # plt.tight_layout()

        # whratio = [w / h for (w, h) in slide_stats]
        # plt.hist(whratio, bins=64)
        # plt.xlabel("width to height ratio")
        # plt.ylabel("# images")
        # plt.title("Image shapes (width to height)")
        # plt.tight_layout()

        # hwratio = [h / w for (w, h) in slide_stats]
        # plt.hist(hwratio, bins=64)
        # plt.xlabel("height to width ratio")
        # plt.ylabel("# images")
        # plt.title("Image shapes (height to width)")
        # plt.tight_layout()
        # t.elapsed_display()
        return basic_stats, fig

    @property
    def total_slides(self):
        return len(self.slides)

    @property
    def _avg_width_slide(self):
        return sum(d["width"] for d in self.slides_dimensions) / self.total_slides

    @property
    def _avg_height_slide(self):
        return sum(d["height"] for d in self.slides_dimensions) / self.total_slides

    @property
    def _avg_size_slide(self):
        return sum(d["size"] for d in self.slides_dimensions) / self.total_slides

    @property
    def _max_height_slide(self):
        max_height = max(self.slides_dimensions, key=lambda x: x["height"])
        return {"slide": max_height["wsi"], "height": max_height["height"]}

    @property
    def _max_size_slide(self):
        max_size = max(self.slides_dimensions, key=lambda x: x["size"])
        return {"slide": max_size["wsi"], "size": max_size["size"]}

    @property
    def _max_width_slide(self):
        max_width = max(self.slides_dimensions, key=lambda x: x["width"])
        return {"slide": max_width["wsi"], "width": max_width["width"]}

    @property
    def _min_width_slide(self):
        min_width = min(self.slides_dimensions, key=lambda x: x["width"])
        return {"slide": min_width["wsi"], "width": min_width["width"]}

    @property
    def _min_height_slide(self):
        min_height = min(self.slides_dimensions, key=lambda x: x["height"])
        return {"slide": min_height["wsi"], "height": min_height["height"]}

    @property
    def _min_size_slide(self):
        min_size = min(self.slides_dimensions, key=lambda x: x["size"])
        return {"slide": min_size["wsi"], "height": min_size["size"]}
