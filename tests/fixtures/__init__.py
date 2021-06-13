# encoding: utf-8

"""SVS source files loader for testing purposes."""

import os

import numpy as np
from PIL import Image

from ..unitutil import on_ci


class LazyResponder:
    """Loads and caches fixtures files by name from fixture directory.
    Provides access to all the svs fixtures in a directory by
    a standardized mapping of the file name, e.g. ca-1-c.svs is available
    as the `.CA_1_C` attribute of the loader.

    The fixture directory is specified relative to this (fixture root)
    directory.
    """

    def __init__(self, relpath):
        self._relpath = relpath
        self._cache = {}

    def __getattr__(self, fixture_name):
        if fixture_name not in self._cache:
            self._load_to_cache(fixture_name)
        return self._cache[fixture_name]

    @property
    def _dirpath(self):
        thisdir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(thisdir, self._relpath))


class LazySVSResponseLoader(LazyResponder):
    """Specific class for SVS fixtures loader"""

    slide_format = "SVS"

    def _slide_path(self, fixture_name):
        return "%s/%s.svs" % (self._dirpath, fixture_name.replace("_", "-").lower())

    def _load_slide(self, path):
        with open(path, "rb") as f:
            svs_image = f.name
        return svs_image

    def _load_to_cache(self, fixture_name):
        slide_path = self._slide_path(fixture_name)
        if not os.path.exists(slide_path):
            raise ValueError(f"no {self.slide_format} fixture found at {slide_path}")
        self._cache[fixture_name] = self._load_slide(slide_path)


class LazyExternalSVSResponseLoader(LazySVSResponseLoader):
    def _load_to_cache(self, fixture_name):
        slide_path = self._slide_path(fixture_name)
        if not os.path.exists(slide_path) and on_ci():
            raise ValueError(f"no {self.slide_format} fixture found at {slide_path}")
        self._cache[fixture_name] = slide_path


class LazyTIFFResponseLoader(LazySVSResponseLoader):
    """Specific class for TIFF fixtures loader"""

    slide_format = "TIFF"

    def _slide_path(self, fixture_name):
        return "%s/%s.tif" % (self._dirpath, fixture_name.replace("_", "-").lower())


class LazyPILResponseLoader(LazyResponder):
    """Specific class for PIL fixtures loader"""

    def _pil_path(self, fixture_name):
        return "%s/%s.png" % (self._dirpath, fixture_name.replace("_", "-").lower())

    def _load_pil(self, path):
        return Image.open(path)

    def _load_to_cache(self, fixture_name):
        pil_path = self._pil_path(fixture_name)
        if not os.path.exists(pil_path):
            raise ValueError("no PIL fixture found at %s" % pil_path)
        self._cache[fixture_name] = self._load_pil(pil_path)


class LazyNPYResponseLoader(LazyResponder):
    """Specific class for PIL fixtures loader"""

    def _npy_path(self, fixture_name):
        return "%s/%s.npy" % (self._dirpath, fixture_name.replace("_", "-").lower())

    def _load_npy(self, path):
        return np.load(path)

    def _load_to_cache(self, fixture_name):
        npy_path = self._npy_path(fixture_name)
        if not os.path.exists(npy_path):
            raise ValueError("no NPY fixture found at %s" % npy_path)
        self._cache[fixture_name] = self._load_npy(npy_path)


SVS = LazySVSResponseLoader("./svs-images")
EXTERNAL_SVS = LazyExternalSVSResponseLoader("./external-svs")
TIFF = LazyTIFFResponseLoader("./tiff-images")
RGBA = LazyPILResponseLoader("./pil-images-rgba")
RGB = LazyPILResponseLoader("./pil-images-rgb")
GS = LazyPILResponseLoader("./pil-images-gs")
MASKNPY = LazyNPYResponseLoader("./mask-arrays")
NPY = LazyNPYResponseLoader("./arrays")
TILES = LazyPILResponseLoader("./tiles")
