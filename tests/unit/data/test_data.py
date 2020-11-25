# coding: utf-8

import copy
import os
import sys
from importlib import reload
from unittest.mock import patch

import openslide
import PIL
import pytest
import numpy as np
from requests.exceptions import HTTPError

from histolab import __version__ as v
from histolab.data import (
    _fetch,
    _has_hash,
    _load_svs,
    _registry,
    cmu_small_region,
    data_dir,
    registry,
)

from ...fixtures import SVS
from ...unitutil import ANY, fetch, function_mock


def test_data_dir():
    # data_dir should be a directory that can be used as a standard directory
    data_directory = data_dir
    assert "cmu_small_region.svs" in os.listdir(data_directory)


def test_download_file_from_internet():
    # NOTE: This test will be skipped when internet connection is not available
    fetch("histolab/kidney.png")
    kidney_path = _fetch("histolab/kidney.png")
    kidney_image = np.array(PIL.Image.open(kidney_path))

    assert kidney_image.shape == (537, 809, 4)


def test_download_file_from_internet_but_it_is_broken():
    # NOTE: This test will be skipped when internet connection is not available
    fetch("histolab/broken.svs")
    with pytest.raises(PIL.UnidentifiedImageError) as err:
        _load_svs("histolab/broken.svs")

    assert str(err.value) == "Your wsi has something broken inside, a doctor is needed"


def test_cmu_small_region():
    """ Test that "cmu_small_region" svs can be loaded. """
    cmu_small_region_image, path = cmu_small_region()
    assert cmu_small_region_image.dimensions == (2220, 2967)


@patch.dict(registry, {"data/cmu_small_region_broken.svs": "bar"}, clear=True)
@patch.object(_registry, "legacy_datasets", ["data/cmu_small_region_broken.svs"])
def test_file_url_not_found():
    data_filename = "data/cmu_small_region_broken.svs"
    with pytest.raises(HTTPError) as err:
        _fetch(data_filename)

    assert (
        str(err.value) == f"404 Client Error: Not Found for url: "
        f"https://github.com/histolab/histolab/raw/{v}/histolab/{data_filename}"
    )


def test_load_svs(request):
    file = SVS.CMU_1_SMALL_REGION
    _fetch = function_mock(request, "histolab.data._fetch")
    _fetch.return_value = file

    svs, path = _load_svs(file)

    assert type(svs) == openslide.OpenSlide
    assert path == file


@pytest.mark.parametrize(
    "file, hash, expected_value",
    ((SVS.CMU_1_SMALL_REGION, "1234abcd", True), ("/fake/file", "1234abcd", False)),
)
def it_knows_its_hash(request, file, hash, expected_value):
    file = file
    file_hash_ = function_mock(request, "histolab.data.file_hash")
    file_hash_.return_value = hash

    has_hash = _has_hash(file, hash)

    assert has_hash is expected_value


@patch.dict(registry, {"data/cmu_small_region.svs": "bar"}, clear=True)
@patch("histolab.data.image_fetcher", None)
def it_raises_error_on_fetch_if_image_fetcher_is_None():
    with pytest.raises(ModuleNotFoundError) as err:
        _fetch("data/cmu_small_region.svs")

    assert (
        str(err.value)
        == "The requested file is part of the histolab distribution, but requires the "
        "installation of an optional dependency, pooch. To install pooch, use your "
        "preferred python package manager. Follow installation instruction found at "
        "https://www.fatiando.org/pooch/latest/install.html"
    )


def test_pooch_missing(monkeypatch):
    from histolab import data

    fakesysmodules = copy.copy(sys.modules)
    fakesysmodules["pooch.utils"] = None
    monkeypatch.delitem(sys.modules, "pooch.utils")
    monkeypatch.setattr("sys.modules", fakesysmodules)
    file = SVS.CMU_1_SMALL_REGION
    reload(data)

    data.file_hash(file)

    assert data.file_hash.__module__ == "histolab.data"


def test_file_hash_with_wrong_algorithm(monkeypatch):
    from histolab import data

    fakesysmodules = copy.copy(sys.modules)
    fakesysmodules["pooch.utils"] = None
    monkeypatch.delitem(sys.modules, "pooch.utils")
    monkeypatch.setattr("sys.modules", fakesysmodules)
    file = SVS.CMU_1_SMALL_REGION
    reload(data)

    with pytest.raises(ValueError) as err:
        data.file_hash(file, "fakesha")
    assert str(err.value) == "Algorithm 'fakesha' not available in hashlib"
    assert data.file_hash.__module__ == "histolab.data"


def test_create_image_fetcher_without_pooch(monkeypatch):
    from histolab import data

    fakesysmodules = copy.copy(sys.modules)
    fakesysmodules["pooch"] = None
    monkeypatch.delitem(sys.modules, "pooch")
    monkeypatch.setattr("sys.modules", fakesysmodules)
    reload(data)

    create_image_fetcher = data._create_image_fetcher()

    assert create_image_fetcher == (None, ANY)
