# coding: utf-8

import copy

import os
import sys
from unittest.mock import patch
from importlib import reload

import pytest

import openslide
from histolab.data import (
    _fetch,
    _has_hash,
    _load_svs,
    cmu_small_region,
    data_dir,
    registry,
)

from ...fixtures import SVS
from ...unitutil import function_mock, ANY


def test_data_dir():
    # data_dir should be a directory that can be used as a standard directory
    data_directory = data_dir
    assert "cmu_small_region.svs" in os.listdir(data_directory)


def test_cmu_small_region():
    """ Test that "cmu_small_region" svs can be loaded. """
    cmu_small_region_image, path = cmu_small_region()
    assert cmu_small_region_image.dimensions == (2220, 2967)


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
