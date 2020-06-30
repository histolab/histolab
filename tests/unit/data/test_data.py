# coding: utf-8


import os
from unittest.mock import MagicMock, patch

import pytest

import openslide
from histolab.data import _has_hash, _load_svs, cmu_small_region, data_dir, file_hash

from ...fixtures import SVS
from ...unitutil import function_mock


def it_can_calculate_file_hash():
    with patch.dict("sys.modules", unknown=MagicMock()):
        file = SVS.CMU_1_SMALL_REGION

        fh = file_hash(file)

        assert fh == "ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7"


def but_it_raises_valueerror_when_algorithm_is_not_available():
    with patch(
        "pooch.utils.file_hash", side_effect=ModuleNotFoundError("mocked error")
    ):
        file = SVS.CMU_1_SMALL_REGION

        with pytest.raises(ValueError) as err:
            file_hash(file, alg="fake_alg")

        assert isinstance(err.value, ValueError)
        assert str(err.value) == "Algorithm 'fake_alg' not available in hashlib"


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
