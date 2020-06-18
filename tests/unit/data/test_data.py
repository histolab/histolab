# coding: utf-8

import os

import histolab.data as data


def test_data_dir():
    # data_dir should be a directory that can be used as a standard directory
    data_dir = data.data_dir
    assert "cmu_small_region.svs" in os.listdir(data_dir)


def test_cmu_small_region():
    """ Test that "cmu_small_region" svs can be loaded. """
    cmu_small_region, path = data.cmu_small_region()
    assert cmu_small_region.dimensions == (2220, 2967)
