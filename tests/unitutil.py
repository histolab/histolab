# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2022 All Histolab Contributors
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

"""Functions that make mocking with pytest easier and more readable."""

import os

import numpy as np
import pytest
from PIL import Image

from histolab import data
from histolab.slide import Slide

from unittest.mock import ANY, call  # noqa # isort:skip
from unittest.mock import create_autospec, patch, PropertyMock  # isort:skip


def base_test_slide(tmpdir, image, use_largeimage=False):
    tmp_path_ = tmpdir.mkdir("myslide")
    image.save(os.path.join(tmp_path_, "mywsi.png"), "PNG")
    slide_path = os.path.join(tmp_path_, "mywsi.png")
    slide = Slide(slide_path, "processed", use_largeimage=use_largeimage)
    return slide, tmp_path_


def dict_list_eq(l1, l2):
    sorted_l1 = sorted(sorted(d.items()) for d in l1)
    sorted_l2 = sorted(sorted(d.items()) for d in l2)
    return sorted_l1 == sorted_l2


def class_mock(request, q_class_name, autospec=True, **kwargs):
    """Return mock patching class with qualified name *q_class_name*.

    The mock is autospec'ed based on the patched class unless the optional
    argument *autospec* is set to False. Any other keyword arguments are
    passed through to Mock(). Patch is reversed after calling test returns.
    """
    _patch = patch(q_class_name, autospec=autospec, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def function_mock(request, q_function_name, autospec=True, **kwargs):
    """Return mock patching function with qualified name *q_function_name*.

    Patch is reversed after calling test returns.
    """
    _patch = patch(q_function_name, autospec=autospec, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def initializer_mock(request, cls, autospec=True, **kwargs):
    """Return mock for __init__() method on *cls*.

    The patch is reversed after pytest uses it.
    """
    _patch = patch.object(
        cls, "__init__", autospec=autospec, return_value=None, **kwargs
    )
    request.addfinalizer(_patch.stop)
    return _patch.start()


def instance_mock(request, cls, name=None, spec_set=True, **kwargs):
    """Return mock for instance of *cls* that draws its spec from the class.

    The mock will not allow new attributes to be set on the instance. If
    *name* is missing or |None|, the name of the returned |Mock| instance is
    set to *request.fixturename*. Additional keyword arguments are passed
    through to the Mock() call that creates the mock.
    """
    name = name if name is not None else request.fixturename
    return create_autospec(cls, _name=name, spec_set=spec_set, instance=True, **kwargs)


def method_mock(request, cls, method_name, autospec=True, **kwargs):
    """Return mock for method *method_name* on *cls*.

    The patch is reversed after pytest uses it.
    """
    _patch = patch.object(cls, method_name, autospec=autospec, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def property_mock(request, cls, prop_name, **kwargs):
    """Return mock for property *prop_name* on class *cls*.

    The patch is reversed after pytest uses it.
    """
    _patch = patch.object(cls, prop_name, new_callable=PropertyMock, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def fetch(data_filename):
    """Attempt to fetch data, but if unavailable, skip the tests."""
    try:
        return data._fetch(data_filename)
    except (ConnectionError, ModuleNotFoundError):
        pytest.skip(f"Unable to download {data_filename}")


def on_ci():
    # GitHub Actions, Travis and AppVeyor have "CI"
    return "CI" in os.environ


class PILImageMock:
    np.random.seed(0)
    RGBA_COLOR_500X500_155_249_240 = Image.new(
        "RGBA", size=(500, 500), color=(155, 249, 240)
    )

    RGBA_COLOR_50X50_155_0_0 = Image.new("RGBA", size=(50, 50), color=(155, 0, 0))

    RGBA_COLOR_49X51_155_0_0 = Image.new("RGBA", size=(49, 51), color=(155, 0, 0))

    RGB_RANDOM_COLOR_500X500 = Image.fromarray(
        (np.random.rand(500, 500, 3) * 255).astype("uint8")
    ).convert("RGB")

    RGBA_RANDOM_COLOR_500X500 = RGB_RANDOM_COLOR_500X500.convert("RGBA")

    RGB_RANDOM_COLOR_10X10 = Image.fromarray(
        (np.random.rand(10, 10, 3) * 255).astype("uint8")
    ).convert("RGB")

    GRAY_RANDOM_10X10 = Image.fromarray((np.random.rand(10, 10) * 255).astype("uint8"))


class NpArrayMock:
    ONES_30X30_UINT8 = np.ones([30, 30], dtype="uint8")
    ONES_500X500X4_BOOL = np.ones([500, 500, 4], dtype="bool")
    ONES_500X500_BOOL = np.ones([500, 500], dtype="bool")
    RANDOM_500X500_BOOL = np.random.rand(500, 500) > 0.5


PILIMG = PILImageMock
