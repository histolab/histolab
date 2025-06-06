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

"""Utilities for histolab tests."""

import base64
import os
from io import BytesIO

import numpy as np
from PIL import Image


def pil_to_base64(pilimage):  # pragma: no cover
    """Returns base64 encoded image given a PIL Image"""
    buffer = BytesIO()
    pilimage.save(buffer, "png")
    return base64.b64encode(buffer.getvalue()).decode()


def expand_tests_report(request, **kwargs):  # pragma: no cover
    """Augment request with key value args that will be passed to pytest markreport."""
    setattr(request.node, "extra_args", kwargs)


def load_expectation(expectation_file_name, type_=None):  # pragma: no cover
    """Returns np.ndarray related to the *expectation_file_name*.

    Expectation file path is rooted at tests/expectations.
    """
    thisdir = os.path.dirname(__file__)
    expectation_file_path = os.path.abspath(
        os.path.join(thisdir, "expectations", f"{expectation_file_name}.{type_}")
    )
    if type_ == "npy":
        expectation_data = np.load(expectation_file_path)
    elif type_ == "png":
        expectation_data = Image.open(expectation_file_path)
    else:
        raise Exception("Type format not recognized")
    return expectation_data


def load_python_expression(expression_file_name):  # pragma: no cover
    """Return a Python object (list, dict) formed by parsing `expression_file_name`.

    Expectation file path is rooted at tests/expectations.
    """
    thisdir = os.path.dirname(__file__)
    expression_file_path = os.path.abspath(
        os.path.join(thisdir, "expectations", "%s.py" % expression_file_name)
    )
    with open(expression_file_path) as f:
        expression_bytes = f.read()
    return eval(expression_bytes)
