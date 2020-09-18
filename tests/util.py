# encoding: utf-8

"""Utilities for histolab tests."""

import os

import numpy as np
from PIL import Image


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
