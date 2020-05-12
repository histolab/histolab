# encoding: utf-8

"""Utilities for histolab tests."""

import os
import numpy as np


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
    elif type_ == "txt":
        with open(expectation_file_path, "rb") as f:
            expectation_byte = f.read()
        expectation_data = expectation_byte.decode("utf-8")
    else:
        raise Exception("Type format not recognized")
    return expectation_data
