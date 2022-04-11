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

"""Utilities for pytest reports."""

import PIL
import pytest

from .util import pil_to_base64


class HTMLRenderer:
    """HTML object for rendering header and body on the pytest html report."""

    def __init__(self, items):
        self._items = items

    @property
    def body(self):
        """Table body for the html report."""
        return "".join([f"<td>{self._html_tag(item)}</td>" for item in self._values])

    @property
    def head(self):
        """Table header for the html report."""
        return "".join([f"<th>{item.title()}</th>" for item in self._keys])

    @property
    def _keys(self):
        return self._items.keys() if self._items else []

    @staticmethod
    def _html_tag(item):
        """Convert the given item in a proper html tag."""
        if isinstance(item, PIL.Image.Image):
            return f"<img src='data:image/png;base64,{pil_to_base64(item)}'>"
        return item

    @property
    def _values(self):
        return self._items.values() if self._items else []


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item):
    pytest_html = item.config.pluginmanager.getplugin("html")
    outcome = yield
    report = outcome.get_result()
    extra = getattr(report, "extra", [])
    if report.when == "call" and report.failed:
        extra_args = getattr(item, "extra_args", None)
        html = HTMLRenderer(extra_args)
        if extra_args:
            extra.append(
                pytest_html.extras.html(
                    f"""
                    <table width=100%>
                        <thead><tr>{html.head}</tr></thead>
                        <tbody><tr>{html.body}</tr><tbody>
                    </table>
                    """
                )
            )
        report.extra = extra
