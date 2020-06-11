# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2020 All Histolab Contributors
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

import os
import re
import tempfile
import urllib.parse
import urllib.request
from contextlib import contextmanager
from urllib.error import HTTPError, URLError

URL_REGEX = re.compile(r"http://|https://|ftp://|file://|file:\\")


def _is_url(filename):
    """Return True if string is an http or ftp path."""
    return isinstance(filename, str) and URL_REGEX.match(filename) is not None


@contextmanager
def file_or_url_context(resource_name):
    """Yield name of file from the given resource (i.e. file or url)."""
    if _is_url(resource_name):
        url_components = urllib.parse.urlparse(resource_name)
        _, ext = os.path.splitext(url_components.path)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                u = urllib.request.urlopen(resource_name)
                f.write(u.read())
            yield f.name
        except (URLError, HTTPError):
            # could not open the given URL
            os.remove(f.name)
            raise
        except (FileNotFoundError, FileExistsError, PermissionError, BaseException):
            # could not create temporary file
            raise
        else:
            os.remove(f.name)
    else:
        yield resource_name
