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


class HistolabException(Exception):
    """Histolab custom exception main class"""

    def __init__(self, *args) -> None:
        if args:
            self.message = args[0]
        else:
            self.message = None
        super(HistolabException, self).__init__()

    def __str__(self):
        if self.message:
            return self.message
        return ""


class LevelError(HistolabException):
    """Raised when a requested level is not available"""


class FilterCompositionError(HistolabException):
    """Raised when a filter composition for the class is not available"""
