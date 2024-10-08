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

import pytest

from histolab import exceptions as exp

from ..unitutil import ANY, initializer_mock


class DescribeExceptions:
    @pytest.mark.parametrize("arg", (["hello", "error"], None))
    def it_can_construct_from_str_or_none_level_error(self, request, arg):
        _init = initializer_mock(request, exp.LevelError)

        level_error = exp.LevelError(arg)

        _init.assert_called_once_with(ANY, arg)
        assert isinstance(level_error, exp.LevelError)
        assert isinstance(level_error, Exception)

    def it_can_construct_from_list_level_error(self, request):
        _init = initializer_mock(request, exp.LevelError)
        args = ["hello", "error"]

        level_error = exp.LevelError(*args)

        _init.assert_called_once_with(ANY, *args)
        assert isinstance(level_error, exp.LevelError)
        assert isinstance(level_error, Exception)

    def it_knows_its_message_from_str_level_error(self):
        arg = "error"
        level_error = exp.LevelError(arg)

        message = level_error.message

        assert isinstance(message, str)
        assert message == arg

    def it_knows_its_message_from_list_level_error(self):
        args = ["hello", "error"]
        level_error = exp.LevelError(*args)

        message = level_error.message

        assert isinstance(message, str)
        assert message == list(args)[0]

    def it_knows_its_message_from_none_level_error(self):
        arg = None
        level_error = exp.LevelError(arg)

        message = level_error.message

        assert message is None

    def it_knows_its_message_from_empty_level_error(self):
        level_error = exp.LevelError()

        message = level_error.message

        assert message is None

    def it_knows_its_str_from_str_level_error(self):
        arg = "error"
        level_error = exp.LevelError(arg)

        s = str(level_error)

        assert s == level_error.message

    def it_knows_its_str_from_list_level_error(self):
        args = ["hello", "error"]
        level_error = exp.LevelError(*args)

        s = str(level_error)

        assert s == level_error.message

    def it_knows_its_str_from_none_level_error(self):
        arg = None
        level_error = exp.LevelError(arg)

        s = str(level_error)

        assert s == ""
