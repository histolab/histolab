import pytest

from histolab import exceptions as exp

from ..unitutil import ANY, initializer_mock


class DescribeExceptions(object):
    @pytest.mark.parametrize("args", ("error", ["hello", "error"], None))
    def it_can_construct_from_args_level_error(self, request, args):
        _init = initializer_mock(request, exp.LevelError)

        if type(args) == list:
            level_error = exp.LevelError(*args)
        else:
            level_error = exp.LevelError(args)

        if type(args) == list:
            _init.assert_called_once_with(ANY, *args)
        else:
            _init.assert_called_once_with(ANY, args)
        assert isinstance(level_error, exp.LevelError)
        assert isinstance(level_error, Exception)

    @pytest.mark.parametrize("args", ("error", ["hello", "error"], None))
    def it_knows_its_message_level_error(self, request, args):
        if type(args) == list:
            level_error = exp.LevelError(*args)
        else:
            level_error = exp.LevelError(args)

        message = level_error.message

        if args:
            if type(args) == str:
                assert type(message) == str
                assert message == args
            if type(args) == list:
                assert type(message) == str
                assert message == list(args)[0]
        else:
            assert message is None

    @pytest.mark.parametrize("args", ("error", ["hello", "error"], None))
    def it_knows_its_str_level_error(self, request, args):
        if type(args) == list:
            level_error = exp.LevelError(*args)
        else:
            level_error = exp.LevelError(args)

        s = str(level_error)

        if not args:
            assert s == ""
        else:
            assert s == level_error.message
