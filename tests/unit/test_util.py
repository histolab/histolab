# encoding: utf-8

"""Unit test suite for src.histolab.util module."""

import pytest

from src.histolab.util import lazyproperty


class DescribeLazyPropertyDecorator(object):
    """Tests @lazyproperty decorator class."""

    def it_is_a_lazyproperty_object_on_class_access(self, Obj):
        assert isinstance(Obj.fget, lazyproperty)

    def but_it_adopts_the_name_of_the_decorated_method(self, Obj):
        assert Obj.fget.__name__ == "fget"

    def and_it_adopts_the_module_of_the_decorated_method(self, Obj):
        # ---the module name actually, not a module object
        assert Obj.fget.__module__ == __name__

    def and_it_adopts_the_docstring_of_the_decorated_method(self, Obj):
        assert Obj.fget.__doc__ == "Docstring of Obj.fget method definition."

    def it_only_calculates_value_on_first_call(self, obj):
        assert obj.fget == 1
        assert obj.fget == 1

    def it_raises_on_attempt_to_assign(self, obj):
        assert obj.fget == 1
        with pytest.raises(AttributeError):
            obj.fget = 42
        assert obj.fget == 1
        assert obj.fget == 1

    # fixture components ---------------------------------------------

    @pytest.fixture
    def Obj(self):
        class Obj(object):
            @lazyproperty
            def fget(self):
                """Docstring of Obj.fget method definition."""
                if not hasattr(self, "_n"):
                    self._n = 0
                self._n += 1
                return self._n

        return Obj

    @pytest.fixture
    def obj(self, Obj):
        return Obj()
