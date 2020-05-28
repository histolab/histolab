# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

import collections
import functools
import PIL

import numpy as np

from itertools import filterfalse as ifilterfalse

from PIL import Image


def np_to_pil(np_img):
    """ Convert a NumPy array to a PIL Image.

    Args:
      np_img: The image represented as a NumPy array.

    Returns:
       PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def filter_threshold(img: PIL.Image.Image, threshold: float) -> np.ndarray:
    """Mask image with pixel below the threshold value.

    Parameters
    ----------
    img: PIL.Image.Image
        input image
    threshold: float
        The threshold value to exceed.

    Returns
    -------
    np.ndarray
        Boolean NumPy array representing a mask where a pixel has a value True
        if the corresponding input array pixel exceeds the threshold value.
    """
    img_arr = np.array(img)
    return img_arr > threshold


class Counter(dict):
    """Mapping where default values are zero"""

    def __missing__(self, key):
        return 0


class lazyproperty(object):
    """Decorator like @property, but evaluated only on first access.

    Like @property, this can only be used to decorate methods having only
    a `self` parameter, and is accessed like an attribute on an instance,
    i.e. trailing parentheses are not used. Unlike @property, the decorated
    method is only evaluated on first access; the resulting value is cached
    and that same value returned on second and later access without
    re-evaluation of the method.

    Like @property, this class produces a *data descriptor* object, which is
    stored in the __dict__ of the *class* under the name of the decorated
    method ('fget' nominally). The cached value is stored in the __dict__ of
    the *instance* under that same name.

    Because it is a data descriptor (as opposed to a *non-data descriptor*),
    its `__get__()` method is executed on each access of the decorated
    attribute; the __dict__ item of the same name is "shadowed" by the
    descriptor.

    While this may represent a performance improvement over a property, its
    greater benefit may be its other characteristics. One common use is to
    construct collaborator objects, removing that "real work" from the
    constructor, while still only executing once. It also de-couples client
    code from any sequencing considerations; if it's accessed from more than
    one location, it's assured it will be ready whenever needed.

    Loosely based on: https://stackoverflow.com/a/6849299/1902513.

    A lazyproperty is read-only. There is no counterpart to the optional
    "setter" (or deleter) behavior of an @property. This is critically
    important to maintaining its immutability and idempotence guarantees.
    Attempting to assign to a lazyproperty raises AttributeError
    unconditionally.

    The parameter names in the methods below correspond to this usage
    example::

        class Obj(object)

            @lazyproperty
            def fget(self):
                return 'some result'

        obj = Obj()

    Not suitable for wrapping a function (as opposed to a method) because it
    is not callable.
    """

    def __init__(self, fget):
        """*fget* is the decorated method (a "getter" function).

        A lazyproperty is read-only, so there is only an *fget* function (a
        regular @property can also have an fset and fdel function). This name
        was chosen for consistency with Python's `property` class which uses
        this name for the corresponding parameter.
        """
        # ---maintain a reference to the wrapped getter method
        self._fget = fget
        # ---adopt fget's __name__, __doc__, and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, type=None):
        """Called on each access of 'fget' attribute on class or instance.

        *self* is this instance of a lazyproperty descriptor "wrapping" the
        property method it decorates (`fget`, nominally).

        *obj* is the "host" object instance when the attribute is accessed
        from an object instance, e.g. `obj = Obj(); obj.fget`. *obj* is None
        when accessed on the class, e.g. `Obj.fget`.

        *type* is the class hosting the decorated getter method (`fget`) on
        both class and instance attribute access.
        """
        # ---when accessed on class, e.g. Obj.fget, just return this
        # ---descriptor instance (patched above to look like fget).
        if obj is None:
            return self

        # ---when accessed on instance, start by checking instance __dict__
        value = obj.__dict__.get(self.__name__)
        if value is None:
            # ---on first access, __dict__ item will absent. Evaluate fget()
            # ---and store that value in the (otherwise unused) host-object
            # ---__dict__ value of same name ('fget' nominally)
            value = self._fget(obj)
            obj.__dict__[self.__name__] = value
        return value

    def __set__(self, obj, value):
        """Raises unconditionally, to preserve read-only behavior.

        This decorator is intended to implement immutable (and idempotent)
        object attributes. For that reason, assignment to this property must
        be explicitly prevented.

        If this __set__ method was not present, this descriptor would become
        a *non-data descriptor*. That would be nice because the cached value
        would be accessed directly once set (__dict__ attrs have precedence
        over non-data descriptors on instance attribute lookup). The problem
        is, there would be nothing to stop assignment to the cached value,
        which would overwrite the result of `fget()` and break both the
        immutability and idempotence guarantees of this decorator.

        The performance with this __set__() method in place was roughly 0.4
        usec per access when measured on a 2.8GHz development machine; so
        quite snappy and probably not a rich target for optimization efforts.
        """
        raise AttributeError("can't set attribute")


def lru_cache(maxsize=100):  # pragma: no cover
    """Least-recently-used cache decorator.

    Arguments to the cached function must be hashable.
    Cache performance statistics stored in f.hits and f.misses.
    Clear the cache with f.clear().
    http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used
    """
    maxqueue = maxsize * 10

    def decorating_function(
        user_function, len=len, iter=iter, tuple=tuple, sorted=sorted, KeyError=KeyError
    ):
        cache = {}  # mapping of args to results
        queue = collections.deque()  # order that keys have been used
        refcount = Counter()  # times each key is in the queue
        sentinel = object()  # marker for looping around the queue
        kwd_mark = object()  # separate positional and keyword args

        # lookup optimizations (ugly but fast)
        queue_append, queue_popleft = queue.append, queue.popleft
        queue_appendleft, queue_pop = queue.appendleft, queue.pop

        @functools.wraps(user_function)
        def wrapper(*args, **kwds):
            # cache key records both positional and keyword args
            key = args
            if kwds:
                key += (kwd_mark,) + tuple(sorted(kwds.items()))

            # record recent use of this key
            queue_append(key)
            refcount[key] += 1

            # get cache entry or compute if not found
            try:
                result = cache[key]
                wrapper.hits += 1
            except KeyError:
                result = user_function(*args, **kwds)
                cache[key] = result
                wrapper.misses += 1

                # purge least recently used cache entry
                if len(cache) > maxsize:
                    key = queue_popleft()
                    refcount[key] -= 1
                    while refcount[key]:
                        key = queue_popleft()
                        refcount[key] -= 1
                    del cache[key], refcount[key]

            # periodically compact the queue by eliminating duplicate keys
            # while preserving order of most recent access
            if len(queue) > maxqueue:
                refcount.clear()
                queue_appendleft(sentinel)
                for key in ifilterfalse(
                    refcount.__contains__, iter(queue_pop, sentinel)
                ):
                    queue_appendleft(key)
                    refcount[key] = 1

            return result

        def clear():  # pragma: no cover
            cache.clear()
            queue.clear()
            refcount.clear()
            wrapper.hits = wrapper.misses = 0

        wrapper.hits = wrapper.misses = 0
        wrapper.clear = clear
        return wrapper

    return decorating_function


memoize = lru_cache(100)
