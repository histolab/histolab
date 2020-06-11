# encoding: utf-8

# ================This module is completely inspired by scikit-image================
# https://github.com/scikit-image/scikit-image/blob/master/skimage/data/__init__.py
# ==================================================================================

import os
import shutil
import openslide

from .. import __version__
from ._registry import legacy_registry, registry, registry_urls

legacy_data_dir = os.path.abspath(os.path.dirname(__file__))
histolab_distribution_dir = os.path.join(legacy_data_dir, "..")

try:
    from pooch.utils import file_hash
except ModuleNotFoundError:
    # Function taken from
    # https://github.com/fatiando/pooch/blob/master/pooch/utils.py
    def file_hash(fname, alg="sha256"):
        """
        Calculate the hash of a given file.
        Useful for checking if a file has changed or been corrupted.
        Parameters
        ----------
        fname : str
            The name of the file.
        alg : str
            The type of the hashing algorithm
        Returns
        -------
        hash : str
            The hash of the file.
        Examples
        --------
        >>> fname = "test-file-for-hash.txt"
        >>> with open(fname, "w") as f:
        ...     __ = f.write("content of the file")
        >>> print(file_hash(fname))
        0fc74468e6a9a829f103d069aeb2bb4f8646bad58bf146bb0e3379b759ec4a00
        >>> import os
        >>> os.remove(fname)
        """
        import hashlib

        if alg not in hashlib.algorithms_available:
            raise ValueError("Algorithm '{}' not available in hashlib".format(alg))
        # Calculate the hash in chunks to avoid overloading the memory
        chunksize = 65536
        hasher = hashlib.new(alg)
        with open(fname, "rb") as fin:
            buff = fin.read(chunksize)
            while buff:
                hasher.update(buff)
                buff = fin.read(chunksize)
        return hasher.hexdigest()


def create_image_fetcher():
    try:
        import pooch
    except ImportError:
        # Without pooch, fallback on the standard data directory
        # which for now, includes a few limited data samples
        return None, legacy_data_dir

    pooch_version = __version__.replace(".dev", "+")
    url = "https://github.com/MPBA/histolab/raw/{version}/histolab/"

    # Create a new friend to manage your sample data storage
    image_fetcher = pooch.create(
        # Pooch uses appdirs to select an appropriate directory for the cache
        # on each platform.
        # https://github.com/ActiveState/appdirs
        # On linux this converges to
        # '$HOME/.cache/histolab-image'
        # With a version qualifier
        path=pooch.os_cache("histolab-image"),
        base_url=url,
        version=pooch_version,
        env="HISTOLAB_DATADIR",
        registry=registry,
        urls=registry_urls,
    )

    data_dir = os.path.join(str(image_fetcher.abspath), "data")
    return image_fetcher, data_dir


image_fetcher, data_dir = create_image_fetcher()

if image_fetcher is None:
    has_pooch = False
else:
    has_pooch = True


def _has_hash(path, expected_hash):
    """Check if the provided path has the expected hash."""
    if not os.path.exists(path):
        return False
    return file_hash(path) == expected_hash


def _fetch(data_filename):
    """Fetch a given data file from either the local cache or the repository.
    This function provides the path location of the data file given
    its name in the histolab repository.
    Parameters
    ----------
    data_filename:
        Name of the file in the histolab repository. e.g.
        'breast/sample1.svs'.
    Returns
    -------
    Path of the local file as a python string.
    Raises
    ------
    KeyError:
        If the filename is not known to the histolab distribution.
    ModuleNotFoundError:
        If the filename is known to the histolab distribution but pooch
        is not installed.
    ConnectionError:
        If histolab is unable to connect to the internet but the
        dataset has not been downloaded yet.
    """
    resolved_path = os.path.join(data_dir, "..", data_filename)
    expected_hash = registry[data_filename]
    # Case 1:
    # The file may already be in the data_dir.
    # We may have decided to ship it in the histolab distribution.
    if _has_hash(resolved_path, expected_hash):
        # Nothing to be done, file is where it is expected to be
        return resolved_path

    # Case 2:
    # The user is using a cloned version of the github repo, which
    # contains both the publicly shipped data, and test data.
    # In this case, the file would be located relative to the
    # histolab_distribution_dir
    gh_repository_path = os.path.join(histolab_distribution_dir, data_filename)
    if _has_hash(gh_repository_path, expected_hash):
        parent = os.path.dirname(resolved_path)
        os.makedirs(parent, exist_ok=True)
        shutil.copy2(gh_repository_path, resolved_path)
        return resolved_path

    # Case 3:
    # Pooch not found.
    if image_fetcher is None:
        raise ModuleNotFoundError(
            "The requested file is part of the histolab distribution, "
            "but requires the installation of an optional dependency, pooch. "
            "To install pooch, use your preferred python package manager. "
            "Follow installation instruction found at "
            "https://www.fatiando.org/pooch/latest/install.html"
        )

    # Case 4:
    # Pooch needs to download the data. Let the image fetcher to search for
    # our data. A ConnectionError is raised if no internet connection is
    # available.
    try:
        resolved_path = image_fetcher.fetch(data_filename)
    except ConnectionError as err:
        # If we decide in the future to suppress the underlying 'requests'
        # error, change this to `raise ... from None`. See PEP 3134.
        raise ConnectionError(
            "Tried to download a histolab dataset, but no internet "
            "connection is available. To avoid this message in the "
            "future, try `histolab.data.download_all()` when you are "
            "connected to the internet."
        ) from err
    return resolved_path


def _init_pooch():
    os.makedirs(data_dir, exist_ok=True)
    # data_base_dir = os.path.join(data_dir, "..")
    # Fetch all legacy data so that it is available by default
    for filename in legacy_registry:
        _fetch(filename)


if has_pooch:
    _init_pooch()


def _load_svs(f):
    """Load an image file located in the data directory."""
    try:
        svs = openslide.open_slide(_fetch(f))
    except openslide.OpenSlideError:
        raise openslide.OpenSlideError(
            "Your wsi has something broken inside, a doctor is needed"
        )
    return svs


def cmu_small_region():
    return _load_svs("data/cmu_small_region.svs")


def aperio1():
    return _load_svs("aperio/JP2K-33003-1.svs")
