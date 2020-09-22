# encoding: utf-8

# ================This module is completely inspired by scikit-image================
# https://github.com/scikit-image/scikit-image/blob/master/skimage/data/__init__.py
# ==================================================================================

import os
import shutil
from typing import Tuple

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
    def file_hash(fname: str, alg: str = "sha256") -> str:
        """Calculate the hash of a given file.

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


def _create_image_fetcher():
    try:
        import pooch
    except ImportError:
        # Without pooch, fallback on the standard data directory
        # which for now, includes a few limited data samples
        return None, legacy_data_dir

    pooch_version = __version__.replace(".dev", "+")
    url = "https://github.com/histolab/histolab/raw/{version}/histolab/"

    # Create a new friend to manage your sample data storage
    image_fetcher = pooch.create(
        # Pooch uses appdirs to select an appropriate directory for the cache
        # on each platform.
        # https://github.com/ActiveState/appdirs
        # On linux this converges to
        # '$HOME/.cache/histolab-image'
        # With a version qualifier
        path=pooch.os_cache("histolab-images"),
        base_url=url,
        version=pooch_version,
        env="HISTOLAB_DATADIR",
        registry=registry,
        urls=registry_urls,
    )

    data_dir = os.path.join(str(image_fetcher.abspath), "data")
    return image_fetcher, data_dir


image_fetcher, data_dir = _create_image_fetcher()

if image_fetcher is None:
    HAS_POOCH = False
else:
    HAS_POOCH = True


def _has_hash(path: str, expected_hash: str) -> bool:
    """Check if the provided path has the expected hash.

    Parameters
    ----------
    path: str
    expected_hash: str

    Returns
    -------
    bool
        True if the file hash and the expected one are equal
    """
    if not os.path.exists(path):
        return False
    return file_hash(path) == expected_hash


def _fetch(data_filename: str) -> str:
    """Fetch a given data file from either the local cache or the repository.
    This function provides the path location of the data file given
    its name in the histolab repository.

    Parameters
    ----------
    data_filename: str
        Name of the file in the histolab repository. e.g.
        'breast/sample1.svs'.

    Returns
    -------
    resolved_path: str
        Path of the local file

    Raises
    ------
    KeyError:
        If the filename is not known to the histolab distribution.
    ModuleNotFoundError:
        If the filename is known to the histolab distribution but pooch is not
        installed.
    ConnectionError:
        If the dataset has not been downloaded yet and histolab is unable to connect
        to the internet
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
    # Pooch needs to download the data. Let the image fetcher search for
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


def _init_pooch() -> None:
    os.makedirs(data_dir, exist_ok=True)
    # data_base_dir = os.path.join(data_dir, "..")
    # Fetch all legacy data so that it is available by default
    for filename in legacy_registry:
        _fetch(filename)


if HAS_POOCH:
    _init_pooch()


def _load_svs(filename: str) -> Tuple[openslide.OpenSlide, str]:
    """Load an image file located in the data directory.

    Parameters
    ----------
    filename : str
        Name of the file in the histolab repository

    Returns
    -------
    slide : openslide.OpenSlide
        An OpenSlide object representing a whole-slide image.
    path : str
        Path where the slide is saved

    Raises
    ------
    OpenSlideError:
        OpenSlide cannot open the given input
    """
    try:
        svs = openslide.open_slide(_fetch(filename))
    except openslide.OpenSlideError:  # pragma: no cover
        raise openslide.OpenSlideError(
            "Your wsi has something broken inside, a doctor is needed"
        )
    return svs, _fetch(filename)


def aorta_tissue() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """aorta_tissue() -> Tuple[openslide.OpenSlide, str]

    Aorta tissue, brightfield, JPEG 2000, YCbCr

    This image is available here
    http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/

    Free to use and distribute, with or without modification

    Returns
    -------
    aorta_tissue : openslide.OpenSlide
        Aorta tissue Whole-Slide-Image
    path : str
        Path where the slide is saved
    """
    return _load_svs("aperio/JP2K-33003-1.svs")


def breast_tissue() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """breast_tissue() -> Tuple[openslide.OpenSlide, str]

    Breast tissue, TCGA-BRCA dataset.

    This image is available here
    https://portal.gdc.cancer.gov/files/9c960533-2e58-4e54-97b2-8454dfb4b8c8
    or through the API
    https://api.gdc.cancer.gov/data/9c960533-2e58-4e54-97b2-8454dfb4b8c8

    Access: open

    Returns
    -------
    breast_tissue : openslide.OpenSlide
        Breast tissue Whole-Slide-Image
    path : str
        Path where the slide is saved
    """
    return _load_svs("tcga/breast/9c960533-2e58-4e54-97b2-8454dfb4b8c8")


def breast_tissue_diagnostic_green_pen() -> Tuple[
    openslide.OpenSlide, str
]:  # pragma: no cover
    """breast_tissue_diagnostic_green_pen() -> Tuple[openslide.OpenSlide, str]

    Breast tissue, TCGA-BRCA dataset. Diagnostic slide with green pen.

    This image is available here
    https://portal.gdc.cancer.gov/files/da36d3aa-9b19-492a-af4f-cc028a926d96
    or through the API
    https://api.gdc.cancer.gov/data/da36d3aa-9b19-492a-af4f-cc028a926d96

    Access: open

    Returns
    -------
    breast_tissue : openslide.OpenSlide
        Breast tissue Whole-Slide-Image
    path : str
        Path where the slide is saved
    """
    return _load_svs("tcga/breast/da36d3aa-9b19-492a-af4f-cc028a926d96")


def breast_tissue_diagnostic_red_pen() -> Tuple[
    openslide.OpenSlide, str
]:  # pragma: no cover
    """breast_tissue_diagnostic_red_pen() -> Tuple[openslide.OpenSlide, str]

    Breast tissue, TCGA-BRCA dataset. Diagnostic slide with red pen.

    This image is available here
    https://portal.gdc.cancer.gov/files/f8b4cee6-9149-45b4-ae53-82b0547e1e34
    or through the API
    https://api.gdc.cancer.gov/data/f8b4cee6-9149-45b4-ae53-82b0547e1e34

    Access: open

    Returns
    -------
    breast_tissue : openslide.OpenSlide
        Breast tissue Whole-Slide-Image
    path : str
        Path where the slide is saved
    """
    return _load_svs("tcga/breast/f8b4cee6-9149-45b4-ae53-82b0547e1e34")


def breast_tissue_diagnostic_black_pen() -> Tuple[
    openslide.OpenSlide, str
]:  # pragma: no cover
    """breast_tissue_diagnostic_black_pen() -> Tuple[openslide.OpenSlide, str]

    Breast tissue, TCGA-BRCA dataset. Diagnostic slide with black pen.

    This image is available here
    https://portal.gdc.cancer.gov/files/31e248bf-ee24-4d18-bccb-47046fccb461
    or through the API
    https://api.gdc.cancer.gov/data/31e248bf-ee24-4d18-bccb-47046fccb461

    Access: open

    Returns
    -------
    breast_tissue : openslide.OpenSlide
        Breast tissue Whole-Slide-Image
    path : str
        Path where the slide is saved
    """
    return _load_svs("tcga/breast/31e248bf-ee24-4d18-bccb-47046fccb461")


def cmu_small_region() -> Tuple[openslide.OpenSlide, str]:
    """cmu_small_region() -> Tuple[openslide.OpenSlide, str]

    Carnegie Mellon University MRXS sample tissue

    This image is available here
    http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/

    Licensed under a CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.

    Returns
    -------
    cmu_mrxs_tissue : openslide.OpenSlide
        Sample CMU tissue Whole-Slide-Image
    path : str
        Path where the slide is saved
    """
    return _load_svs("data/cmu_small_region.svs")


def heart_tissue() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """heart_tissue() -> Tuple[openslide.OpenSlide, str]

    Heart tissue, brightfield, JPEG 2000, YCbCr

    This image is available here
    http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/

    Free to use and distribute, with or without modification

    Returns
    -------
    heart_tissue : openslide.OpenSlide
        Heart tissue Whole-Slide-Image
    path : str
        Path where the slide is saved
    """
    return _load_svs("aperio/JP2K-33003-2.svs")


def ovarian_tissue() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """ovarian_tissue() -> Tuple[openslide.OpenSlide, str]

    Ovarian tissue, TCGA-OV dataset.

    This image is available here
    https://portal.gdc.cancer.gov/files/b777ec99-2811-4aa4-9568-13f68e380c86
    or through the API
    https://api.gdc.cancer.gov/data/b777ec99-2811-4aa4-9568-13f68e380c86


    Access: open

    Returns
    -------
    prostate_tissue : openslide.OpenSlide
        Ovarian tissue Whole-Slide-Image
    path : str
        Path where the slide is saved
    """
    return _load_svs("tcga/ovarian/b777ec99-2811-4aa4-9568-13f68e380c86")


def prostate_tissue() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """prostate_tissue() -> Tuple[openslide.OpenSlide, str]

    Prostate tissue, TCGA-PRAD dataset.

    This image is available here
    https://portal.gdc.cancer.gov/files/6b725022-f1d5-4672-8c6c-de8140345210
    or through the API
    https://api.gdc.cancer.gov/data/6b725022-f1d5-4672-8c6c-de8140345210


    Access: open

    Returns
    -------
    prostate_tissue : openslide.OpenSlide
        Prostate tissue Whole-Slide-Image
    path : str
        Path where the slide is saved
    """
    return _load_svs("tcga/prostate/6b725022-f1d5-4672-8c6c-de8140345210")
