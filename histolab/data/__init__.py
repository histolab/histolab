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

# ================This module is completely inspired by scikit-image================
# https://github.com/scikit-image/scikit-image/blob/master/skimage/data/__init__.py
# ==================================================================================

import os
import shutil
from typing import Tuple

import openslide
import PIL
from requests.exceptions import HTTPError

from .. import __version__
from ._registry import legacy_registry, registry, registry_urls

legacy_data_dir = os.path.abspath(os.path.dirname(__file__))
histolab_distribution_dir = os.path.join(legacy_data_dir, "..")

try:
    from pooch import file_hash
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
        from pooch import create, os_cache
    except ImportError:
        # Without pooch, fallback on the standard data directory
        # which for now, includes a few limited data samples
        return None, legacy_data_dir

    pooch_version = __version__.replace(".dev", "+")
    url = "https://github.com/histolab/histolab/raw/{version}/histolab/"

    # Create a new friend to manage your sample data storage
    image_fetcher = create(
        # Pooch uses appdirs to select an appropriate directory for the cache
        # on each platform.
        # https://github.com/ActiveState/appdirs
        # On linux this converges to
        # '$HOME/.cache/histolab-image'
        # With a version qualifier
        path=os_cache("histolab-images"),
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
    except HTTPError as httperror:
        raise HTTPError(f"{httperror}")
    except ConnectionError:  # pragma: no cover
        # If we decide in the future to suppress the underlying 'requests'
        # error, change this to `raise ... from None`. See PEP 3134.
        raise ConnectionError(
            "Tried to download a histolab dataset, but no internet "
            "connection is available."
        )
    return resolved_path


def _init_pooch() -> None:
    os.makedirs(data_dir, exist_ok=True)
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
    except PIL.UnidentifiedImageError:
        raise PIL.UnidentifiedImageError(
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
        H&E-stained Whole-Slide-Image of aortic tissue.
    path : str
        Path where the slide is saved
    """
    return _load_svs("aperio/JP2K-33003-1.svs")


def breast_tissue() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """breast_tissue() -> Tuple[openslide.OpenSlide, str]

    Breast tissue, TCGA-BRCA dataset.

    This image is available here
    https://portal.gdc.cancer.gov/files/ad9ed74a-2725-49e6-bf7a-ef100e299989
    or through the API
    https://api.gdc.cancer.gov/data/ad9ed74a-2725-49e6-bf7a-ef100e299989

    It corresponds to TCGA file
    `TCGA-A8-A082-01A-01-TS1.3cad4a77-47a6-4658-becf-d8cffa161d3a.svs`

    Access: open

    Returns
    -------
    breast_tissue : openslide.OpenSlide
        H&E-stained Whole-Slide-Image of breast tissue.
    path : str
        Path where the slide is saved
    """
    return _load_svs(
        "tcga/breast/TCGA-A8-A082-01A-01-TS1.3cad4a77-47a6-4658-becf-d8cffa161d3a.svs"
    )


def breast_tissue_diagnostic_green_pen() -> Tuple[
    openslide.OpenSlide, str
]:  # pragma: no cover
    """breast_tissue_diagnostic_green_pen() -> Tuple[openslide.OpenSlide, str]

    Breast tissue, TCGA-BRCA dataset. Diagnostic slide with green pen marks.

    This image is available here
    https://portal.gdc.cancer.gov/files/3845b8bd-cbe0-49cf-a418-a8120f6c23db
    or through the API
    https://api.gdc.cancer.gov/data/3845b8bd-cbe0-49cf-a418-a8120f6c23db

    It corresponds to TCGA file
    `TCGA-A1-A0SH-01Z-00-DX1.90E71B08-E1D9-4FC2-85AC-062E56DDF17C.svs`

    Access: open

    Returns
    -------
    breast_tissue : openslide.OpenSlide
        H&E-stained Whole-Slide-Image of breast tissue with green pen marks.
    path : str
        Path where the slide is saved
    """
    return _load_svs(
        "tcga/breast/TCGA-A1-A0SH-01Z-00-DX1.90E71B08-E1D9-4FC2-85AC-062E56DDF17C.svs"
    )


def breast_tissue_diagnostic_red_pen() -> Tuple[
    openslide.OpenSlide, str
]:  # pragma: no cover
    """breast_tissue_diagnostic_red_pen() -> Tuple[openslide.OpenSlide, str]

    Breast tissue, TCGA-BRCA dataset. Diagnostic slide with red pen marks.

    This image is available here
    https://portal.gdc.cancer.gov/files/682e4d74-2200-4f34-9e96-8dee968b1568
    or through the API
    https://api.gdc.cancer.gov/data/682e4d74-2200-4f34-9e96-8dee968b1568

    It corresponds to TCGA file
    `TCGA-E9-A24A-01Z-00-DX1.F0342837-5750-4172-B60D-5F902E2A02FD.svs`

    Access: open

    Returns
    -------
    breast_tissue : openslide.OpenSlide
        H&E-stained Whole-Slide-Image of breast tissue with red pen marks.
    path : str
        Path where the slide is saved
    """
    return _load_svs(
        "tcga/breast/TCGA-E9-A24A-01Z-00-DX1.F0342837-5750-4172-B60D-5F902E2A02FD.svs"
    )


def breast_tissue_diagnostic_black_pen() -> Tuple[
    openslide.OpenSlide, str
]:  # pragma: no cover
    """breast_tissue_diagnostic_black_pen() -> Tuple[openslide.OpenSlide, str]

    Breast tissue, TCGA-BRCA dataset. Diagnostic slide with black pen marks.

    This image is available here
    https://portal.gdc.cancer.gov/files/e70c89a5-1c2f-43f8-b6be-589beea55338
    or through the API
    https://api.gdc.cancer.gov/data/e70c89a5-1c2f-43f8-b6be-589beea55338

    It corresponds to TCGA file
    `TCGA-BH-A201-01Z-00-DX1.6D6E3224-50A0-45A2-B231-EEF27CA7EFD2.svs`

    Access: open

    Returns
    -------
    breast_tissue : openslide.OpenSlide
        H&E-stained Whole-Slide-Image of breast tissue with green black marks.
    path : str
        Path where the slide is saved
    """
    return _load_svs(
        "tcga/breast/TCGA-BH-A201-01Z-00-DX1.6D6E3224-50A0-45A2-B231-EEF27CA7EFD2.svs"
    )


def cmu_small_region() -> Tuple[openslide.OpenSlide, str]:
    """cmu_small_region() -> Tuple[openslide.OpenSlide, str]

    Carnegie Mellon University MRXS sample tissue

    This image is available here
    http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/

    Licensed under a CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.

    Returns
    -------
    cmu_mrxs_tissue : openslide.OpenSlide
        H&E-stained Whole-Slide-Image of small tissue region.
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
        H&E-stained Whole-Slide-Image of heart tissue.
    path : str
        Path where the slide is saved
    """
    return _load_svs("aperio/JP2K-33003-2.svs")


def ihc_breast() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """ihc_breast() -> Tuple[openslide.OpenSlide, str]

    Breast cancer resection, staining CD3 (brown) and CD20 (red).

    This image is available here
    https://idr.openmicroscopy.org/ under accession number idr0073, ID `breastCancer12`.

    Returns
    -------
    ihc_breast : openslide.OpenSlide
        IHC-stained Whole-Slide-Image of Breast tissue.
    path : str
        Path where the slide is saved
    """
    return _load_svs("9798433/?format=tif")


def ihc_kidney() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """ihc_kidney() -> Tuple[openslide.OpenSlide, str]

    Renal allograft, staining CD3 (brown) and CD20 (red).

    This image is available here
    https://idr.openmicroscopy.org/ under accession number idr0073, ID `kidney_46_4`.

    Returns
    -------
    ihc_kidney : openslide.OpenSlide
        IHC-stained Whole-Slide-Image of kidney tissue.
    path : str
        Path where the slide is saved
    """
    return _load_svs("9798554/?format=tif")


def ovarian_tissue() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """ovarian_tissue() -> Tuple[openslide.OpenSlide, str]

    tissue of Ovarian Serous Cystadenocarcinoma, TCGA-OV dataset.

    This image is available here
    https://portal.gdc.cancer.gov/files/e968375e-ef58-4607-b457-e6818b2e8431
    or through the API
    https://api.gdc.cancer.gov/data/e968375e-ef58-4607-b457-e6818b2e8431

    It corresponds to TCGA file
    `CGA-13-1404-01A-01-TS1.cecf7044-1d29-4d14-b137-821f8d48881e.svs`


    Access: open

    Returns
    -------
    prostate_tissue : openslide.OpenSlide
        H&E-stained Whole-Slide-Image of ovarian tissue.
    path : str
        Path where the slide is saved
    """
    return _load_svs(
        "tcga/ovarian/TCGA-13-1404-01A-01-TS1.cecf7044-1d29-4d14-b137-821f8d48881e.svs"
    )


def prostate_tissue() -> Tuple[openslide.OpenSlide, str]:  # pragma: no cover
    """prostate_tissue() -> Tuple[openslide.OpenSlide, str]

    tissue of Prostate Adenocarcinoma, TCGA-PRAD dataset.

    This image is available here
    https://portal.gdc.cancer.gov/files/5a8ce04a-0178-49e2-904c-30e21fb4e41e
    or through the API
    https://api.gdc.cancer.gov/data/5a8ce04a-0178-49e2-904c-30e21fb4e41e

    It corresponds to TCGA file
    `TCGA-CH-5753-01A-01-BS1.4311c533-f9c1-4c6f-8b10-922daa3c2e3e.svs`


    Access: open

    Returns
    -------
    prostate_tissue : openslide.OpenSlide
        H&E-stained Whole-Slide-Image of prostate tissue.
    path : str
        Path where the slide is saved
    """
    return _load_svs(
        "tcga/prostate/TCGA-CH-5753-01A-01-BS1.4311c533-f9c1-4c6f-8b10-922daa3c2e3e.svs"
    )
