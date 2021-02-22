Data
====================================

The ``data`` module gives access to a set of WSIs in The Cancer Genome Atlas (`TCGA <https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga>`_ ): as detailed in the methods docstring, for each WSI, we access the URL pointing to the corresponding location within the portal, e.g. https://portal.gdc.cancer.gov/files/9c960533-2e58-4e54-97b2-8454dfb4b8c8, to retrieve the WSI. Additional test data are also gathered from OpenSlide, a repository of freely-distributed test `slides <http://openslide.cs.cmu.edu/download/openslide-testdata>`_ from different scanner vendors.

``histolab`` gives access to a pool of selected slides from the two organizations through the ``data`` module. The module leverages the Python package `Pooch <https://pypi.org/project/pooch/>`_, explicitly designed to be integrated in a Python package to include sample datasets. Pooch manages a registry containing the file names, the SHA-256 cryptographic hashes, and download URLs of the accessible data files. Moreover, Pooch verifies the integrity of the downloaded files via the SHA-256 hashes.

.. list-table:: Set of downloadable WSIs.
   :widths: 25 25 25 25
   :header-rows: 1

   * - Tissue
     - Dimensions (wxh)
     - Size (MB)
     - Repository
   * - Aorta
     - 15374x17497
     - 63.8
     - OpenSlide
   * - Heart
     - 32672x47076
     - 289.3
     - OpenSlide
   * - Breast
     - 96972x30682
     - 299.1
     - TCGA-BRCA
   * - Prostate
     - 16000x15316
     - 46.1
     - TCGA-PRAD
   * - Ovary
     - 30001x33987
     - 389.1
     - TCGA-OV
   * - Breast
     - 60928x75840
     - 510.9
     - TCGA-BRCA
   * - Breast
     - 98874x64427
     - 719.6
     - TCGA-BRCA
   * - Breast
     - 121856x94697
     - 1740.8
     - TCGA-BRCA

TCGA-BRCA: TCGA Breast Invasive Carcinoma dataset; TCGA-PRAD: TCGA Prostate Adenocarcinoma dataset; TCGA-OV: Ovarian Serous Cystadenocarcinoma dataset.

.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: src.histolab.data
    :members:

.. toctree::