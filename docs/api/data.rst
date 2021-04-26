Data
====================================

The ``data`` module gives access to a set of publicly available WSIs, stained with different techniques (H&E and IHC). In particular, slides in the ``data`` module are retrieved from the following repositories:

* The Cancer Genome Atlas (`TCGA <https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga>`_): as detailed in the methods docstring, for each WSI, we access the URL pointing to the corresponding location within the portal, e.g. https://portal.gdc.cancer.gov/files/9c960533-2e58-4e54-97b2-8454dfb4b8c8, to retrieve the WSI;
* OpenSlide, a repository of freely-distributed test `slides <http://openslide.cs.cmu.edu/download/openslide-testdata>`_ from different scanner vendors;
* Image Data Resource (`IDR <https://idr.openmicroscopy.org>`_): the WSIs are selected from the data collection provided by Schaadt et al. [1]_ and available at IDR under the accession number `idr0073`.

.. note::
    We use `Pooch <https://pypi.org/project/pooch/>`_ under the hood, which is an optional requirement for ``histolab`` and needs to be installed separately with:

    .. code-block:: bash

        pip install pooch

.. list-table:: Set of downloadable WSIs.
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Tissue
     - Dimensions (wxh)
     - Size (MB)
     - Repository
     - Staining
   * - `Aorta <data.html#src.histolab.data.aorta_tissue>`_
     - 15374x17497
     - 63.8
     - OpenSlide
     - H&E
   * - `CMU small sample <data.html#src.histolab.data.cmu_small_region>`_
     - 2220x2967
     - 1.8
     - OpenSlide
     - H&E
   * - `Breast <data.html#src.histolab.data.breast_tissue>`_
     - 96972x30682
     - 299.1
     - TCGA-BRCA
     - H&E
   * - `Breast (black pen) <data.html#src.histolab.data.breast_tissue_diagnostic_black_pen>`_
     - 121856x94697
     - 1740.8
     - TCGA-BRCA
     - H&E
   * - `Breast (green pen) <data.html#src.histolab.data.breast_tissue_diagnostic_green_pen>`_
     - 98874x64427
     - 719.6
     - TCGA-BRCA
     - H&E
   * - `Breast (red pen) <data.html#src.histolab.data.breast_tissue_diagnostic_red_pen>`_
     - 60928x75840
     - 510.9
     - TCGA-BRCA
     - H&E
   * - `Breast (IHC) <data.html#src.histolab.data.ihc_breast>`_
     - 99606x7121
     - 218.3
     - IDR
     - IHC
   * - `Heart <data.html#src.histolab.data.heart_tissue>`_
     - 32672x47076
     - 289.3
     - OpenSlide
     - H&E
   * - `Kidney <data.html#src.histolab.data.ihc_kidney>`_
     - 5179x4192
     - 66.1
     - IDR
     - IHC
   * - `Ovary <data.html#src.histolab.data.ovarian_tissue>`_
     - 30001x33987
     - 389.1
     - TCGA-OV
     - H&E
   * - `Prostate <data.html#src.histolab.data.prostate_tissue>`_
     - 16000x15316
     - 46.1
     - TCGA-PRAD
     - H&E


TCGA-BRCA: TCGA Breast Invasive Carcinoma dataset; TCGA-PRAD: TCGA Prostate Adenocarcinoma dataset; TCGA-OV: Ovarian Serous Cystadenocarcinoma dataset.

.. figure:: https://user-images.githubusercontent.com/31658006/115860383-b3185080-a431-11eb-89c1-8b6da793cc42.jpeg
   :alt: Thumbnails of avaliable WSIs
   :align: center
   :figclass: align-center


.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: src.histolab.data
    :members:

.. toctree::

References
----------

.. [1] Schaadt NS, Schönmeyer R, Forestier G, et al. "Graph-based description of tertiary lymphoid organs at single-cell level." PLoS Comput Biol. (2020)
