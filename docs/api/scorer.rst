Scorer
====================================
The goal of the ``scorer`` module is to provide the grading functions for the ``scoretiler`` extractor. The ``scorer`` objects input a ``Tile`` object and return their computed score.
In particular, the ``NucleiScorer`` estimates the presence of nuclei in an H&E-stained tile and assigns a higher score to tiles with more nuclei.

The ``NucleiScorer`` class implements an hybrid algorithm that combines thresholding and morphological operations to segment nuclei on H&E-stained histological images. The proposed method is build upon native ``histolab`` filters, namely the ``HematoxylinChannel`` filter, the ``YenThreshold`` filter, and the ``WhiteTopHat`` filter.

The ``NucleiScorer`` class defines the score of a given tile t as:

.. math::

    s_t = N_t\cdot \mathrm{tanh}(T_t) \mathrm{, } \; 0\le s_t<1

where :math:`N_t` is the nuclei ratio on t, computed as number of white pixels on the segmented mask over the tile size, and :math:`T_t` the fraction of tissue in t. Notice that we introduced the hyperbolic tangent to bound the weight of the tissue ratio over the nuclei ratio.

.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: src.histolab.scorer
    :members:

.. toctree::