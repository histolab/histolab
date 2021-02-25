Scorer
====================================
The goal of the ``scorer`` module is to provide the grading functions for the `ScoreTiler <tiler.html#src.histolab.tiler.ScoreTiler>`_ extractor. The ``scorer`` objects input a `Tile <tile.html#src.histolab.tile.Tile>`_ object and return their computed score.
In particular, the ``NucleiScorer`` estimates the presence of nuclei in an H&E-stained tile and assigns a higher score to tiles with more nuclei.

.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: src.histolab.scorer
    :members:

.. toctree::