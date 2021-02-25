Tile
====================================
The ``tile`` module contains the ``Tile`` class to manage a rectangular region cropped from a ``Slide``. A ``Tile`` object is described by (i) its extraction coordinates at native magnification (corresponding to level 0 in ``OpenSlide``), (ii) the level of extraction, (iii) the actual image, stored as a ``PIL Image``. A ``Tile`` object will be created internally during the tile extraction process.

A ``tile`` object can be evaluated for informativeness. In particular, the ``has_enough_tissue`` method checks if the proportion of the detected tissue over the total area of the tile is above a specified threshold (by default 80%). Internally,  the  method  quantifies  the  amount  of  tissue  by  applying  a  chain  of ``histolab`` filters, including conversion to grayscale, Otsu thresholding, binary dilation and small holes filling.


.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: src.histolab.tile
    :members:

.. toctree::