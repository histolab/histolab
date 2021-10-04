Tile
====================================
The ``tile`` module contains the `Tile <tile.html#histolab.tile.Tile>`_ class to manage a rectangular region cropped from a `Slide <slide.html#histolab.slide.Slide>`_. A `Tile <tile.html#histolab.tile.Tile>`_ object is described by (i) its extraction coordinates at native magnification (corresponding to level 0 in ``OpenSlide``), (ii) the level of extraction, (iii) the actual image, stored as a PIL Image. A `Tile <tile.html#histolab.tile.Tile>`_ object will be created internally during the tile extraction process.


.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: histolab.tile
    :members:

.. toctree::
