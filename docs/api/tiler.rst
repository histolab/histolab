Tiler
====================================
Different logics are implemented for tile extraction in the ``tiler`` module. The constructor of the three extractors `RandomTiler <tiler.html#src.histolab.tiler.RandomTiler>`_, `GridTiler <tiler.html#src.histolab.tiler.GridTiler>`_, and `ScoreTiler <tiler.html#src.histolab.tiler.ScoreTiler>`_ share a similar interface and common parameters that define the extraction design:

#. ``tile_size``: the tile size;
#. ``level``: the extraction level, from 0 to the number of available levels; negative indexing is also possible, counting backward from the number of available levels to 0 (e.g. ``level`` =-1 means selecting the last available level);
#. ``check_tissue``: True if a minimum percentage of tissue over the total area of the tile is required to save the tiles, False otherwise;
#. ``tissue_percent``: number between 0.0 and 100.0 representing the minimum required ratio of tissue over the total area of the image, considered only if ``check_tissue`` equals to True (default is 80.0);
#. ``prefix``: a prefix to be added at the beginning of the tiles' filename (optional, default is the empty string);
#. ``suffix``: a suffix to be added to the end of the tiles' filename (optional, default is ``.png``).

The general mechanism is to (i) create a tiler object, (ii) define a ``Slide`` object, used to identify the input image, and (iii) create a mask object to determine the area for tile extraction within the tissue. The extraction process starts when the tiler's ``extract()`` method is called, with the slide and the mask passed as parameters.

RandomTiler
-----------
The `RandomTiler <tiler.html#src.histolab.tiler.RandomTiler>`_ extractor allows for the extraction of tiles picked at random within the regions defined by the binary mask object. Since there is no intrinsic upper bound of the number of the tiles that could be extracted (no overlap check is performed), the number of wanted tiles must be specified.

In addition to 1-6, the `RandomTiler <tiler.html#src.histolab.tiler.RandomTiler>`_ constructor requires as two additional parameters the number of tiles requested (``n_tiles``), and the random seed (``seed``), to ensure reproducibility between different runs on the same WSI. Note that less than ``n_tiles`` could be extracted from a slide with not enough tissue pixels and a lot of background, which is checked when the parameter ``check_tissue`` is set to True.
``n_tiles`` will be interpreted as the upper bound of the number of tiles requested: it might not be possible to extract ``n_tiles`` tiles from a slide with a little tissue sample and a lot of background.

The extraction procedure will (i) find the regions to extract tiles from, defined by the binary mask object; (ii) generate ``n_tiles`` random tiles; (iii) save only the tiles with enough tissue if the attribute ``check_tissue`` was set to True, save all the generated tiles otherwise.

GridTiler
----------
A second basic approach consists of extracting all the tiles in the areas defined by the binary mask. This strategy is implemented in the `GridTiler <tiler.html#src.histolab.tilerGridTiler>`_ class. The additional ``pixel_overlap`` parameter specifies the number of overlapping pixels between two adjacent tiles, i.e. tiles are cropped by using a sliding window with stride s defined as:

.. math::

     s=(w - \mathrm{pixel\_overlap}) \cdot (h - \mathrm{pixel\_overlap})

where w and h are customizable parameters defining the width and the height of the resulting tiles.
Calling the ``extract`` method on the `GridTiler <tiler.html#src.histolab.tiler.GridTiler>`_ instance will automatically (i) find the regions to extract tiles from, defined by the binary mask object; (ii) generate all the tiles according to the grid structure; (iii) save only the tiles with "enough tissue" if the attribute ``check_tissue`` was set to True, save all the generated tiles otherwise.

ScoreTiler
----------
Tiles extracted from the same WSI may not be equally informative; for example, if the goal is the detection of mitotic activity on H&E slides, tiles with no nuclei are of little interest. The `ScoreTiler <tiler.html#src.histolab.tiler.ScoreTiler>`_ extractor ranks the tiles with respect to a scoring function, described in the ``scorer`` module. In particular, the `ScoreTiler <tiler.html#src.histolab.tiler.ScoreTiler>`_ class extends the ``GridTiler`` extractor by sorting the extracted tiles in a decreasing order, based on the computed score. Notably, the `ScoreTiler <tiler.html#src.histolab.tilerScoreTiler>`_ is agnostic to the scoring function adopted, thus a custom function can be implemented provided that it inputs a ``Tile`` object and outputs a number. The additional parameter ``n_tiles`` controls the number of highest-ranked tiles to save; if ``n_tiles`` =0 all the tiles are kept.
Similarly to the `GridTiler <tiler.html#src.histolab.tiler.GridTiler>`_ extraction process, calling the ``extract`` method on the `ScoreTiler <tiler.html#src.histolab.tiler.ScoreTiler>`_ instance will automatically (i) find the largest tissue area in the WSI; (ii) generate all the tiles according to the grid structure; (iii) retain all the tiles with enough tissue if the attribute ``check_tissue`` was set to True, all the generated tiles otherwise; (iv) sort the tiles in a decreasing order according to the scoring function defined in the ``scorer`` parameter; (v) save only the highest-ranked ``n_tiles`` tiles, if ``n_tiles``>0; (vi) write a summary of the saved tiles and their scores in a CSV file, if the ``report_path`` is specified in the ``extract`` method. The summary reports for each tile t: (i) the tile filename; (ii) its raw score :math:`s_t`; (iii) the normalized score, scaled in the interval [0,1], computed as:

.. math::

     \hat{s}_t = \frac{s_t-\displaystyle{\min_{s\in S}}(s)}{\displaystyle{\max_{s\in S}}(s)-\displaystyle{\min_{s\in S}}(s)}\ ,

where S is the set of the raw scores of all the extracted tiles.


.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: src.histolab.tiler
    :members:
    :inherited-members: locate_tiles
    :exclude-members: Tiler

.. toctree::
