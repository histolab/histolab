Quick Start
===========

Here we present a step-by-step tutorial on the use of ``histolab`` to
extract a tile dataset from example WSIs. The corresponding Jupyter
Notebook is available at https://github.com/histolab/histolab-box: this
repository contains a complete ``histolab`` environment that can be used
through `Docker <http://www.docker.com>`_
on all platforms.

Thus, the user can decide either to use ``histolab`` through
``histolab-box`` or installing it in his/her python virtual environment
(using conda, pipenv, pyenv, virtualenv, etc...). In the latter case, as
the ``histolab`` package has been published on
`PyPi <http://www.pypi.org>`_, it can be easily installed via the command:


.. code-block:: bash

   pip install histolab

TCGA data
---------

First things first, let’s import some data to work with, for example
the prostate tissue slide and the ovarian tissue slide available in
the ``data`` module:

.. code-block:: ipython3

   from histolab.data import prostate_tissue, ovarian_tissue

**Note:** To use the ``data`` module, you need to install ``pooch``,
also available on PyPI (https://pypi.org/project/pooch/). This step is
needless if we are using the Vagrant/Docker virtual environment.

The calling to a ``data`` function will automatically download the WSI
from the corresponding repository and save the slide in a cached
directory:

.. code-block:: ipython3

   prostate_svs, prostate_path = prostate_tissue()
   ovarian_svs, ovarian_path = ovarian_tissue()

Notice that each ``data`` function outputs the corresponding slide, as
an OpenSlide object, and the path where the slide has been saved.

Slide initialization
--------------------

``histolab`` maps a WSI file into a ``Slide`` object. Each usage of a
WSI requires a 1-o-1 association with a ``Slide`` object contained in
the ``slide`` module:

.. code-block:: ipython3

   from histolab.slide import Slide

To initialize a Slide it is necessary to specify the WSI path, and the
``processed_path`` where the thumbnail and the tiles will be saved. In
our example, we want the ``processed_path`` of each slide to be a
subfolder of the current working directory:

.. code-block:: ipython3

   import os

   BASE_PATH = os.getcwd()

   PROCESS_PATH_PROSTATE = os.path.join(BASE_PATH, 'prostate', 'processed')
   PROCESS_PATH_OVARIAN = os.path.join(BASE_PATH, 'ovarian', 'processed')

   prostate_slide = Slide(prostate_path, processed_path=PROCESS_PATH_PROSTATE)
   ovarian_slide = Slide(ovarian_path, processed_path=PROCESS_PATH_OVARIAN)

**Note:** If the slides were stored in the same folder, this can be
done directly on the whole dataset by using the ``SlideSet`` object
of the ``slide`` module.

With a ``Slide`` object we can easily retrieve information about the
slide, such as the slide name, the number of available levels, the
dimensions at native magnification or at a specified level:

.. code-block:: ipython3

   print(f"Slide name: {prostate_slide.name}")
   print(f"Levels: {prostate_slide.levels}")
   print(f"Dimensions at level 0: {prostate_slide.dimensions}")
   print(f"Dimensions at level 1: {prostate_slide.level_dimensions(level=1)}")
   print(f"Dimensions at level 2: {prostate_slide.level_dimensions(level=2)}")

.. code-block:: bash

   Slide name: 6b725022-f1d5-4672-8c6c-de8140345210
   Levels: [0, 1, 2]
   Dimensions at level 0: (16000, 15316)
   Dimensions at level 1: (4000, 3829)
   Dimensions at level 2: (2000, 1914)

.. code-block:: ipython3

   print(f"Slide name: {ovarian_slide.name}")
   print(f"Levels: {ovarian_slide.levels}")
   print(f"Dimensions at level 0: {ovarian_slide.dimensions}")
   print(f"Dimensions at level 1: {ovarian_slide.level_dimensions(level=1)}")
   print(f"Dimensions at level 2: {ovarian_slide.level_dimensions(level=2)}")

.. code-block:: bash

   Slide name: b777ec99-2811-4aa4-9568-13f68e380c86
   Levels: [0, 1, 2]
   Dimensions at level 0: (30001, 33987)
   Dimensions at level 1: (7500, 8496)
   Dimensions at level 2: (1875, 2124)

Moreover, we can save and show the slide thumbnail in a separate window.
In particular, the thumbnail image will be automatically saved in a subdirectory
of the ``processed_path``:

.. code-block:: ipython3

   prostate_slide.save_thumbnail()
   prostate_slide.show()

.. figure:: https://user-images.githubusercontent.com/4196091/92748324-5033e680-f385-11ea-812b-6a9a225ceca4.png
   :alt: prostate

.. code-block:: ipython3

   ovarian_slide.save_thumbnail()
   ovarian_slide.show()

.. figure:: https://user-images.githubusercontent.com/4196091/92748248-3db9ad00-f385-11ea-846b-a5ce8cf3ca09.png
   :alt: ovarian


Tile extraction
---------------

Once that the ``Slide`` objects are defined, we can proceed to extract
the tiles. To speed up the extraction process, ``histolab``
automatically detects the tissue region with the largest connected area
and crops the tiles within this field. The ``tiler`` module implements
different strategies for the tiles extraction and provides an intuitive
interface to easily retrieve a tile dataset suitable for our task. In
particular, each extraction method is customizable with several common
parameters:

-  ``tile_size``: the tile size;

-  ``level``: the extraction level (from 0 to the number of available
   levels);

-  ``check_tissue``: if a minimum percentage of tissue is required to
   save the tiles (default is 80%);

-  ``prefix``: a prefix to be added at the beginning of the tiles’
   filename (default is the empty string);

-  ``suffix``: a suffix to be added to the end of the tiles’ filename
   (default is .png).

Random Extraction
~~~~~~~~~~~~~~~~~
The simplest approach we may adopt is to randomly crop a fixed number
of tiles from our slides; in this case, we need the ``RandomTiler`` extractor:

.. code-block:: ipython3

   from histolab.tiler import RandomTiler

Let us suppose that we want to randomly extract 6 squared tiles at level 2 of
size 512 from our prostate slide, and that we want to save them only if
they have at least 80% of tissue inside. We then initialize our ``RandomTiler``
extractor as follows:

.. code-block:: ipython3

   random_tiles_extractor = RandomTiler(
       tile_size=(512, 512),
       n_tiles=6,
       level=2,
       seed=42,
       check_tissue=True, # default
       prefix="random/", # save tiles in the "random" subdirectory of slide's processed_path
       suffix=".png" # default
   )

Notice that we also specify the random seed to ensure the reproducibility of
the extraction process. Starting the extraction is as simple as calling
the ``extract`` method on the extractor, passing the slide as parameter:

.. code-block:: ipython3

   random_tiles_extractor.extract(prostate_slide)

.. figure:: https://user-images.githubusercontent.com/4196091/92750145-1663df80-f387-11ea-8d98-7794eef2fd47.png
   :alt: extracted tiles

Random tiles extracted from the prostate slide at level 2.


Grid Extraction
~~~~~~~~~~~~~~~

Instead of picking tiles at random,  we may want to retrieve all the tiles available.
The Grid Tiler extractor crops the tiles following a grid structure on the largest tissue
region detected in the WSI:

.. code-block:: ipython3

   from histolab.tiler import GridTiler

In our example, we want to extract squared tiles at level 0 of size
512 from our ovarian slide, independently of the amount of tissue
detected. By default, tiles will not overlap, namely the parameter
defining the number of overlapping pixels between two adjacent tiles,
``pixel_overlap``, is set to zero:

.. code-block:: ipython3

   grid_tiles_extractor = GridTiler(
      tile_size=(512, 512),
      level=0,
      check_tissue=False,
      pixel_overlap=0, # default
      prefix="grid/", # save tiles in the "grid" subdirectory of slide's processed_path
      suffix=".png" # default
   )

Again, the extraction process starts when the extract method is called
on our extractor:

.. code-block:: ipython3

   grid_tiles_extractor.extract(ovarian_slide)

.. figure:: https://user-images.githubusercontent.com/4196091/92751173-0993bb80-f388-11ea-9d30-a6cd17769d76.png
   :alt: ovarian tile extraction

Examples of non-overlapping grid tiles extracted from the ovarian slide at level 0.

Score-based extraction
~~~~~~~~~~~~~~~~~~~~~~

Depending on the task we will use our tile dataset for, the extracted
tiles may not be equally informative. The ``ScoreTiler`` allows us to
save only the "best" tiles, among all the ones extracted with a grid
structure, based on a specific scoring function. For example, let us
suppose that our goal is the detection of mitotic activity on our
ovarian slide. In this case, tiles with a higher presence of nuclei are
preferable over tiles with few or no nuclei. We can leverage the
``NucleiScorer`` function of the ``scorer`` module to order the
extracted tiles based on the proportion of the tissue and of the
hematoxylin staining. In particular, the score is computed as
:math:`N_t\cdot\mathrm{tanh}(T_t)`, where :math:`N_t` is the percentage
of nuclei and :math:`T_t` the percentage of tissue in the tile
:math:`t`.

First, we need the extractor and the scorer:

.. code-block:: ipython3

   from histolab.tiler import ScoreTiler
   from histolab.scorer import NucleiScorer

As the ``ScoreTiler`` extends the ``GridTiler`` extractor, we also set
the ``pixel_overlap`` as additional parameter. Moreover, we can
specify the number of the top tiles we want to save with the
``n_tile`` parameter:

.. code-block:: ipython3

   scored_tiles_extractor = ScoreTiler(
       scorer = NucleiScorer(),
       tile_size=(512, 512),
       n_tiles=100,
       level=0,
       check_tissue=True,
       pixel_overlap=0, # default
       prefix="scored/", # save tiles in the "scored" subdirectory of slide's processed_path
       suffix=".png" # default
   )

Finally, when we extract our cropped images, we can also write a report
of the saved tiles and their scores in a CSV file:

.. code-block:: ipython3

   summary_filename = "summary_ovarian_tiles.csv"
   SUMMARY_PATH = os.path.join(ovarian_slide.processed_path, summary_filename)

   scored_tiles_extractor.extract(ovarian_slide, report_path=SUMMARY_PATH)

.. figure:: https://user-images.githubusercontent.com/4196091/92751801-9d658780-f388-11ea-8132-5d0c82bb112b.png
   :alt: nuclei scorer

Representation of the score assigned to each extracted tile by the NucleiScorer,
based on the amount of nuclei detected.


.. toctree::

.. toctree::