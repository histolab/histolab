Define the tissue mask
=======================

When extracting the tile dataset from the WSI collection, we may want to consider only a portion of the tissue, rather than the whole slide.
For example, a single WSI can include multiple individual slices of tissue, or we may have pathologist annotations with the regions of interest (ROIs) we need to consider.

In this tutorial, we will see how to define different tissue masks in order to refine the tile extraction procedure.

First, we need to load some modules and an example `Slide <slide.html#src.histolab.slide.Slide>`_; here we will consider the `Kidney <data.html#src.histolab.data.ihc_kidney>`_ WSI available in the `data <data.html#data>`_ module.

.. code-block:: ipython3

   from histolab.slide import Slide
   from histolab.data import ihc_kidney
   import os

   BASE_PATH = os.getcwd()

   PROCESS_PATH = os.path.join(BASE_PATH, 'kidney', 'processed')

   ihc_kidney_svs, ihc_kidney_path = ihc_kidney_svs()
   ihc_kidney_slide = Slide(ihc_kidney_path, processed_path=PROCESS_PATH)

   ihc_kidney_slide.thumbnail

.. figure:: https://user-images.githubusercontent.com/31658006/116117338-672d0c00-a6bc-11eb-87a3-1b752aff46d0.png
   :alt: ihc_kidney


From our `Slide <slide.html#src.histolab.slide.Slide>`_ object we can now retrieve a binary mask considering specific regions of the tissue. Notice that available masks are defined in the `masks <masks.html#src.histolab.masks>`_ module.

As a diagnostic check to visualize the mask, we can call the `locate mask <slide.html#src.histolab.slide.Slide.locate_mask>`_ method on the Slide, which outlines the boundaries of the selected mask on the slide's thumbnail.


TissueMask
----------

If we want to account for all the tissue detected on the slide, the `TissueMask <masks.html#src.histolab.masks.TissueMask>`_ is what we need:

.. code-block:: ipython3

   from histolab.masks import TissueMask
   all_tissue_mask = TissueMask()
   ihc_kidney_slide.locate_mask(all_tissue_mask)

.. figure:: https://user-images.githubusercontent.com/31658006/116119755-0f43d480-a6bf-11eb-86eb-3f5b933ede1c.png
   :alt: all tissue


BiggestTissueBoxMask
--------------------

The `BiggestTissueBoxMask <masks.html#src.histolab.masks.BiggestTissueBoxMask>`_ keeps only the largest connected component of the tissue, and returns the bounding box including that region:

.. code-block:: ipython3

   from histolab.masks import BiggestTissueBoxMask
   largest_area_mask = BiggestTissueBoxMask()
   ihc_kidney_slide.locate_mask(largest_area_mask)

.. figure:: https://user-images.githubusercontent.com/31658006/116119576-e02d6300-a6be-11eb-85b2-01df96c9c3eb.png
   :alt: largest bbox


Custom Mask
------------

It is also possible to define a custom binary mask by subclassing the `BinaryMask <masks.html#src.histolab.masks.BinaryMask>`_ object.
For example, we can limit a rectangular region with upper-left coordinates (400, 280) and bottom-right coordinates (300, 320):

.. code-block:: ipython3

   from histolab.masks import BinaryMask
   from histolab.util import rectangle_to_mask
   from histolab.types import CP

   class MyCustomMask(BinaryMask):
        def _mask(self, slide):
            thumb = slide.thumbnail
            my_mask = rectangle_to_mask(thumb.size, CP(400, 280, 300, 320))
            return my_mask

   custom_mask = MyCustomMask()

   ihc_kidney_slide.locate_mask(custom_mask)

.. figure:: https://user-images.githubusercontent.com/31658006/116122414-0acceb00-a6c2-11eb-9af7-b948592ab9ec.png
   :alt: all tissue


Tile extraction within the mask
-------------------------------

We can finally pass our mask to the `extract <tiler.html#src.histolab.tiler.RandomTiler.extract>`_ method of our `Tiler <tiler.html#src.histolab.tiler>`_ object, and visualize the location of the extracted tiles:

.. code-block:: ipython3

    from histolab.tiler import RandomTiler

    rtiler = RandomTiler(
        tile_size=(128, 128),
        n_tiles=50,
        level=0,
        tissue_percent=90,
        seed=0,
    )

    rtiler.extract(ihc_kidney_slide, all_tissue_mask)

    rtiler.locate_tiles(
        slide=ihc_kidney_slide,
        extraction_mask=all_tissue_mask,
    )

.. figure:: https://user-images.githubusercontent.com/31658006/116124001-00135580-a6c4-11eb-90bb-2bed9689e48b.png
   :alt: all tissue

.. note::
    The `BiggestTissueBoxMask <masks.html#src.histolab.masks.BiggestTissueBoxMask>`_ is considered as default binary mask.
