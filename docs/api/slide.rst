Slide
====================================
The ``slide`` module provides a simple high-level interface to handle a WSI; it contains the `Slide <slide.html#src.histolab.slide.Slide>`_ class, which wraps functions, methods and properties of a virtual slide in a single object. The `Slide <slide.html#src.histolab.slide.Slide>`_ class encapsulates ``OpenSlide``, and relies on the ``openslide-python`` library for the low-level operations on digital slides. A WSI is usually stored in pyramidal format, where each level corresponds to a specific magnification factor. Therefore, two relevant properties of a WSI are: (i) its dimensions at native magnification; (ii) the number of levels and the dimensions at a specified level.

.. note::
    ``OpenSlide`` identifies each magnification level of the WSI with a positive integer number, starting from 0.

A `Slide <slide.html#src.histolab.slide.Slide>`_ is initialized by providing the path where the WSI is stored and the path where processed images (such as the WSI thumbnail or the extracted tiles) will be saved.
Further, the ``slide`` module implements the `SlideSet <slide.html#src.histolab.slide.SlideSet>`_ class, which handles a collection of `Slide <slide.html#src.histolab.slide.Slide>`_ objects stored in the same directory, possibly filtered by the ``valid_extensions`` parameter.

The `slides_stats <slide.html#src.histolab.slide.SlideSet.slides_stats>`_ property of a `SlideSet <slide.html#src.histolab.slide.SlideSet>`_ computes statistics for the WSI collection, namely the number of available slides; the slide with the maximum/minimum width; the slide with the maximum/minimum height; the slide with the maximum/minimum size; the average width/height/size of the slides.


.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: src.histolab.slide
    :members:

.. toctree::
