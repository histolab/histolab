Masks
====================================
The classes implemented in the ``masks`` module define how to retrieve a binary mask of the tissue from a `Slide <slide.html#src.histolab.slide.Slide>`_ object. This step is necessary during the tiles extraction phase.
The ``masks`` module already implements different approaches to retrieve specific tissue regions within the slide:
the `TissueMask <masks.html#src.histolab.masks.TissueMask>`_ class segments the whole tissue area in the slide leveraging a sequence of native filters, including conversion to grayscale, Otsu thresholding, binary dilation, small holes and small objects removal; the `BiggestTissueBoxMask <masks.html#src.histolab.masks.BiggestTissueBoxMask>`_ class applies the same chain of filters as `TissueMask <masks.html#src.histolab.masks.TissueMask>`_ but it returns a binary mask corresponding to the bounding box of the largest connected tissue region within the slide.
Alternatively, a custom binary mask can be defined with the `BinaryMask <masks.html#src.histolab.masks.BinaryMask>`_ class.

.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: src.histolab.masks
    :members:
    :special-members: __call__
    :inherited-members:
    :member-order: bysource

.. toctree::
