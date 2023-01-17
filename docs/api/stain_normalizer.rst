Stain Normalizer
================

Digital pathology images can present strong color differences due to diverse acquisition techniques (e.g., scanners, laboratory equipment and procedures)
The ``stain_normalizer`` module collects methods to transfer the staining style of an image (target) to another image (source).
Stain normalization is often adopted as data standardization procedure in deep learning model training.
The `MacenkoStainNormalizer <stain_normalizer.html#histolab.stain_normalizer.MacenkoStainNormalizer>`_ implements the stain normalization method proposed by Macenko et al. [1]_. 
The `ReinhardStainNormalizer <stain_normalizer.html#histolab.stain_normalizer.ReinhardStainNormalizer>`_ implements the Reinhard's stain normalization described in [2]_. 

.. figure:: https://user-images.githubusercontent.com/31658006/212481701-062e00ba-6f5e-4ebd-85da-ac5655802281.jpeg
   :alt: Target image for stain normalization.
   :align: center
   :figclass: align-center

.. figure:: https://user-images.githubusercontent.com/31658006/212481611-1678c2f4-bb4e-4c14-8bc3-a3d3101ee82f.jpeg
   :alt: Comparison of available stain normalization techniques.
   :align: center
   :figclass: align-center

.. toctree::
   :caption: API Reference
   :maxdepth: 2

.. automodule:: src.histolab.stain_normalizer
    :members:
    :inherited-members:

.. toctree::

References
----------

.. [1] Macenko, Marc, et al. "A method for normalizing histology slides for quantitative analysis." 2009 IEEE international symposium on biomedical imaging: from nano to macro. IEEE (2009)
.. [2] Reinhard, Erik, et al. "Color transfer between images." IEEE Computer graphics and applications 21.5 (2001)