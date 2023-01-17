Stain Normalizer
================

Digital pathology images can present strong color differences due to diverse acquisition techniques (e.g., scanners, laboratory equipment and procedures)
The ``stain_normalizer`` module collects methods to transfer the staining style of an image (target) to another image (source).
Stain normalization is often adopted as data standardization procedure in deep learning model training.
The `MacenkoStainNormalizer <stain_normalizer.html#histolab.stain_normalizer.MacenkoStainNormalizer>`_ implements the stain normalization method proposed by Macenko et al. [1]_.
The `ReinhardStainNormalizer <stain_normalizer.html#histolab.stain_normalizer.ReinhardStainNormalizer>`_ implements the Reinhard's stain normalization [2]_.

.. figure:: https://user-images.githubusercontent.com/31658006/212481701-062e00ba-6f5e-4ebd-85da-ac5655802281.jpeg
   :alt: Target image for stain normalization.
   :width: 300
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


Usage
----------

The stain normalization methods can be used to transfer the staining's appearance of any histology image. First, we need to establish the image to be used as a style reference, and the image to be transformed accordingly:

.. code-block:: ipython3

   from histolab.stain_normalizer import ReinhardStainNormalizer
   from PIL import Image
   target_image = Image.open("/path/target/img/img2.png")
   target_image

.. figure:: https://user-images.githubusercontent.com/31658006/212924301-c80f454e-f99a-4479-9852-6ef988c078aa.png
   :align: center
   :width: 200
   :figclass: align-center

.. code-block:: ipython3

   source_image = Image.open("/path/img/to/normalize/img1.png")
   source_image

.. figure:: https://user-images.githubusercontent.com/31658006/212924179-a85573b6-1bb3-4f9b-a8ab-00a26b1d652e.png
   :align: center
   :width: 200
   :figclass: align-center



The chosen stain normalization method must be first fit on the target image and then applied to the source image:

.. code-block:: ipython3

   normalizer = ReinhardStainNormalizer()
   normalizer.fit(target_image)
   normalized_img = normalizer.transform(source_image)
   normalized_img

.. figure:: https://user-images.githubusercontent.com/31658006/212924592-2d591852-0e3c-4f59-b821-655c793a4426.png
   :align: center
   :width: 200
   :figclass: align-center

References
----------

.. [1] Macenko, Marc, et al. "A method for normalizing histology slides for quantitative analysis." 2009 IEEE international symposium on biomedical imaging: from nano to macro. IEEE (2009)
.. [2] Reinhard, Erik, et al. "Color transfer between images." IEEE Computer graphics and applications 21.5 (2001)
