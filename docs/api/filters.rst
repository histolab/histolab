Filters
====================================
The filters subpackage implements a pool of functions for image manipulation, including  contrast enhancement, color deconvolution, and background removal.
Two modalities of filters are defined by their input types: image filters, and morphological filters, which act on binary masks.

Filters in ``histolab`` are designed to be applied singularly or combined in a chain of transformations. A composition of filters is predefined for tissue segmentation, while custom filter combinations can be used for tissue detection or other tasks.


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   filters/image_filters
   filters/image_filters_functional
   filters/morphological_filters
   filters/morphological_filters_functional



* Image filters:
    * **Transforming image color space**: Color images can be represented using alternative color spaces and the most common one is the RGB space, where the image is represented using distinct channels for Red, Green and Blue. RGB images can be converted to grayscale, namely shifting from 3-channel images to single channel images, e.g. for use with thresholding. ``histolab`` leverages the `Pillow's ImageOps <https://pillow.readthedocs.io/en/stable/reference/ImageOps.html>`_ module for the conversion to the grayscale color space. Besides RGB and grayscale, several color models are extensively used, such as the HSV space, a cylindrical color model where the three independent channels represent the Hue, Saturation and Value of the color. The  HED space [1]_ has been designed to specifically represent the contribution of Hematoxylin, Eosin and Diaminobenzidine dyes in H&E-stained slides and therefore is widely used in pathology (for example, to analyse microscopic blood images [2]_).

    .. figure:: https://user-images.githubusercontent.com/31658006/111822707-6bf3e880-88e4-11eb-8b38-6e108d7adbb7.jpeg
       :alt: Color space conversion filters
       :align: center



    * **Threshold-based filters**: Thresholding is used to compute a binary mask from a grayscale image: the pixels above (or below) a specified threshold become True values, False otherwise. Color images can also be thresholded, using a different cut-off value for each color channel, and then combining the results using a :math:`\land` (logical AND) or a :math:`\lor` (logical OR) operator. ``histolab`` implements different threshold-based filters, based on popular algorithms of interest for pathology. The threshold-based filters in ``histolab`` output a binary mask that results from replacing each pixel in the input image above (or below) the computed threshold with 1, 0 otherwise.


    .. figure:: https://user-images.githubusercontent.com/31658006/111824236-38b25900-88e6-11eb-91ec-3df3c3188123.png
       :alt: Thresholding filter
       :align: center
       :figclass: align-center



    * **Color-based segmentation filters**: The *k*-means algorithm is one of the most popular unsupervised Machine Learning algorithms for clustering multidimensional data. The *k*-means approach can be also applied in image segmentation to separate pixel groups in terms of color, e.g. to detect variation in staining [3]_ or to segment specific structures [4]_ on histological images. ``histolab`` implements two color segmentation filters based on the *k*-means algorithm, namely `KmeansSegmentation <filters/image_filters.html#src.histolab.filters.image_filters.KmeansSegmentation>`_ and `RagThreshold <filters/image_filters.html#src.histolab.filters.image_filters.RagThreshold>`_. The first one segments the image into *n* segments (by default n=800) using *k*-means in the color space, and then colors each segment based on its average color. To overcome the over-segmentation that the *k*-means algorithm may generate, the ``RagThreshold`` filter allows similarly colored segments to be grouped together: (i) the image is segmented with *k*-means; (ii) the Region Adjacency Graph (RAG) is built based on the segments; (iii) nodes in the graph connected by an edge with weight less than a specific threshold *t* (by default t=9) are combined.

    .. figure:: https://user-images.githubusercontent.com/31658006/111825534-dce8cf80-88e7-11eb-9d7f-f05e3276a204.jpeg
       :alt: rag
       :align: center
       :figclass: align-center


    * **Channel extractor filters**: The preparation of histopathological slides is based on processing tissue samples with a sequence of histochemical staining steps. In order to reveal specific structural elements in the tissue, different staining protocols are applied, with a wide range of techniques, colouring reagents (e.g. H&E or specific IHC), and their combinations. Developed for human readers, the protocol modifies the color information to detect, or ignore, specific regions on the WSI according to the task. For example, on H&E-stained images, filtering out pixels with high green channel value remove areas with no presence of nuclei and/or tissue (purple and pink colors, respectively). Also, extracting the hematoxylin channel helps in selecting the regions with cell nuclei, and therefore ease the detection of mitosis. ``histolab`` provides a set of filters designed to extract a single channel from 3-channels images (e.g. RGB, HSV, HED). In particular, the `HematoxylinChannel <filters/image_filters.html#src.histolab.filters.image_filters.HematoxylinChannel>`_ and the `EosinChannel <filters/image_filters.html#src.histolab.filters.image_filters.EosinChannel>`_ methods extract the hematoxylin and eosin channel, respectively, after converting the image into the appropriate color space (HED), and enhancing the contrast.

    .. figure:: https://user-images.githubusercontent.com/31658006/115878802-43ae5b00-a449-11eb-96aa-b502ce302950.jpeg
       :alt: he
       :align: center
       :figclass: align-center


    * **Diagnostic annotations filters**: In clinical practice, pathologists often annotate slides with pen marks to simplify the diagnostic process, for example by delineating cancerous areas or by segmenting regions of interest. These manual annotations, while useful for human analysis, are confounds for an automatic pipeline due to the lack in the standardization of annotation procedures (handwritten labels may be subjective and error-prone [5]_, and because they could alter the feature extraction process. Therefore, it is essential to either eliminate or correct these artifacts [6]_. A Deep Learning pipeline has been introduced to erase ink marks from digital slides by Ali et al. [8]_. Although the method is efficient on reconstructing regions hidden by the annotations, it requires large (manually annotated) datasets and a relevant computational cost. ``histolab`` includes methods to clean ink signs in a combination of image filters [7]_. In particular, green, red and blue marks are deleted by progressively removing pixels within fixed ranges of intensity. While the green and the blue pen filters are extremely effective on annotated H&E slides, the red pen filter should be used carefully: due to similarity with the eosin staining, it could erode reddish regions, such as aggregation of erythrocytes (blood cells).

    .. figure:: https://user-images.githubusercontent.com/31658006/115879398-e6ff7000-a449-11eb-8288-838acda47354.jpeg
       :alt: pen
       :align: center
       :figclass: align-center


.. note:: ``histolab`` stores masks as NumPy arrays. The utility class `ToPILImage <filters/image_filters.html#src.histolab.filters.image_filters.ToPILImage>`_ in the image filters module retrieves the Pillow Image from the corresponding array.

* Morphological filters:
    * **Image preprocessing**: Morphology is a comprehensive set of image processing operations that transform images by using geometrical structures. Morphological operations act by adjusting image pixels based on the value of the pixels in the neighborhood. The choice of the neighborhood's size and shape will affect the behaviour of the morphological operation so that it will be sensitive to specific shapes in the input image [9]_. The *structuring element* is the component of the morphological operations that defines the considered neighborhood: it is a shape (typically a circle or a square) that determines the area used to process each pixel in the image. Usually, the shape and size of the structuring element are chosen to reflect the geometry of the objects in the image that the structuring element will transform: for example, linear structuring element would be used to detect lines in an image. Morphological operations can be applied to binary masks to shrink or enlarge regions of the image. Classic morphological operations include dilation, erosion, opening and closing; ``histolab`` implements these operations in the filters submodule `morphological_filters <filters/morphological_filters.html#morphological-filters>`_. The default structuring element is a disk, with radius 5 for both dilation and erosion, and radius 3 for both opening and closing. However, it is possible to override the default value by passing ``disk_size=N`` as parameter to the filter constructor. The `morphological_filters <filters/morphological_filters.html#morphological-filters>`_ module implements three additional morphological operations useful for manipulating binary masks: `WhiteTopHat <filters/morphological_filters.html#src.histolab.filters.morphological_filters.WhiteTopHat>`_, `RemoveSmallObjects <filters/morphological_filters.html#src.histolab.filters.morphological_filters.RemoveSmallObjects>`_, and `RemoveSmallHoles <filters/morphological_filters.html#src.histolab.filters.morphological_filters.RemoveSmallHoles>`_. The *white top-hat* transformation is defined as the difference between the image and its morphological opening with respect to a structuring element. This operation results in an image including only structures smaller than the structuring element and brighter than their neighborhood, and it is thus used to extract light details on a dark background. The white top-hat filter uses a cross-shaped structuring element with connectivity 1 by default. The `RemoveSmallObjects <filters/morphological_filters.html#src.histolab.filters.morphological_filters.RemoveSmallObjects>`_ filter removes objects with an area smaller than a specified value while the `RemoveSmallHoles <filters/morphological_filters.html#src.histolab.filters.morphological_filters.RemoveSmallHoles>`_ filter "fills" holes with an area smaller than the specified threshold. The minimal area value is set to 3000 for both filters.

    .. figure:: https://user-images.githubusercontent.com/31658006/115879618-275eee00-a44a-11eb-8b52-f24413fce012.jpeg
       :alt: morph
       :align: center
       :figclass: align-center

    * **Segmentation**: ``histolab`` implements the Watershed algorithm, a popular segmentation method for binary masks based on image morphology, useful to separate overlapping objects [10]_. This algorithm works by treating the mask as a topographic map, with the value of each pixel representing the elevation, and by flooding basins from user-defined markers until basins attributed to different markers meet on watershed lines. The `WatershedSegmentation <filters/morphological_filters.html#src.histolab.filters.morphological_filters.WatershedSegmentation>`_ filter first computes an image that represents for each pixel the Euclidean distance D of that pixel to the closest pixel on the background. Then, the points corresponding to the maxima of the distance D are chosen as markers for the algorithm.

.. note:: Notice that both the input and the output of morphological filters are binary masks.

To ease combining filters together, inspired by the ``transforms`` module of torchvision [11]_, ``histolab`` implements the `Compose <filters/image_filters.html#src.histolab.filters.image_filters.Compose>`_ class in the ``filters`` subpackage. `Compose <filters/image_filters.html#src.histolab.filters.image_filters.Compose>`_ allows ``histolab`` filters - both image and morphological filters - to be concatenated as a chain of functions, without any intermediate transformation or prior knowledge on their ordering. In order to enable this composition, filters have been designed to follow the same usage pattern, and to input/output either a PIL Image or a NumPy array object.
To clarify how the `Compose <filters/image_filters.html#src.histolab.filters.image_filters.Compose>`_ object works, let us consider the sequential application of a set of filters, where the output of a filter is the input of the following one:

.. code-block:: python3

    from PIL import Image
    from histolab.filters.image_filters import (
        ApplyMaskImage,
        OtsuThreshold,
        RgbToGrayscale,
    )
    from histolab.filters.morphological_filters import BinaryDilation

    def not_composed_filters(image_rgb):
        rgb_to_grayscale = RgbToGrayscale()
        otsu_threshold = OtsuThreshold()
        binary_dilation = BinaryDilation()
        apply_mask_image = ApplyMaskImage(
            image_rgb
        )  # apply the resulting mask on the original image
        image_gray = rgb_to_grayscale(image_rgb)
        image_thresholded = otsu_threshold(image_gray)
        image_dilated = binary_dilation(image_thresholded)
        return apply_mask_image(image_dilated)

    image_rgb = Image.open("path/to/image.png")
    resulting_image = not_composed_filters(image_rgb)

Despite being formally correct, the above implementation in neither intuitive or economic (in terms of memory used and lines of code). The use of a `Compose <filters/image_filters.html#src.histolab.filters.image_filters.Compose>`_ object leads to a more compact implementation:

.. code-block:: python3

    from PIL import Image
    from histolab.filters.image_filters import (
        ApplyMaskImage,
        Compose,
        OtsuThreshold,
        RgbToGrayscale,
    )
    from histolab.filters.morphological_filters import BinaryDilation

    def composed_filters(image_rgb):
        filters = Compose(
            [
                RgbToGrayscale(),
                OtsuThreshold(),
                BinaryDilation(),
                ApplyMaskImage(image_rgb),
            ]
        )
        return filters(image_rgb)

    image_rgb = Image.open("path/to/image.png")
    resulting_image = composed_filters(image_rgb)

.. note::
    Although ``not_composed_filters`` and ``composed_filters`` functions return the same result, the use of `Compose <filters/image_filters.html#src.histolab.filters.image_filters.Compose>`_ avoids storing intermediate results and wasting memory in case of very large input image. For example, the peak RAM usage for an image of size 19,394px x 6,136px (84.9 MB) is ~2600 MB when using ``composed_filters`` in contrast to the ~3200 MB allocated for the ``not_composed_filters`` function.


References
----------

.. [1] A Ruifrok and et al. “Quantification of histochemical staining by color deconvolution”. *Anal Quant Cytol Histol* 23.4 (2001)
.. [2] K B Suliman and A Krzyzak. “Computerized Counting-Based System for Acute Lymphoblastic Leukemia Detection in Microscopic Blood Images”. Artificial Neural Networks and Machine Learning (ICANN 2018). *Springer* (2018)
.. [3] R D Peng. “Reproducible research in computational science”. Science 334.6060 (2011)
.. [4] J Sieren andet al. “An automated segmentation approach for highlighting the histologicalcomplexity of human lung cancer”. *Ann Biomed Eng* 38.12 (2010)
.. [5] E A Wagar and et al. “Specimen labeling errors: a Q-probes analysis of 147 clinical laboratories”. *Arch Pathol Lab Med* 132.10 (2008)
.. [6] S K Phan and et al. “Biomedical Imaging Informatics for Diagnostic Imaging Marker Selection”. *Health Informatics Data Analysis*. Springer (2017)
.. [7] S Ali and et al. “Ink removal from histopathology whole slide images by combining classification, detection and image generation models”. *IEEE 16th International Symposium on Biomedical Imaging* (ISBI 2019).
.. [8] M Dusenberry and et al. `deep-histopath <https://github.com/CODAIT/deep-histopath>`
.. [9] A P Vartak and V Mankar. “Morphological image segmentation analysis”. *Int J Comput Appl* 6.2 (2013)
.. [10] L Vincent and P Soille. “Watersheds in digital spaces: an efficient algorithm based on immersion simulations”. *IEEE Trans Pattern Anal Mach Intell* 6 (1991)
.. [11] A Paszke and et al. “PyTorch: An Imperative Style, High-Performance Deep Learning Library”. *Advances in Neural Information Processing Systems* 32 (NeurIPS  2019)
