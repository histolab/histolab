import os

import numpy as np
import PIL
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
from PIL import Image, ImageOps

from src.histolab import util


def invert(img: PIL.Image.Image) -> PIL.Image.Image:
    """Invert an image.

    Invert the intensity range of the input image, so that the dtype
    maximum is now the dtype minimum, and vice-versa.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        Inverted image
    """
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        rgb_img = Image.merge("RGB", (r, g, b))

        inverted_img = ImageOps.invert(rgb_img)

        r2, g2, b2 = inverted_img.split()
        final_inverted_img = Image.merge("RGBA", (r2, g2, b2, a))

    else:
        final_inverted_img = ImageOps.invert(img)

    return final_inverted_img


def filter_rgb_to_grayscale(img, output_type="uint8"):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.

    Shape (h, w, c) to (h, w).

    Returns:
        Grayscale image as NumPy array with shape (h, w).
    """
    # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    grayscale = np.dot(img._slide.resampled_array[..., :3], [0.2125, 0.7154, 0.0721])
    grayscale = (
        grayscale.astype(output_type)
        if output_type == "float"
        else grayscale.astype("uint8")
    )
    return grayscale


def filter_rgb_to_hed(img, np_array, output_type="uint8"):
    """
    Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.

    Args:
        np_img: RGB image as a NumPy array.
        output_type: Type of array to return (float or uint8).

    Returns:
        NumPy array (float or uint8) with HED channels.
    """
    hed = sk_color.rgb2hed(np_array)
    if output_type == "float":
        hed = sk_exposure.rescale_intensity(hed, out_range=(0.0, 1.0))
    else:
        hed = (sk_exposure.rescale_intensity(hed, out_range=(0, 255))).astype("uint8")

    return hed


def filter_rgb_to_hsv(img, np_array):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).

    Args:
        np_img: RGB image as a NumPy array.
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        Image as NumPy array in HSV representation.
    """
    hsv = sk_color.rgb2hsv(np_array)
    return hsv


def filter_contrast_stretch(img, np_array, low=40, high=60):
    """
    Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
    a specified range.

    Args:
        np_img: Image as a NumPy array (gray or RGB).
        low: Range low value (0 to 255).
        high: Range high value (0 to 255).

    Returns:
        Image as NumPy array with contrast enhanced.
    """
    low_p, high_p = np.percentile(np_array, (low * 100 / 255, high * 100 / 255))
    contrast_stretch = sk_exposure.rescale_intensity(np_array, in_range=(low_p, high_p))
    return contrast_stretch


def filter_histogram_equalization(img, np_array, nbins=256, output_type="uint8"):
    """
    Filter image (gray or RGB) using histogram equalization to increase contrast in image.

    Args:
        np_img: Image as a NumPy array (gray or RGB).
        nbins: Number of histogram bins.
        output_type: Type of array to return (float or uint8).

    Returns:
        NumPy array (float or uint8) with contrast enhanced by histogram equalization.
    """
    # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
    if np_img.dtype == "uint8" and nbins != 256:
        np_img = np_img / 255
    hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
    if output_type == "float":
        pass
    else:
        hist_equ = (hist_equ * 255).astype("uint8")
    util.np_info(hist_equ, "Hist Equalization", t.elapsed())
    return hist_equ


def filter_adaptive_equalization(
    img, np_array, nbins=256, clip_limit=0.01, output_type="uint8"
):
    """
    Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
    is enhanced.

    Args:
        np_img: Image as a NumPy array (gray or RGB).
        nbins: Number of histogram bins.
        clip_limit: Clipping limit where higher value increases contrast.
        output_type: Type of array to return (float or uint8).

    Returns:
        NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
    """
    adapt_equ = sk_exposure.equalize_adapthist(
        np_img, nbins=nbins, clip_limit=clip_limit
    )
    if output_type == "float":
        pass
    else:
        adapt_equ = (adapt_equ * 255).astype("uint8")
    util.np_info(adapt_equ, "Adapt Equalization", t.elapsed())
    return adapt_equ


def filter_local_equalization(img, disk_size=50):
    """
    Filter image (gray) using local equalization, which uses local histograms based on the disk structuring element.

    Args:
        np_img: Image as a NumPy array.
        disk_size: Radius of the disk structuring element used for the local histograms

    Returns:
        NumPy array with contrast enhanced using local equalization.
    """
    local_equ = sk_filters.rank.equalize(img, selem=sk_morphology.disk(disk_size))
    return local_equ


def filter_kmeans_segmentation(img, np_array, compactness=10, n_segments=800):
    """
    Use K-means segmentation (color/space proximity) to segment RGB image where each segment is
    colored based on the average color for that segment.

    Args:
        np_img: Binary image as a NumPy array.
        compactness: Color proximity versus space proximity factor.
        n_segments: The number of segments.

    Returns:
        NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
        color for that segment.
    """
    labels = sk_segmentation.slic(
        np_array, compactness=compactness, n_segments=n_segments
    )
    result = sk_color.label2rgb(labels, np_array, kind="avg")
    return result


def filter_rag_threshold(img, np_array, compactness=10, n_segments=800, threshold=9):
    """
    Use K-means segmentation to segment RGB image, build region adjacency graph based on the segments, combine
    similar regions based on threshold value, and then output these resulting region segments.

    Args:
        np_img: Binary image as a NumPy array.
        compactness: Color proximity versus space proximity factor.
        n_segments: The number of segments.
        threshold: Threshold value for combining regions.

    Returns:
        NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
        color for that segment (and similar segments have been combined).
    """
    labels = sk_segmentation.slic(
        np_array, compactness=compactness, n_segments=n_segments
    )
    g = sk_future.graph.rag_mean_color(np_array, labels)
    labels2 = sk_future.graph.cut_threshold(labels, g, threshold)
    result = sk_color.label2rgb(labels2, np_array, kind="avg")
    return result


# -------- Branching function
def filter_hysteresis_threshold(img, low=50, high=100, output_type="uint8"):
    """
    Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.

    Args:
        low: Low threshold.
        high: High threshold.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
    """
    hyst = sk_filters.apply_hysteresis_threshold(img._slide.resampled_array, low, high)
    return img._type_dispatcher(hyst, output_type)


def filter_otsu_threshold(img, output_type="uint8"):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

    Args:
        np_img: Image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
    """
    otsu_thresh_value = sk_filters.threshold_otsu(img._slide.resampled_array)
    return img.filter_threshold(
        img._slide.resampled_array, otsu_thresh_value, output_type
    )


def filter_local_otsu_threshold(img, disk_size=3, output_type="uint8"):
    """
    Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the
    local Otsu threshold.

    Args:
        disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
    """
    local_otsu = sk_filters.rank.otsu(
        img._slide.resampled_array, sk_morphology.disk(disk_size)
    )
    return img._type_dispatcher(local_otsu, output_type)


def filter_entropy(img, neighborhood=9, threshold=5, output_type="uint8"):
    """
    Filter image based on entropy (complexity).

    Args:
        neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
        threshold: Threshold value.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
    """
    entropy = (
        sk_filters.rank.entropy(
            img._slide.resampled_array, np.ones((neighborhood, neighborhood))
        )
        > threshold
    )
    return img._type_dispatcher(entropy, output_type)


# input of canny filter is a greyscale
def filter_canny(img, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8"):
    """
    Filter image based on Canny algorithm edges.

    Args:
        sigma: Width (std dev) of Gaussian.
        low_threshold: Low hysteresis threshold value.
        high_threshold: High hysteresis threshold value.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
    """
    filter_canny = sk_feature.canny(
        img._slide.resampled_array,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    return img._type_dispatcher(filter_canny, output_type)


def filter_grays(img: PIL.Image.Image, tolerance: int = 15) -> PIL.Image.Image:
    """Filter out pixels where the red, green, and blue channel values are similar, i.e. under
    a specified tolerance.

    Parameters
    ----------
    img : PIL.Image.Image
          Input image
    tolerance: int
               if difference between values is below this threshold,
               values are considered similar and thus filtered out

    Returns
    -------
    PIL.Image.Image
         Mask image where the similar values are masked out
    """
    img_arr = np.array(img)
    (h, w, c) = img_arr.shape

    img_arr = img_arr.astype(np.int)
    rg_diff = abs(img_arr[:, :, 0] - img_arr[:, :, 1]) <= tolerance
    rb_diff = abs(img_arr[:, :, 0] - img_arr[:, :, 2]) <= tolerance
    gb_diff = abs(img_arr[:, :, 1] - img_arr[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    return PIL.Image.fromarray(result)


def filter_threshold(img, np_array, threshold, output_type="bool"):
    """
    Return mask where a pixel has a value if it exceeds the threshold value.

    Args:
        np_img: Binary image as a NumPy array.
        threshold: The threshold value to exceed.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
        pixel exceeds the threshold value.
    """
    result = np_array > threshold
    return img._type_dispatcher(result, output_type)


def filter_green_channel(
    np_img,
    green_thresh=200,
    avoid_overmask=True,
    overmask_thresh=90,
    output_type="bool",
):
    """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.

  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """
    t = Time()

    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (
        (mask_percentage >= overmask_thresh)
        and (green_thresh < 255)
        and (avoid_overmask is True)
    ):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        print(
            "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d"
            % (mask_percentage, overmask_thresh, green_thresh, new_green_thresh)
        )
        gr_ch_mask = filter_green_channel(
            np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type
        )
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    util.np_info(np_img, "Filter Green Channel", t.elapsed())
    return np_img


def filter_red(
    img, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
):
    """
    Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
    red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_lower_thresh: Red channel lower threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_upper_thresh: Blue channel upper threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    r = img._slide.resampled_array[:, :, 0] > red_lower_thresh
    g = img._slide.resampled_array[:, :, 1] < green_upper_thresh
    b = img._slide.resampled_array[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    return img._type_dispatcher(result, output_type)


def filter_red_pen(img, output_type="bool"):
    """
    Create a mask to filter out red pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    result = (
        filter_red(
            img._slide.resampled_array,
            red_lower_thresh=150,
            green_upper_thresh=80,
            blue_upper_thresh=90,
        )
        & filter_red(
            img._slide.resampled_array,
            red_lower_thresh=110,
            green_upper_thresh=20,
            blue_upper_thresh=30,
        )
        & filter_red(
            img._slide.resampled_array,
            red_lower_thresh=185,
            green_upper_thresh=65,
            blue_upper_thresh=105,
        )
        & filter_red(
            img._slide.resampled_array,
            red_lower_thresh=195,
            green_upper_thresh=85,
            blue_upper_thresh=125,
        )
        & filter_red(
            img._slide.resampled_array,
            red_lower_thresh=220,
            green_upper_thresh=115,
            blue_upper_thresh=145,
        )
        & filter_red(
            img._slide.resampled_array,
            red_lower_thresh=125,
            green_upper_thresh=40,
            blue_upper_thresh=70,
        )
        & filter_red(
            img._slide.resampled_array,
            red_lower_thresh=200,
            green_upper_thresh=120,
            blue_upper_thresh=150,
        )
        & filter_red(
            img._slide.resampled_array,
            red_lower_thresh=100,
            green_upper_thresh=50,
            blue_upper_thresh=65,
        )
        & filter_red(
            img._slide.resampled_array,
            red_lower_thresh=85,
            green_upper_thresh=25,
            blue_upper_thresh=45,
        )
    )

    return img._type_dispatcher(result, output_type)


def filter_green(
    img,
    red_upper_thresh,
    green_lower_thresh,
    blue_lower_thresh,
    output_type="bool",
    display_np_info=False,
):
    """
    Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
    red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
    Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
    lower threshold value rather than a blue channel upper threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_lower_thresh: Green channel lower threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = img._slide.resampled_array[:, :, 0] < red_upper_thresh
    g = img._slide.resampled_array[:, :, 1] > green_lower_thresh
    b = img._slide.resampled_array[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return img._type_dispatcher(result, output_type)


def filter_green_pen(img, output_type="bool"):
    """
    Create a mask to filter out green pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    result = (
        filter_green(
            img._slide.resampled_array,
            red_upper_thresh=150,
            green_lower_thresh=160,
            blue_lower_thresh=140,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=70,
            green_lower_thresh=110,
            blue_lower_thresh=110,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=45,
            green_lower_thresh=115,
            blue_lower_thresh=100,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=30,
            green_lower_thresh=75,
            blue_lower_thresh=60,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=195,
            green_lower_thresh=220,
            blue_lower_thresh=210,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=225,
            green_lower_thresh=230,
            blue_lower_thresh=225,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=170,
            green_lower_thresh=210,
            blue_lower_thresh=200,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=20,
            green_lower_thresh=30,
            blue_lower_thresh=20,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=50,
            green_lower_thresh=60,
            blue_lower_thresh=40,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=30,
            green_lower_thresh=50,
            blue_lower_thresh=35,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=65,
            green_lower_thresh=70,
            blue_lower_thresh=60,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=100,
            green_lower_thresh=110,
            blue_lower_thresh=105,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=165,
            green_lower_thresh=180,
            blue_lower_thresh=180,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=140,
            green_lower_thresh=140,
            blue_lower_thresh=150,
        )
        & filter_green(
            img._slide.resampled_array,
            red_upper_thresh=185,
            green_lower_thresh=195,
            blue_lower_thresh=195,
        )
    )
    return img._type_dispatcher(result, output_type)


def filter_blue(
    img,
    red_upper_thresh,
    green_upper_thresh,
    blue_lower_thresh,
    output_type="bool",
    display_np_info=False,
):
    """
    Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = img._slide.resampled_array[:, :, 0] < red_upper_thresh
    g = img._slide.resampled_array[:, :, 1] < green_upper_thresh
    b = img._slide.resampled_array[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return img._type_dispatcher(result)


def filter_blue_pen(img, output_type="bool"):
    """
    Create a mask to filter out blue pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    result = (
        filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=60,
            green_upper_thresh=120,
            blue_lower_thresh=190,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=120,
            green_upper_thresh=170,
            blue_lower_thresh=200,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=175,
            green_upper_thresh=210,
            blue_lower_thresh=230,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=145,
            green_upper_thresh=180,
            blue_lower_thresh=210,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=37,
            green_upper_thresh=95,
            blue_lower_thresh=160,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=30,
            green_upper_thresh=65,
            blue_lower_thresh=130,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=130,
            green_upper_thresh=155,
            blue_lower_thresh=180,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=40,
            green_upper_thresh=35,
            blue_lower_thresh=85,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=30,
            green_upper_thresh=20,
            blue_lower_thresh=65,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=90,
            green_upper_thresh=90,
            blue_lower_thresh=140,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=60,
            green_upper_thresh=60,
            blue_lower_thresh=120,
        )
        & filter_blue(
            img._slide.resampled_array,
            red_upper_thresh=110,
            green_upper_thresh=110,
            blue_lower_thresh=175,
        )
    )
    return img._type_dispatcher(result, output_type)
