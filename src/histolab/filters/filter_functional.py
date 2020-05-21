# # encoding: utf-8
#
# # ------------------------------------------------------------------------
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # ------------------------------------------------------------------------
#
# """Provides the Filter class.
#
# Filter is the main API class for applying filters to WSI slides images.
# """
# import math
# import os
#
# import numpy as np
# import PIL
#
# import skimage.exposure as sk_exposure
# from PIL import Image
# from src.histolab import util
#
#
# # ----------- Move to util?
# def mask_percent(img: PIL.Image.Image) -> float:
#     """Compute percentage of an image that is masked.
#
#     If the image is RGB (or RGBA), the pixels are first sum along
#     the third axis and then the pixels different from zero are counted.
#     Otherwise, count the pixels different from zero.
#
#     Parameters
#     ----------
#     img : PIL.Image.Image
#         Input image
#
#     Returns
#     -------
#     float
#         Percentage of image masked
#     """
#     img_arr = np.array(img)
#     if img.mode == "RGB" or img.mode == "RGBA":
#         squashed = np.sum(img_arr, axis=2)
#         mask_percentage = 100 - np.count_nonzero(squashed) / squashed.size * 100
#     else:
#         mask_percentage = 100 - np.count_nonzero(img_arr) / img_arr.size * 100
#     return mask_percentage
#
#
# def tissue_percent(img: PIL.Image.Image) -> float:
#     """Compute percentage of tissue in an image.
#
#         Parameters
#         ----------
#         img : PIL.Image.Image
#             Input image
#
#         Returns
#         -------
#         float
#             Percentage of image that is not masked
#         """
#     return 100 - mask_percent(img)
#
#
# # -----------
#
# ###############
#
#
# def filter_hsv_to_h(img, np_array, output_type="int"):
#     """
#     Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
#     values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
#     https://en.wikipedia.org/wiki/HSL_and_HSV
#
#     Args:
#         hsv: HSV image as a NumPy array.
#         output_type: Type of array to return (float or int).
#
#     Returns:
#         Hue values (float or int) as a 1-dimensional NumPy array.
#     """
#     h = np_array[:, :, 0]
#     h = h.flatten()
#     if output_type == "int":
#         h *= 360
#         h = h.astype("int")
#     return h
#
#
# def filter_hed_to_hematoxylin(img, np_array, output_type="uint8"):
#     """
#     Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
#     contrast.
#
#     Args:
#         np_img: HED image as a NumPy array.
#         output_type: Type of array to return (float or uint8).
#
#     Returns:
#         NumPy array for Hematoxylin channel.
#     """
#     hematoxylin = np_array[:, :, 0]
#     if output_type == "float":
#         hematoxylin = sk_exposure.rescale_intensity(hematoxylin, out_range=(0.0, 1.0))
#     else:
#         hematoxylin = (
#             sk_exposure.rescale_intensity(hematoxylin, out_range=(0, 255))
#         ).astype("uint8")
#     return hematoxylin
#
#
# def filter_hed_to_eosin(img, np_array, output_type="uint8"):
#     """
#     Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
#     contrast.
#
#     Args:
#         np_img: HED image as a NumPy array.
#         output_type: Type of array to return (float or uint8).
#
#     Returns:
#         NumPy array for Eosin channel.
#     """
#     eosin = np_array[:, :, 1]
#     if output_type == "float":
#         eosin = sk_exposure.rescale_intensity(eosin, out_range=(0.0, 1.0))
#     else:
#         eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype(
#             "uint8"
#         )
#     return eosin
#
#
# def filter_hsv_to_s(np_array):
#     """
#     Experimental HSV to S (saturation).
#
#     Args:
#         hsv:  HSV image as a NumPy array.
#
#     Returns:
#         Saturation values as a 1-dimensional NumPy array.
#     """
#     s = np_array[:, :, 1]
#     s = s.flatten()
#     return s
#
#
# def filter_hsv_to_v(np_array):
#     """
#     Experimental HSV to V (value).
#
#     Args:
#         hsv:  HSV image as a NumPy array.
#
#     Returns:
#         Value values as a 1-dimensional NumPy array.
#     """
#     v = np_array[:, :, 2]
#     v = v.flatten()
#     return v
#
#
# def convert_uint8_to_bool(np_array):
#     """
#     Convert NumPy array of uint8 (255,0) values to bool (True,False) values
#
#     Args:
#         np_img: Binary image as NumPy array of uint8 (255,0) values.
#
#     Returns:
#         NumPy array of bool (True,False) values.
#     """
#     result = (np_array / 255).astype(bool)
#     return result
#
#
# # ------------------- improve with amore
#
#
# def apply_image_filters(np_img, slide_num=None, info=None):
#     """
#     Apply filters to image as NumPy array and optionally save and/or display filtered images.
#
#     Args:
#         np_img: Image as NumPy array.
#         slide_num: The slide number (used for saving/displaying).
#         info: Dictionary of slide information (used for HTML display).
#         save: If True, save image.
#         display: If True, display image.
#
#     Returns:
#         Resulting filtered image as a NumPy array.
#     """
#     rgb = np_img
#
#     mask_not_green = filter_green_channel(rgb)
#     rgb_not_green = util.mask_rgb(rgb, mask_not_green)
#
#     mask_not_gray = filter_grays(rgb)
#     rgb_not_gray = util.mask_rgb(rgb, mask_not_gray)
#
#     mask_no_red_pen = filter_red_pen(rgb)
#     rgb_no_red_pen = util.mask_rgb(rgb, mask_no_red_pen)
#
#     mask_no_green_pen = filter_green_pen(rgb)
#     rgb_no_green_pen = util.mask_rgb(rgb, mask_no_green_pen)
#     save_display(
#         save,
#         display,
#         info,
#         rgb_no_green_pen,
#         slide_num,
#         5,
#         "No Green Pen",
#         "rgb-no-green-pen",
#     )
#
#     mask_no_blue_pen = filter_blue_pen(rgb)
#     rgb_no_blue_pen = util.mask_rgb(rgb, mask_no_blue_pen)
#     save_display(
#         save,
#         display,
#         info,
#         rgb_no_blue_pen,
#         slide_num,
#         6,
#         "No Blue Pen",
#         "rgb-no-blue-pen",
#     )
#
#     mask_gray_green_pens = (
#         mask_not_gray
#         & mask_not_green
#         & mask_no_red_pen
#         & mask_no_green_pen
#         & mask_no_blue_pen
#     )
#     rgb_gray_green_pens = util.mask_rgb(rgb, mask_gray_green_pens)
#
#     mask_remove_small = filter_remove_small_objects(
#         mask_gray_green_pens, min_size=500, output_type="bool"
#     )
#     rgb_remove_small = util.mask_rgb(rgb, mask_remove_small)
#     save_display(
#         save,
#         display,
#         info,
#         rgb_remove_small,
#         slide_num,
#         8,
#         "Not Gray, Not Green, No Pens,\nRemove Small Objects",
#         "rgb-not-green-not-gray-no-pens-remove-small",
#     )
#
#     img = rgb_remove_small
#     return img
#
#
# # ---private interface methods and properties---
#
# #
# # def _is_rgb(np_array):
# #     if np_array.ndim == 3 and np_array.ndim.shape[2] == 3:
# #         return True
# #     return False
#
#
# def _type_dispatcher(img, np_array, output_type):
#     _map = {"bool": np_array.astype("bool"), "float": np_array.astype("float")}
#     return _map.get(output_type, (255 * np_array).astype("uint8"))
#
#
# # ---------------------
#
#
# def apply_filters_to_image(slide_num, save=True, display=False):
#     """
#     Apply a set of filters to an image and optionally save and/or display filtered images.
#
#     Args:
#         slide_num: The slide number.
#         save: If True, save filtered images.
#         display: If True, display filtered images to screen.
#
#     Returns:
#         Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
#         (used for HTML page generation).
#     """
#     print("Processing slide #%d" % slide_num)
#     if save and not os.path.exists(slide.FILTER_DIR):
#         os.makedirs(slide.FILTER_DIR)
#     img_path = slide.get_training_image_path(slide_num)
#     np_orig = slide.open_image_np(img_path)
#     filtered_np_img = apply_image_filters(
#         np_orig, slide_num, save=save, display=display
#     )
#
#     if save:
#         result_path = slide.get_filter_image_result(slide_num)
#         pil_img = util.np_to_pil(filtered_np_img)
#         pil_img.save(result_path)
#
#         thumbnail_path = slide.get_filter_thumbnail_result(slide_num)
#         slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_path)
#
#     return filtered_np_img
#
#
# def mask_percentage_text(mask_percentage):
#     """
#   Generate a formatted string representing the percentage that an image is masked.
#
#   Args:
#     mask_percentage: The mask percentage.
#
#   Returns:
#     The mask percentage formatted as a string.
#   """
#     return "%3.2f%%" % mask_percentage
#
#
# def image_cell(slide_num, filter_num, display_text, file_text):
#     """
#   Generate HTML for viewing a processed image.
#
#   Args:
#     slide_num: The slide number.
#     filter_num: The filter number.
#     display_text: Filter display name.
#     file_text: Filter name for file.
#
#   Returns:
#     HTML for a table cell for viewing a filtered image.
#   """
#     filt_img = slide.get_filter_image_path(slide_num, filter_num, file_text)
#     filt_thumb = slide.get_filter_thumbnail_path(slide_num, filter_num, file_text)
#     img_name = slide.get_filter_image_filename(slide_num, filter_num, file_text)
#     return (
#         "      <td>\n"
#         + '        <a target="_blank" href="%s">%s<br/>\n' % (filt_img, display_text)
#         + '          <img src="%s" />\n' % (filt_thumb)
#         + "        </a>\n"
#         + "      </td>\n"
#     )
#
#
# def html_header(page_title):
#     """
#   Generate an HTML header for previewing images.
#
#   Returns:
#     HTML header for viewing images.
#   """
#     html = (
#         '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" '
#         + '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n'
#         + '<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">\n'
#         + "  <head>\n"
#         + "    <title>%s</title>\n" % page_title
#         + '    <style type="text/css">\n'
#         + "     img { border: 2px solid black; }\n"
#         + "     td { border: 2px solid black; }\n"
#         + "    </style>\n"
#         + "  </head>\n"
#         + "  <body>\n"
#     )
#     return html
#
#
# def html_footer():
#     """
#   Generate an HTML footer for previewing images.
#
#   Returns:
#     HTML footer for viewing images.
#   """
#     html = "</body>\n" + "</html>\n"
#     return html
#
#
# def save_filtered_image(np_img, slide_num, filter_num, filter_text):
#     """
#   Save a filtered image to the file system.
#
#   Args:
#     np_img: Image as a NumPy array.
#     slide_num:  The slide number.
#     filter_num: The filter number.
#     filter_text: Descriptive text to add to the image filename.
#   """
#     filepath = slide.get_filter_image_path(slide_num, filter_num, filter_text)
#     pil_img = util.np_to_pil(np_img)
#     pil_img.save(filepath)
#     print("%-20s |   Name: %s", filepath)
#
#     thumbnail_filepath = slide.get_filter_thumbnail_path(
#         slide_num, filter_num, filter_text
#     )
#     slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
#
#
# def generate_filter_html_result(html_page_info):
#     """
#   Generate HTML to view the filtered images. If slide.FILTER_PAGINATE is True, the results will be paginated.
#
#   Args:
#     html_page_info: Dictionary of image information.
#   """
#     if not slide.FILTER_PAGINATE:
#         html = ""
#         html += html_header("Filtered Images")
#         html += "  <table>\n"
#
#         row = 0
#         for key in sorted(html_page_info):
#             value = html_page_info[key]
#             current_row = value[0]
#             if current_row > row:
#                 html += "    <tr>\n"
#                 row = current_row
#             html += image_cell(value[0], value[1], value[2], value[3])
#             next_key = key + 1
#             if next_key not in html_page_info:
#                 html += "    </tr>\n"
#
#         html += "  </table>\n"
#         html += html_footer()
#         text_file = open(os.path.join(slide.FILTER_HTML_DIR, "filters.html"), "w")
#         text_file.write(html)
#         text_file.close()
#     else:
#         slide_nums = set()
#         for key in html_page_info:
#             slide_num = math.floor(key / 1000)
#             slide_nums.add(slide_num)
#         slide_nums = sorted(list(slide_nums))
#         total_len = len(slide_nums)
#         page_size = slide.FILTER_PAGINATION_SIZE
#         num_pages = math.ceil(total_len / page_size)
#
#         for page_num in range(1, num_pages + 1):
#             start_index = (page_num - 1) * page_size
#             end_index = (page_num * page_size) if (page_num < num_pages) else total_len
#             page_slide_nums = slide_nums[start_index:end_index]
#
#             html = ""
#             html += html_header("Filtered Images, Page %d" % page_num)
#
#             html += '  <div style="font-size: 20px">'
#             if page_num > 1:
#                 if page_num == 2:
#                     html += '<a href="filters.html">&lt;</a> '
#                 else:
#                     html += '<a href="filters-%d.html">&lt;</a> ' % (page_num - 1)
#             html += "Page %d" % page_num
#             if page_num < num_pages:
#                 html += ' <a href="filters-%d.html">&gt;</a> ' % (page_num + 1)
#             html += "</div>\n"
#
#             html += "  <table>\n"
#             for slide_num in page_slide_nums:
#                 html += "  <tr>\n"
#                 filter_num = 1
#
#                 lookup_key = slide_num * 1000 + filter_num
#                 while lookup_key in html_page_info:
#                     value = html_page_info[lookup_key]
#                     html += image_cell(value[0], value[1], value[2], value[3])
#                     lookup_key += 1
#                 html += "  </tr>\n"
#
#             html += "  </table>\n"
#
#             html += html_footer()
#             if page_num == 1:
#                 text_file = open(
#                     os.path.join(slide.FILTER_HTML_DIR, "filters.html"), "w"
#                 )
#             else:
#                 text_file = open(
#                     os.path.join(slide.FILTER_HTML_DIR, "filters-%d.html" % page_num),
#                     "w",
#                 )
#             text_file.write(html)
#             text_file.close()
#
#
# def apply_filters_to_image_list(image_num_list, save, display):
#     """
#   Apply filters to a list of images.
#
#   Args:
#     image_num_list: List of image numbers.
#     save: If True, save filtered images.
#     display: If True, display filtered images to screen.
#
#   Returns:
#     Tuple consisting of 1) a list of image numbers, and 2) a dictionary of image filter information.
#   """
#     html_page_info = dict()
#     for slide_num in image_num_list:
#         _, info = apply_filters_to_image(slide_num, save=save, display=display)
#         html_page_info.update(info)
#     return image_num_list, html_page_info
