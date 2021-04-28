Changelog
=========

v0.2.5
------
**Bug Fix**

- `RandomTiler` coordinates selection within the binary mask. (`#256 <https://github.com/histolab/histolab/pull/256>`_)
- `LocalOtsuThreshold` filter: now it returns correct type (PIL Image). (`#258 <https://github.com/histolab/histolab/pull/258>`_)
- Coordinate definition in the scale coordinates of `RandomTiler` were reversed. (`#261 <https://github.com/histolab/histolab/pull/261>`_)

**New Features**

- Support and test for IHC-stained slides. (`#262 <https://github.com/histolab/histolab/pull/262>`_)

**Documentation**

- Extended documentations to include examples, images, and tutorials. Added IHC-stained slides in the data module. (`#232 <https://github.com/histolab/histolab/pull/232>`_)

v0.2.4
------
**Bug Fix**

- `RandomTiler` now respects the given tile size (`#243 <https://github.com/histolab/histolab/pull/243>`_)
- Use logger object instead of logging module when logging tiler updates (`#237 <https://github.com/histolab/histolab/pull/237>`_)

**New Features**

- New `masks` module to create binary masks from slides with different strategies: `BiggestTissueBoxMask` and `TissueMask` (`#234 <https://github.com/histolab/histolab/pull/234>`_)
- Refactor locate_mask to draw mask contours on the slide from an arbitrary BinaryMask object (`#248 <https://github.com/histolab/histolab/pull/248>`_)

**Breaking Changes**

- Refactor `Slide`: return thumbnail and scaled image instead of saving them (`#236 <https://github.com/histolab/histolab/pull/236>`_)

v0.2.3
------
**New Features**

- Allow pathlib.Path as Slide path parameter (`#226 <https://github.com/histolab/histolab/pull/226>`_)
- Tilers `extract` method now has `log_level` param that set the threshold level for the log messages (`#229 <https://github.com/histolab/histolab/pull/229>`_)

v0.2.2
------
**Bug Fix**

- Fix of `np_to_pil` in case float input but in a correct range (`#199 <https://github.com/histolab/histolab/pull/199>`_)
- Fix tiles extractor checking if the tile size is larger than the slide size (`#202 <https://github.com/histolab/histolab/pull/202>`_)
- Fix RandomTiler border wackiness extraction (`#203 <https://github.com/histolab/histolab/pull/203>`_)

**New Features**

- New parameter `tissue_percent` for all the tilers' to be used during the `has_enough_tissue` check (`#204 <https://github.com/histolab/histolab/pull/204>`_)
- Expose wsi properties. The `Slide.properties` returns the whole OpenSlide WSI properties (`#209 <https://github.com/histolab/histolab/pull/209>`_)
- Allow negative indexing for `slide.level` (`#210 <https://github.com/histolab/histolab/pull/210>`_)
- New Filter Protocol available (`#213 <https://github.com/histolab/histolab/pull/213>`_)

**Breaking Changes**

- Remove pen marks filter (`#201 <https://github.com/histolab/histolab/pull/201>`_)

v0.2.1
------
**Maintenance**

- Pin dependencies in requirements.txt to avoid discrepancy with scikit-image v0.18.0

v0.2.0
------

**Bug Fix**

- Bug: Fix grid tile coordinates calculation (`#186 <https://github.com/histolab/histolab/pull/186>`_)
- Bug: Fix quickstart tutorial slides' paths (`#154 <https://github.com/histolab/histolab/pull/154>`_ and `#165 <https://github.com/histolab/histolab/pull/165>`_)

**New Features**

- Add diagnostic method to locate tiles on a slide with every Tiler (`#179 <https://github.com/histolab/histolab/pull/179>`_)
- Add diagnostic method to locate the biggest tissue bounding box on a slide (`#188 <https://github.com/histolab/histolab/pull/188>`_)
- `SlideSet` is iterable and its `slides` property has been dropped (`#177 <https://github.com/histolab/histolab/pull/177>`_)

v0.1.1
------

**New Features**

- Add RgbToLab image filter (`#147 <https://github.com/histolab/histolab/pull/147>`_)
- Add Watershed segmentation filter (`#153 <https://github.com/histolab/histolab/pull/153>`_)
- Support Python 3.8 on Linux and macOS (`#151 <https://github.com/histolab/histolab/pull/151>`_)

v0.0.1
------
**Bug Fix**

- Fix save path for tiles (`#126 <https://github.com/histolab/histolab/pull/126>`_)
- Fix critical memory issue when extracting biggest tissue box (`#128 <https://github.com/histolab/histolab/pull/128>`_)

**New Features**

- Add Lambda filter (`#124 <https://github.com/histolab/histolab/pull/124>`_)
- Add ScoreTiler and RandomScorer (`#129 <https://github.com/histolab/histolab/pull/129>`_)
- Add NucleiScorer (`#132 <https://github.com/histolab/histolab/pull/132>`_)
- Add Ovarian Tissue sample in data module (`#136 <https://github.com/histolab/histolab/pull/136>`_)

v0.0.5b
-------
**Bug Fix**

- Fix issue (`#100 <https://github.com/histolab/histolab/issues/100>`_)
- Fix issue (`#108 <https://github.com/histolab/histolab/issues/108>`_)

**New Features**

- Grid Tiler (`#99 <https://github.com/histolab/histolab/issues/99>`_)


v0.0.4b
-------
**Bug Fix**

- Fix kmeans segmentation image filter default parameters
- Fix rag threshold image filter default parameters
- Fix check tissue on `Tile` to discard almost white tiles


.. toctree::
