Changelog
=========

v0.3.0
------
**Bug Fix**

- Fix `GridTiler`'s `_are_coordinates_within_extraction_mask` method where tile coordinates are off by 1 or 2 pixels due to conversion of floats to ints. (`#308 <https://github.com/histolab/histolab/pull/308>`_)
- Fix the mismatch between row-column / X-Y coordinates in the RandomTiler (`#317 <https://github.com/histolab/histolab/pull/317>`_)
- Fix return type of RGB to LAB filter. (`#323 <https://github.com/histolab/histolab/pull/323>`_)
- Filter `kmeans_segmentation` is now applied only to RGB images. (`#328 <https://github.com/histolab/histolab/pull/328>`_)
- Conversion from RGB to HED preserves HED color space range (`#334 <https://github.com/histolab/histolab/pull/334>`_)
- Conversion from RGB to HSV preserves HSV color space range (`#337 <https://github.com/histolab/histolab/pull/337>`_)
- Remove HSV and YCBCR references in wrong value range in tests (`#343 <https://github.com/histolab/histolab/pull/343>`_)

**New Features**

- Add RGB to OD filter. (`#290 <https://github.com/histolab/histolab/pull/290>`_ and `#331 <https://github.com/histolab/histolab/pull/331>`_)
- Add method dispatcher compatible with older Python versions. (`#312 <https://github.com/histolab/histolab/pull/312>`_)
- Add LAB to RGB filter. (`#323 <https://github.com/histolab/histolab/pull/323>`_)
- Finer control of `locate_tiles` (pass tiles to avoid re-extraction and color tiles' border individually). (`#304 <https://github.com/histolab/histolab/pull/304>`_)
- Add `TissueMask` mask for `Tile` with type dispatcher. (`#313 <https://github.com/histolab/histolab/pull/313>`_)
- Add conversion level - magnification factor in `Slide`. (`#319 <https://github.com/histolab/histolab/pull/319>`_)
- Add `CellularityScorer`. (`#320 <https://github.com/histolab/histolab/pull/320>`_)

**Maintenance**

- Link automatically issues in PR template. (`#291 <https://github.com/histolab/histolab/pull/291>`_)
- Include histolab version in issue template. (`#296 <https://github.com/histolab/histolab/pull/296>`_)
- Add security linter with Bandit in pre commit and CI. (`#316 <https://github.com/histolab/histolab/pull/316>`_)
- Get rid of `src` directory in favor of `histolab` dir within the root. (`#324 <https://github.com/histolab/histolab/pull/324>`_)
- Use Python 3.9 for benchmarks. (`#342 <https://github.com/histolab/histolab/pull/342>`_)

**Dependencies**

- Support scikit-image 0.18.3. (`#196 <https://github.com/histolab/histolab/pull/196>`_, `#200 <https://github.com/histolab/histolab/pull/200>`_ and `#327 <https://github.com/histolab/histolab/pull/327>`_)
- Support scipy 1.7.1. (`#305 <https://github.com/histolab/histolab/pull/305>`_)
- Upgrade sphinx to 4.2.0 to fix incompatibility with docutils 0.18. (`#339 <https://github.com/histolab/histolab/pull/339>`_)
- Support numpy 1.21.4. (`#344 <https://github.com/histolab/histolab/pull/344>`_)

**Documentation**

- Fix docs links in `tissue_mask` module. (`#321 <https://github.com/histolab/histolab/pull/331>`_)
- Add note on data module for TCGA example data not available. (`#325 <https://github.com/histolab/histolab/pull/325>`_ and `#333 <https://github.com/histolab/histolab/pull/333>`_)

v0.2.6
------
**Bug Fix**

- Fix ``polygon_to_mask_array`` return mask shape. (`#268 <https://github.com/histolab/histolab/pull/268>`_)
- Fix overlapping extraction grids in ``GridTiler``. (`#270 <https://github.com/histolab/histolab/pull/270>`_)

**New Features**

- Add DAB filter. (`#277 <https://github.com/histolab/histolab/pull/277>`_)
- Allow slide name to contain dot. (`#281 <https://github.com/histolab/histolab/pull/281>`_)

**Documentation**

- Docs fixes about Slide's processed_path. (`#276 <https://github.com/histolab/histolab/pull/276>`_)
- Add instructions on how to install Pixman 0.40. (`#280 <https://github.com/histolab/histolab/pull/280>`_)

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
