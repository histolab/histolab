Changelog
=========

v0.2.1
------
- Pin dependencies in requirements.txt to avoid discrepancy with scikit-image v0.18.0

v0.2.0
------
- Add diagnostic method to locate tiles on a slide with every Tiler (`#179 <https://github.com/histolab/histolab/pull/179>`_)
- Add diagnostic method to locate the biggest tissue bounding box on a slide (`#188 <https://github.com/histolab/histolab/pull/188>`_)
- `SlideSet` is iterable and its `slides` property has been dropped (`#177 <https://github.com/histolab/histolab/pull/177>`_)
- Bug: Fix grid tile coordinates calculation (`#186 <https://github.com/histolab/histolab/pull/186>`_,)
- Bug: Fix quickstart tutorial slides' paths (`#154 <https://github.com/histolab/histolab/pull/154>`_ and `#165 <https://github.com/histolab/histolab/pull/165>`_)

v0.1.1
------

- Add RgbToLab image filter (`#147 <https://github.com/histolab/histolab/pull/147>`_)
- Add Watershed segmentation filter (`#153 <https://github.com/histolab/histolab/pull/153>`_)
- Support Python 3.8 on Linux and macOS (`#151 <https://github.com/histolab/histolab/pull/151>`_)

v0.0.1
------

- Add Lambda filter (`#124 <https://github.com/histolab/histolab/pull/124>`_)
- Add ScoreTiler and RandomScorer (`#129 <https://github.com/histolab/histolab/pull/129>`_)
- Add NucleiScorer (`#132 <https://github.com/histolab/histolab/pull/132>`_)
- Add Ovarian Tissue sample in data module (`#136 <https://github.com/histolab/histolab/pull/136>`_)
- Fix tiles's save path (`#126 <https://github.com/histolab/histolab/pull/126>`_)
- Fix critical memory issue when extracting biggest tissue box (`#128 <https://github.com/histolab/histolab/pull/128>`_)

v0.0.5b
-------

- Fix `issue #100 <https://github.com/histolab/histolab/issues/100>`_
- Fix `issue #108 <https://github.com/histolab/histolab/issues/108>`_
- `Grid Tiler <https://github.com/histolab/histolab/issues/99>`_ added


v0.0.4b
-------

- Fix kmeans segmentation image filter default parameters
- Fix rag threshold image filter default parameters
- Fix check tissue on `Tile` to discard almost white tiles


.. toctree::
