[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![Coverage Status](https://coveralls.io/repos/github/histolab/histolab/badge.svg?branch=master&kill_cache=1)](https://coveralls.io/github/histolab/histolab?branch=master)![CI](https://github.com/histolab/histolab/workflows/CI/badge.svg?branch=master)
[![Documentation Status](https://readthedocs.org/projects/histolab/badge/?version=latest)](https://histolab.readthedocs.io/en/latest/?badge=latest)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/histolab/histolab.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/histolab/histolab/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/histolab/histolab.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/histolab/histolab/context:python)
<a href="https://www.code-inspector.com"><img src="https://www.code-inspector.com/project/13419/score/svg?kill_cache=1"></a>
<a href="https://www.code-inspector.com"><img src="https://www.code-inspector.com/project/13419/status/svg?kill_cache=1"></a>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/histolab)](https://pypi.org/project/histolab/)
[![GitHub](https://img.shields.io/github/license/histolab/histolab)](https://github.com/histolab/histolab/blob/master/LICENSE.txt)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/histolab)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/histolab)

**Compatibility Details**

| Operating System  | Python version  |
|-------------------|-----------------|
|  Linux            | 3.6, 3.7, 3.8 |
|  MacOs            | 3.6, 3.7, 3.8 |
|  Windows          | 3.6, 3.7       |

<img width="390" alt="histolab" src="https://user-images.githubusercontent.com/4196091/84828232-048fcc00-b026-11ea-8caa-5c14bb8565bd.png">


## Table of Contents

* [Motivation](#motivation)
 
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Quickstart](#quickstart)
* [Versioning](#versioning)
* [Authors](#authors)
* [License](#license)
* [Roadmap](#roadmap)
* [Acknowledgements](#acknowledgements)
* [References](#references)
* [Contribution guidelines](#contribution-guidelines)


## Motivation 
The histo-pathological analysis of tissue sections is the gold standard to assess the presence of many complex diseases, such as tumors, and understand their nature. 
In daily practice, pathologists usually perform microscopy examination of tissue slides considering a limited number of regions and the clinical evaulation relies on several factors such as nuclei morphology, cell distribution, and color (staining): this process is time consuming, could lead to information loss, and suffers from inter-observer variability.

The advent of digital pathology is changing the way patholgists work and collaborate, and has opened the way to a new era in computational pathology. In particular, histopathology is expected to be at the center of the AI revolution in medicine [1], prevision supported by the increasing success of deep learning applications to digital pathology.

Whole Slide Images (WSIs), namely the translation of tissue slides from glass to digital format, are a great source of information from both a medical and a computational point of view. WSIs can be coloured with different staining techniques (e.g. H&E or IHC), and are usually very large in size (up to several GB per slide). Because of WSIs typical pyramidal structure, images can be retrieved at different magnification factors, providing a further layer of information beyond color.

However, processing WSIs is far from being trivial. First of all, WSIs can be stored in different proprietary formats, according to the scanner used to digitalize the slides, and a standard protocol is still missing. WSIs can also present artifacts, such as shadows, mold, or annotations (pen marks) that are not useful. Moreover, giving their dimensions, it is not possible to process a WSI all at once, or, for example, to feed a neural network: it is necessary to crop smaller regions of tissues (tiles), which in turns require a tissue detection step.  

The aim of this project is to provide a tool for WSI processing in a reproducible environment to support clinical and scientific research. Histolab is designed to handle WSIs, automatically detect the tissue, and retrieve informative tiles, and it can thus be integrated in a deep learning pipeline.

## Getting Started 

### Prerequisites 

Histolab has only one system-wide dependency: OpenSlide.

You can download and install it from [OpenSlide](https://openslide.org/download/) according to your operating system.

### Documentation

Read the full documentation here https://histolab.readthedocs.io/en/latest/.

# Quickstart 
Here we present a step-by-step tutorial on the use of `histolab` to
extract a tile dataset from example WSIs. The corresponding Jupyter
Notebook is available at <https://github.com/histolab/histolab-box>:
this repository contains a complete `histolab` environment that can be
used through [Docker](http://www.docker.com) on all platforms.

Thus, the user can decide either to use `histolab` through
`histolab-box` or installing it in his/her python virtual environment
(using conda, pipenv, pyenv, virtualenv, etc...). In the latter case, as
the `histolab` package has been published on ([PyPi](http://www.pypi.org)), 
it can be easily installed via the command:

```
pip install histolab
```

## TCGA data

First things first, let’s import some data to work with, for example the
prostate tissue slide and the ovarian tissue slide available in the
`data` module:

```python
from histolab.data import prostate_tissue, ovarian_tissue
```

**Note:** To use the `data` module, you need to install `pooch`, also
available on PyPI (<https://pypi.org/project/pooch/>). This step is
needless if we are using the Vagrant/Docker virtual environment.

The calling to a `data` function will automatically download the WSI
from the corresponding repository and save the slide in a cached
directory:

```python
prostate_svs, prostate_path = prostate_tissue()
ovarian_svs, ovarian_path = ovarian_tissue()
```

Notice that each `data` function outputs the corresponding slide, as an
OpenSlide object, and the path where the slide has been saved.

## Slide initialization

`histolab` maps a WSI file into a `Slide` object. Each usage of a WSI
requires a 1-o-1 association with a `Slide` object contained in the
`slide` module:

```python
from histolab.slide import Slide
```

To initialize a Slide it is necessary to specify the WSI path, and the
`processed_path` where the thumbnail and the tiles will be saved. In our
example, we want the `processed_path` of each slide to be a subfolder of
the current working directory:

```python
import os

BASE_PATH = os.getcwd()

PROCESS_PATH_PROSTATE = os.path.join(BASE_PATH, 'prostate', 'processed')
PROCESS_PATH_OVARIAN = os.path.join(BASE_PATH, 'ovarian', 'processed')

prostate_slide = Slide(prostate_path, processed_path=PROCESS_PATH_PROSTATE)
ovarian_slide = Slide(ovarian_path, processed_path=PROCESS_PATH_OVARIAN)
```

**Note:** If the slides were stored in the same folder, this can be done
directly on the whole dataset by using the `SlideSet` object of the
`slide` module.

With a `Slide` object we can easily retrieve information about the
slide, such as the slide name, the number of available levels, the
dimensions at native magnification or at a specified level:

```python
print(f"Slide name: {prostate_slide.name}")
print(f"Levels: {prostate_slide.levels}")
print(f"Dimensions at level 0: {prostate_slide.dimensions}")
print(f"Dimensions at level 1: {prostate_slide.level_dimensions(level=1)}")
print(f"Dimensions at level 2: {prostate_slide.level_dimensions(level=2)}")
```

```
Slide name: 6b725022-f1d5-4672-8c6c-de8140345210
Levels: [0, 1, 2]
Dimensions at level 0: (16000, 15316)
Dimensions at level 1: (4000, 3829)
Dimensions at level 2: (2000, 1914)
```

```python
print(f"Slide name: {ovarian_slide.name}")
print(f"Levels: {ovarian_slide.levels}")
print(f"Dimensions at level 0: {ovarian_slide.dimensions}")
print(f"Dimensions at level 1: {ovarian_slide.level_dimensions(level=1)}")
print(f"Dimensions at level 2: {ovarian_slide.level_dimensions(level=2)}")
```

```
Slide name: b777ec99-2811-4aa4-9568-13f68e380c86
Levels: [0, 1, 2]
Dimensions at level 0: (30001, 33987)
Dimensions at level 1: (7500, 8496)
Dimensions at level 2: (1875, 2124)
```

Moreover, we can save and show the slide thumbnail in a separate window.
In particular, the thumbnail image will be automatically saved in a
subdirectory of the `processed_path`:

```python
prostate_slide.save_thumbnail()
prostate_slide.show()
```

![](https://user-images.githubusercontent.com/4196091/92748324-5033e680-f385-11ea-812b-6a9a225ceca4.png)

```python
ovarian_slide.save_thumbnail()
ovarian_slide.show()
```

![](https://user-images.githubusercontent.com/4196091/92748248-3db9ad00-f385-11ea-846b-a5ce8cf3ca09.png)

## Tile extraction

Once that the `Slide` objects are defined, we can proceed to extract the
tiles. To speed up the extraction process, `histolab` automatically
detects the tissue region with the largest connected area and crops the
tiles within this field. The `tiler` module implements different
strategies for the tiles extraction and provides an intuitive interface
to easily retrieve a tile dataset suitable for our task. In particular,
each extraction method is customizable with several common parameters:

-   `tile_size`: the tile size;
-   `level`: the extraction level (from 0 to the number of available
    levels);
-   `check_tissue`: if a minimum percentage of tissue is required to
    save the tiles (default is 80%);
-   `prefix`: a prefix to be added at the beginning of the tiles’
    filename (default is the empty string);
-   `suffix`: a suffix to be added to the end of the tiles’ filename
    (default is `.png`).

### Random Extraction

The simplest approach we may adopt is to randomly crop a fixed number of
tiles from our slides; in this case, we need the `RandomTiler`
extractor:

```python
from histolab.tiler import RandomTiler
```

Let us suppose that we want to randomly extract 6 squared tiles at level
2 of size 512 from our prostate slide, and that we want to save them only
if they have at least 80% of tissue inside. We then initialize our
`RandomTiler` extractor as follows:

```python
random_tiles_extractor = RandomTiler(
    tile_size=(512, 512),
    n_tiles=6,
    level=2,
    seed=42,
    check_tissue=True, # default
    prefix="random/", # save tiles in the "random" subdirectory of slide's processed_path
    suffix=".png" # default
)
```

Notice that we also specify the random seed to ensure the
reproducibility of the extraction process. Starting the extraction is as
simple as calling the `extract` method on the extractor, passing the
slide as parameter:

```python
random_tiles_extractor.extract(prostate_slide)
```

![](https://user-images.githubusercontent.com/4196091/92750145-1663df80-f387-11ea-8d98-7794eef2fd47.png)

Random tiles extracted from the prostate slide at level 2.

### Grid Extraction

Instead of picking tiles at random, we may want to retrieve all the
tiles available. The Grid Tiler extractor crops the tiles following a grid
structure on the largest tissue region detected in the WSI:

```python
from histolab.tiler import GridTiler
```

In our example, we want to extract squared tiles at level 0 of size 512
from our ovarian slide, independently of the amount of tissue detected.
By default, tiles will not overlap, namely the parameter defining the
number of overlapping pixels between two adjacent tiles,
`pixel_overlap`, is set to zero:

```python
grid_tiles_extractor = GridTiler(
   tile_size=(512, 512),
   level=0,
   check_tissue=False,
   pixel_overlap=0, # default
   prefix="grid/", # save tiles in the "grid" subdirectory of slide's processed_path 
   suffix=".png" # default
)
```

Again, the extraction process starts when the extract method is called
on our extractor:

```python
grid_tiles_extractor.extract(ovarian_slide)
```

![](https://user-images.githubusercontent.com/4196091/92751173-0993bb80-f388-11ea-9d30-a6cd17769d76.png)

Examples of non-overlapping grid tiles extracted from the ovarian slide
at level 0.

### Score-based extraction

Depending on the task we will use our tile dataset for, the extracted
tiles may not be equally informative. The `ScoreTiler` allows us to save
only the "best" tiles, among all the ones extracted with a grid
structure, based on a specific scoring function. For example, let us
suppose that our goal is the detection of mitotic activity on our
ovarian slide. In this case, tiles with a higher presence of nuclei are
preferable over tiles with few or no nuclei. We can leverage the
`NucleiScorer` function of the `scorer` module to order the extracted
tiles based on the proportion of the tissue and of the hematoxylin
staining. In particular, the score is computed as ![formula](https://render.githubusercontent.com/render/math?math=N_t\cdot\mathrm{tanh}(T_t)) where ![formula](https://render.githubusercontent.com/render/math?math=N_t) is the percentage of nuclei and  ![formula](https://render.githubusercontent.com/render/math?math=T_t) the percentage of tissue in the tile *t*

First, we need the extractor and the scorer:

```python
from histolab.tiler import ScoreTiler
from histolab.scorer import NucleiScorer
```

As the `ScoreTiler` extends the `GridTiler` extractor, we also set the
`pixel_overlap` as additional parameter. Moreover, we can specify the
number of the top tiles we want to save with the `n_tile` parameter:

```python
scored_tiles_extractor = ScoreTiler(
    scorer = NucleiScorer(),
    tile_size=(512, 512),
    n_tiles=100,
    level=0,
    check_tissue=True,
    pixel_overlap=0, # default
    prefix="scored/", # save tiles in the "scored" subdirectory of slide's processed_path 
    suffix=".png" # default
)
```

Finally, when we extract our cropped images, we can also write a report
of the saved tiles and their scores in a CSV file:

```python
summary_filename = "summary_ovarian_tiles.csv"
SUMMARY_PATH = os.path.join(ovarian_slide.processed_path, summary_filename)

scored_tiles_extractor.extract(ovarian_slide, report_path=SUMMARY_PATH)
```

<img src="https://user-images.githubusercontent.com/4196091/92751801-9d658780-f388-11ea-8132-5d0c82bb112b.png" width=500>

Representation of the score assigned to each extracted tile by the
`NucleiScorer`, based on the amount of nuclei detected.

## Versioning 

We use [PEP 440](https://www.python.org/dev/peps/pep-0440/) for versioning. 

## Authors 

* **[Alessia Marcolini](https://github.com/alessiamarcolini)** 
* **[Ernesto Arbitrio](https://github.com/ernestoarbitrio)**
* **[Nicole Bussola](https://gitlab.fbk.eu/bussola)**


## License 

This project is licensed under `Apache License  Version 2.0` - see the [LICENSE.txt](https://github.com/histolab/histolab/blob/master/LICENSE.txt) file for details

## Roadmap 

[Open issues](https://github.com/histolab/histolab/issues)

## Acknowledgements 

* [https://github.com/deroneriksson](https://github.com/deroneriksson)

## References
[1] Colling, Richard, et al. "Artificial intelligence in digital pathology: A roadmap to routine use in clinical practice." The Journal of pathology 249.2 (2019)

## Contribution guidelines
If you want to contribute to Histolab, be sure to review the [contribution guidelines](CONTRIBUTING.md)
