[![Coverage Status](https://coveralls.io/repos/github/histolab/histolab/badge.svg?branch=master)](https://coveralls.io/github/histolab/histolab?branch=master)
[![Build Status](https://travis-ci.com/histolab/histolab.svg?branch=master)](https://travis-ci.com/histolab/histolab)
[![Documentation Status](https://readthedocs.org/projects/histolab/badge/?version=latest)](https://histolab.readthedocs.io/en/latest/?badge=latest)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/histolab/histolab.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/histolab/histolab/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/histolab/histolab.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/histolab/histolab/context:python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/histolab)](https://pypi.org/project/histolab/)
[![GitHub](https://img.shields.io/github/license/histolab/histolab)](https://github.com/histolab/histolab/blob/master/LICENSE.txt)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/histolab)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/histolab)

<img width="390" alt="histolab" src="https://user-images.githubusercontent.com/4196091/84828232-048fcc00-b026-11ea-8caa-5c14bb8565bd.png">


## Table of Contents

* [Motivation](#motivation)
 
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
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

Histolab has only one sistem-wide dependency: OpenSlide.

You can download and install it from [OpenSlide](https://openslide.org/download/) according to your operating system.


### Installation 

```
pip install histolab
```

### Documentation

Read the full documentation here https://histolab.readthedocs.io/en/latest/.

### Quickstart 

```python
from histolab.data import breast_tissue, heart_tissue
```

**NB** To use the data module, you need to install ```pooch```.

Each data function outputs the corresponding slide as an OpenSlide object, and the path where the slide has been saved:


```python
breast_svs, breast_path = breast_tissue()
heart_svs, heart_path = heart_tissue()
```

### Slide


```python
from histolab.slide import Slide
```

Convert the slide into a ```Slide``` object. ```Slide``` takes as input the path where the slide is stored and the ```processed_path``` where the thumbnail and the tiles will be saved.


```python
breast_slide = Slide(breast_path, processed_path='processed')
heart_slide = Slide(heart_path, processed_path='processed')
```

As a ```Slide``` object, you can now easily retrieve information about the slide, such as the slide name, the dimensions at native magnification, the dimensions at a specified level, save and show the slide thumbnail, or get a scaled version of the slide.


```python
print(f"Slide name: {breast_slide.name}")
print(f"Dimensions at level 0: {breast_slide.dimensions}")
print(f"Dimensions at level 1: {breast_slide.level_dimensions(level=1)}")
print(f"Dimensions at level 2: {breast_slide.level_dimensions(level=2)}")
```

    Slide name: 9c960533-2e58-4e54-97b2-8454dfb4b8c8
    Dimensions at level 0: (96972, 30681)
    Dimensions at level 1: (24243, 7670)
    Dimensions at level 2: (6060, 1917)



```python
print(f"Slide name: {heart_slide.name}")
print(f"Dimensions at level 0: {heart_slide.dimensions}")
print(f"Dimensions at level 1: {heart_slide.level_dimensions(level=1)}")
print(f"Dimensions at level 2: {heart_slide.level_dimensions(level=2)}")
```

    Slide name: JP2K-33003-2
    Dimensions at level 0: (32671, 47076)
    Dimensions at level 1: (8167, 11769)
    Dimensions at level 2: (2041, 2942)



```python
breast_slide.save_thumbnail()
print(f"Thumbnails saved at: {breast_slide.thumbnail_path}") 
heart_slide.save_thumbnail()

print(f"Thumbnails saved at: {heart_slide.thumbnail_path}") 
```

    Thumbnails saved at: processed/thumbnails/9c960533-2e58-4e54-97b2-8454dfb4b8c8.png
    Thumbnails saved at: processed/thumbnails/JP2K-33003-2.png



```python
breast_slide.show() 
heart_slide.show()
```

![thumbnails](https://user-images.githubusercontent.com/31658006/84955475-a4695a80-b0f7-11ea-83d5-db7668801219.png)

### Tiles extraction

Now that your ```Slide``` object is defined, you can automatically extract the tiles. A ```RandomTiler``` object crops random tiles from the slide.
You need to specify the size you want your tiles, the number of tiles to crop, and the level of magnification. If ```check_tissue``` is True, the exracted tiles are taken by default from the **biggest tissue region detected** in the slide, and the tiles are saved only if they have at least 80% of tissue inside.


```python
from histolab.tiler import RandomTiler

random_tiles_extractor = RandomTiler(
    tile_size=(512, 512),
    n_tiles=6,
    level=2,
    seed=42,
    check_tissue=True,
    prefix="processed/breast_slide/",
)

random_tiles_extractor.extract(breast_slide)
```

    	 Tile 0 saved: processed/breast_slide/tile_0_level2_70536-7186-78729-15380.png
    	 Tile 1 saved: processed/breast_slide/tile_1_level2_74393-3441-82586-11635.png
    	 Tile 2 saved: processed/breast_slide/tile_2_level2_82218-6225-90411-14420.png
    	 Tile 3 saved: processed/breast_slide/tile_3_level2_84026-8146-92219-16340.png
    	 Tile 4 saved: processed/breast_slide/tile_4_level2_78969-3953-87162-12147.png
    	 Tile 5 saved: processed/breast_slide/tile_5_level2_78649-3569-86842-11763.png
    	 Tile 6 saved: processed/breast_slide/tile_6_level2_81994-6753-90187-14948.png
    6 Random Tiles have been saved.


![breast 001](https://user-images.githubusercontent.com/31658006/84955724-0f1a9600-b0f8-11ea-92c9-3236dd16bca8.png)

```python
random_tiles_extractor = RandomTiler(
    tile_size=(512, 512),
    n_tiles=6,
    level=0,
    seed=42,
    check_tissue=True,
    prefix="processed/heart_slide/",
)
random_tiles_extractor.extract(heart_slide)
```

    	 Tile 0 saved: processed/heart_slide/tile_0_level0_4299-35755-4811-36267.png
    	 Tile 1 saved: processed/heart_slide/tile_1_level0_7051-39146-7563-39658.png
    	 Tile 2 saved: processed/heart_slide/tile_2_level0_10920-26934-11432-27446.png
    	 Tile 3 saved: processed/heart_slide/tile_3_level0_7151-30986-7663-31498.png
    	 Tile 4 saved: processed/heart_slide/tile_4_level0_11472-26400-11984-26912.png
    	 Tile 5 saved: processed/heart_slide/tile_5_level0_13489-42680-14001-43192.png
    	 Tile 6 saved: processed/heart_slide/tile_6_level0_13281-33895-13793-34407.png
    6 Random Tiles have been saved.

![heart](https://user-images.githubusercontent.com/31658006/84955793-2c4f6480-b0f8-11ea-8970-592dc992d56d.png)

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
