[![Coverage Status](https://coveralls.io/repos/github/MPBA/histolab/badge.svg?branch=master)](https://coveralls.io/github/MPBA/histolab?branch=master)
[![Build Status](https://travis-ci.com/MPBA/histolab.svg?branch=master)](https://travis-ci.com/MPBA/histolab)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/MPBA/histolab.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MPBA/histolab/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MPBA/histolab.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MPBA/histolab/context:python)
# HistoLab

### NOTE: WORK IN PROGRESS PROJECT

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
The histo-pathological analysis of tissue sections is the gold standard to assess the presence  of many complex diseases, such as tumors, and understand their nature. 
In daily practice, pathologists usually perform microscopy examination of tissue slides considering a limited number of regions and the clinical evaulation relies on several factors such as nuclei morphology, cell distribution, and color (staining): this process is time consuming, could lead to information loss, and suffers from inter-observer variability.

The advent of digital pathology is changing the way patholgists work and collaborate, and has opened the way to a new era in computational pathology. In particular, histopathology is expected to be at the center of the AI revolution in medicine [1], prevision supported by the increasing success of deep learning applications to digital pathology.

Whole Slide Images (WSIs), namely the translation of tissue slides from glass to digital format, are a great source of information from both a medical and a computational point of view. WSIs can be coloured with different staining techniques (e.g. H&E or IHC), and are usually very large in size (up to several GB per slide). Because of WSIs typical pyramidal structure, images can be retrieved at different magnification factors, providing a further layer of information beyond color.

However, processing WSIs is far from being trivial. First of all, WSIs can be stored in different proprietary formats, according to the scanner used to digitalize the slides, and a standard protocol is still missing. WSIs can also present artifacts, such as shadows, mold, or annotations (pen marks) that are not useful. Moreover, giving their dimensions, it is not possible to process a WSI all at once, or, for example, to feed a neural network: it is necessary to crop smaller regions of tissues (tiles), which in turns require a tissue detection step.  

The aim of this project is to provide a tool for WSI processing in a reproducible environment to support clinical and scientific research. HistoLab is designed to handle WSIs, automatically detect the tissue, and retrieve informative tiles, and it can thus be integrated in a deep learning pipeline.

## Getting Started

### Prerequisites

HistoLab has only one sistem-wide dependency: OpenSlide.

You can download and install it from [OpenSlide](https://openslide.org/download/) according to your operating system.


### Installation

```
pip install histolab
```

### Quickstart

COMING SOON!

## Versioning

We use [PEP 440](https://www.python.org/dev/peps/pep-0440/) for versioning. 

## Authors

* **[Alessia Marcolini](https://github.com/alessiamarcolini)** 
* **[Ernesto Arbitrio](https://github.com/ernestoarbitrio)**
* **[Nicole Bussola](https://gitlab.fbk.eu/bussola)**


## License

This project is licensed under `Apache License  Version 2.0` - see the [LICENSE.txt](https://github.com/MPBA/histolab/blob/master/LICENSE.txt) file for details

## Roadmap

[Open issues](https://github.com/MPBA/histolab/issues)

## Acknowledgements

* [https://github.com/deroneriksson](https://github.com/deroneriksson)

## References
[1] Colling, Richard, et al. "Artificial intelligence in digital pathology: A roadmap to routine use in clinical practice." The Journal of pathology 249.2 (2019)

## Contribution guidelines
If you want to contribute to Histolab, be sure to review the [contribution guidelines](CONTRIBUTING.md)