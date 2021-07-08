Installation
============

histolab has only one system-wide dependency: ``OpenSlide``.

You can download and install it from `<https://openslide.org/download/>`_ according to your operating system.


.. warning:: There is a known bug in Pixman versions ``0.38.*`` that causes OpenSlide to produce images with black boxes for large images. See issue https://github.com/openslide/openslide/issues/291 for reference. Install version 0.40 (latest) depending on your operating system.

Install Pixman 0.40 on Ubuntu
*****************************

Ubuntu 21.04
------------
If you are running histolab on Ubuntu 21.04 you probably already have Pixman 0.40 and you are all set, go have fun |:sunglasses:|


Ubuntu 20.04 LTS
----------------
If you are using a conda environment, it is sufficient to run:

.. prompt:: text
    :prompts: $

    conda install -c anaconda pixman==0.40

Otherwise, to force a good version of ``libpixman`` to be loaded before a bad version, you need to exploit the LD_PRELOAD mechanism on Linux.
Make sure you have that file installed first.

.. prompt:: text
    :prompts: $

    export LD_PRELOAD=/path/of/libpixman-1.so.0.34.0:$LD_PRELOAD

If necessary, build ``libpixman`` from source. It's an easy build since it doesn't have any dependencies:

.. prompt:: text
    :prompts: $

    wget https://cairographics.org/releases/pixman-0.40.0.tar.gz
    tar -xvf pixman-0.40.0.tar.gz
    cd pixman-0.40.0
    ./configure
    make
    sudo make install

Install Pixman 0.40 on Windows
******************************

``Openslide`` builds are the same for all Windows versions and they include pixman 0.34.

Pixman 0.40 can be retrieved using ``pacman`` (the package manager of Arch Linux, see `https://www.msys2.org/`_ for more info):

.. prompt:: text
    :prompts: $

    pacman -S mingw-w64-x86_64-pixman
