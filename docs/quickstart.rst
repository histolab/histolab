Quick Start
====================

This Quick Start tutorial will guide you with the basic concepts of histolab, such as interacting with Slides and automatically generating tiles from them.

Install
~~~~~~~

If you haven’t already, install the ``histolab`` library:

.. code-block:: bash

   pip install histolab


First things first, let's import some data to work with, for example a breast tissue slide and a heart tissue slide:


.. code:: ipython3

    from histolab.data import breast_tissue, heart_tissue

.. note:: 
    To use the ``data`` module, you need to install ``pooch``.

Each data function outputs the corresponding slide as an OpenSlide object, and the path where the slide has been saved:

.. code:: ipython3

    breast_svs, breast_path = breast_tissue()
    heart_svs, heart_path = heart_tissue()

Slide
~~~~~

.. code:: ipython3

    from histolab.slide import Slide

Convert the slide into a ``Slide`` object. ``Slide`` takes as input the
path where the slide is stored and the ``processed_path`` where the
thumbnail and the tiles will be saved.

.. code:: ipython3

    breast_slide = Slide(breast_path, processed_path='processed')
    heart_slide = Slide(heart_path, processed_path='processed')

With a ``Slide`` object you can now easily retrieve information about the
slide, such as the slide name, the dimensions at native magnification,
the dimensions at a specified level, save and show the slide thumbnail,
or get a scaled version of the slide.

.. code:: ipython3

    print(f"Slide name: {breast_slide.name}")
    print(f"Dimensions at level 0: {breast_slide.dimensions}")
    print(f"Dimensions at level 1: {breast_slide.level_dimensions(level=1)}")
    print(f"Dimensions at level 2: {breast_slide.level_dimensions(level=2)}")


.. parsed-literal::

    Slide name: 9c960533-2e58-4e54-97b2-8454dfb4b8c8
    Dimensions at level 0: (96972, 30681)
    Dimensions at level 1: (24243, 7670)
    Dimensions at level 2: (6060, 1917)


.. code:: ipython3

    print(f"Slide name: {heart_slide.name}")
    print(f"Dimensions at level 0: {heart_slide.dimensions}")
    print(f"Dimensions at level 1: {heart_slide.level_dimensions(level=1)}")
    print(f"Dimensions at level 2: {heart_slide.level_dimensions(level=2)}")


.. parsed-literal::

    Slide name: JP2K-33003-2
    Dimensions at level 0: (32671, 47076)
    Dimensions at level 1: (8167, 11769)
    Dimensions at level 2: (2041, 2942)


.. code:: ipython3

    breast_slide.save_thumbnail()
    print(f"Thumbnails saved at: {breast_slide.thumbnail_path}") 
    heart_slide.save_thumbnail()
    
    print(f"Thumbnails saved at: {heart_slide.thumbnail_path}") 


.. parsed-literal::

    Thumbnails saved at: processed/thumbnails/9c960533-2e58-4e54-97b2-8454dfb4b8c8.png
    Thumbnails saved at: processed/thumbnails/JP2K-33003-2.png


.. code:: ipython3

    breast_slide.show() 
    heart_slide.show()

.. figure::  https://user-images.githubusercontent.com/31658006/84955475-a4695a80-b0f7-11ea-83d5-db7668801219.png
   :alt: caption


Tiles extraction
~~~~~~~~~~~~~~~~

Now that your slide object is defined, you can automatically extract the
tiles. ``RandomTiler`` method crops random tiles from the slide. You
need to specify the size you want your tiles, the number of tiles to
crop, and the level of magnification. If ``check_tissue`` is True, the
extracted tiles are taken by default from the **biggest tissue region
detected** in the slide, and the tiles are saved only if they have at
least 80% of tissue inside.

.. code:: ipython3

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


.. parsed-literal::

    	 Tile 0 saved: processed/breast_slide/tile_0_level2_70536-7186-78729-15380.png
    	 Tile 1 saved: processed/breast_slide/tile_1_level2_74393-3441-82586-11635.png
    	 Tile 2 saved: processed/breast_slide/tile_2_level2_82218-6225-90411-14420.png
    	 Tile 3 saved: processed/breast_slide/tile_3_level2_84026-8146-92219-16340.png
    	 Tile 4 saved: processed/breast_slide/tile_4_level2_78969-3953-87162-12147.png
    	 Tile 5 saved: processed/breast_slide/tile_5_level2_78649-3569-86842-11763.png
    	 Tile 6 saved: processed/breast_slide/tile_6_level2_81994-6753-90187-14948.png
    6 Random Tiles have been saved.


.. figure:: https://user-images.githubusercontent.com/31658006/84955724-0f1a9600-b0f8-11ea-92c9-3236dd16bca8.png
   :alt: caption


.. code:: ipython3

    random_tiles_extractor = RandomTiler(
        tile_size=(512, 512),
        n_tiles=6,
        level=0,
        seed=42,
        check_tissue=True,
        prefix="processed/heart_slide/",
    )
    
    random_tiles_extractor.extract(heart_slide)


.. parsed-literal::

    	 Tile 0 saved: processed/heart_slide/tile_0_level0_4299-35755-4811-36267.png
    	 Tile 1 saved: processed/heart_slide/tile_1_level0_7051-39146-7563-39658.png
    	 Tile 2 saved: processed/heart_slide/tile_2_level0_10920-26934-11432-27446.png
    	 Tile 3 saved: processed/heart_slide/tile_3_level0_7151-30986-7663-31498.png
    	 Tile 4 saved: processed/heart_slide/tile_4_level0_11472-26400-11984-26912.png
    	 Tile 5 saved: processed/heart_slide/tile_5_level0_13489-42680-14001-43192.png
    	 Tile 6 saved: processed/heart_slide/tile_6_level0_13281-33895-13793-34407.png
    6 Random Tiles have been saved.


.. figure:: https://user-images.githubusercontent.com/31658006/84955793-2c4f6480-b0f8-11ea-8970-592dc992d56d.png
   :alt: caption


.. toctree::

.. toctree::