# Create a leakage-free dataset of tiles for GTEx

`extract_tile_pw_gtex.py` is a Python script proposed as reference to retrieve a reproducible dataset of tiles using a collection of WSIs from the [GTEx](https://gtexportal.org/home/) public repository. In particular, it can be easily integrated in deep learning pipeline(s) for computational pathology.

## Prerequisites
To run this script you will need the following packages, other than `histolab`:

- `pandas`
- `requests`
- `tqdm`
- `scikit-learn`

Extra requirements are available in the `examples_reqs.txt` file located in the `examples` folder.
All dependencies can be installed via command line, using `pip`:

```shell
pip install -f examples_reqs.txt
```

Moreover, a CSV file of patient metadata (`metadata_csv`) is required; this file can be retrieved from the GTEx [data portal](https://gtexportal.org/home/histologyPage) and it is structured as follows:

|   Tissue Sample ID  |   Tissue                          |   Subject ID  |   Sex     |   Age Bracket  |   Hardy Scale  |   Pathology Categories  |   Pathology Notes                                               |
|---------------------|-----------------------------------|---------------|-----------|----------------|----------------|-------------------------|-----------------------------------------------------------------|
|   GTEX-1117F-0126   |   Skin - Sun Exposed (Lower leg)  |   GTEX-1117F  |   female  |   60-69        |   Slow death   |                         |   6 pieces, minimal fat, squamous epithelium is ~50-70 microns  |
|   GTEX-1117F-0226   |   Adipose - Subcutaneous          |   GTEX-1117F  |   female  |   60-69        |   Slow death   |                         |   2 pieces, ~15% vessel stroma, rep delineated                  |
| ...                 | ...                               | ...           | ...       | ...            | ...            | ...                     | ...                                                             | 

## Workflow

The `extract_tile_pw_gtex.py` will perform the following steps:

1. the WSIs listed in the metadata file (`Tissue Sample ID` column) are downloaded from GTEx via the `download_wsi_gtex` function; slides are saved in the `wsi_dataset_dir` directory, which is specified as command-line argument.
2. a fixed number of tiles (100 by default) are randomly extracted from each WSI by the `extract_random_tiles` function. The directory where to store the tiles, along with several parameters that detail the extraction protocol (i.e. `n_tiles`, `seed`, `check_tissue`), can be defined as command-line arguments. 

**Note** `histolab` automatically saves the generated tiles in the 'tiles' subdirectory.

3. the `split_tiles_patient_wise` function sorts the tiles into the training and the test set (80-20 partition by default) adopting a *Patient-Wise* splitting protocol, namely ensuring that tiles belonging to the same subject are either in the training or the test set. 

## Usage

```
usage: extract_tile_pw_gtex.py [-h] [--metadata_csv METADATA_CSV]
                               [--wsi_dataset_dir WSI_DATASET_DIR]
                               [--tile_dataset_dir TILE_DATASET_DIR]
                               [--tile_size TILE_SIZE TILE_SIZE]
                               [--n_tiles N_TILES] [--level LEVEL]
                               [--seed SEED] [--check_tissue CHECK_TISSUE]

Retrieve a leakage-free dataset of tiles using a collection of WSI.

optional arguments:
  -h, --help            show this help message and exit
  --metadata_csv METADATA_CSV
                        CSV with WSI metadata. Default examples/GTEx/GTEx_AIDP2021.csv.
  --wsi_dataset_dir WSI_DATASET_DIR
                        Path where to save the WSIs. Default WSI_GTEx.
  --tile_dataset_dir TILE_DATASET_DIR
                        Path where to save the WSIs. Default tiles_GTEx.
  --tile_size TILE_SIZE TILE_SIZE
                        width and height of the cropped tiles. Default (512, 512).
  --n_tiles N_TILES     Maximum number of tiles to extract. Default 100.
  --level LEVEL         Magnification level from which extract the tiles. Default 2.
  --seed SEED           Seed for RandomState. Default 7.
  --check_tissue CHECK_TISSUE
                        Whether to check if the tile has enough tissue to be saved. Default True.
```