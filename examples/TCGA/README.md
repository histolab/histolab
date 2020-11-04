# Create a leakage-free dataset of tiles for TCGA

`extract_tile_pw_tcga.py` is a Python script proposed as reference to retrieve a reproducible dataset of tiles using a collection of WSIs from the [TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga) public repository. In particular, it can be easily integrated in deep learning pipeline(s) for computational pathology.

## Prerequisites
To run this script you will need the following packages, other than `histolab`:

- `pandas`
- `tqdm`
- `scikit-learn`

Moreover, a CSV file of patient clinical data (`clinical_csv`) is required; this file can be retrieved from the TCGA [data portal](https://portal.gdc.cancer.gov/) and it is structured as follows:

|   case_id  |   case_submitter_id                          |   project_id  |   age_at_index     |   ...  |   primary_diagnosis  |   ...  |   treatment_type                                               |
|---------------------|-----------------------------------|---------------|-----------|----------------|----------------|-------------------------|-----------------------------------------------------------------|
|   6cd9baf5-bbe0-4c1e-a87f-c53b3af22890  |   TCGA-A7-A13G  |   TCGA-BRCA  |   79  |   ...        |   Infiltrating duct carcinoma, NOS    |        ...                 |   Pharmaceutical Therapy, NOS  |
|   928c48a0-68ee-4e28-ae83-9832e52850ca   |   TCGA-CH-5753          |   TCGA-PRAD  |   70  |   ...      |   Adenocarcinoma, NOS    |       ...                  |   Radiation Therapy, NOS                  |
| ...                 | ...                               | ...           | ...       | ...            | ...            | ...                     | ...                                                             | 

## Workflow

The `extract_tile_pw_tcga.py` will perform the following steps:

1. a fixed number of tiles (100 by default) are randomly extracted from each WSI by the `extract_random_tiles` function. The directory where to store the tiles, along with several parameters that detail the extraction protocol (i.e. `n_tiles`, `seed`, `check_tissue`), can be defined as command-line arguments. 
**Note** `histolab` automatically saves the generated tiles in the 'tiles' subdirectory.

2. the `split_tiles_patient_wise` function sorts the tiles into the training and the test set (80-20 partition by default) adopting a *Patient-Wise* splitting protocol, namely ensuring that tiles belonging to the same subject are either in the training or the test set. 

## Usage

```
usage: extract_tile_pw_tcga.py [-h] [--clinical_csv CLINICAL_CSV]
                               [--wsi_dataset_dir WSI_DATASET_DIR]
                               [--tile_dataset_dir TILE_DATASET_DIR]
                               [--tile_size TILE_SIZE TILE_SIZE]
                               [--n_tiles N_TILES] [--level LEVEL]
                               [--seed SEED] [--check_tissue CHECK_TISSUE]

Retrieve a leakage-free dataset of tiles using a collection of WSI.

optional arguments:
  -h, --help            show this help message and exit
  --clinical_csv CLINICAL_CSV
                        CSV with WSI clinical data. Default examples/TCGA/clinical_csv_example.csv.
  --wsi_dataset_dir WSI_DATASET_DIR
                        Path where to save the WSIs. Default WSI_TCGA.
  --tile_dataset_dir TILE_DATASET_DIR
                        Path where to save the WSIs. Default tiles_TCGA.
  --tile_size TILE_SIZE TILE_SIZE
                        width and height of the cropped tiles. Default (512, 512).
  --n_tiles N_TILES     Maximum number of tiles to extract. Default 100.
  --level LEVEL         Magnification level from which extract the tiles. Default 2.
  --seed SEED           Seed for RandomState. Default 7.
  --check_tissue CHECK_TISSUE
                        Whether to check if the tile has enough tissue to be saved. Default True.
```