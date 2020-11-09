import argparse
import os
from typing import Tuple
from pathlib import Path

from tqdm import tqdm

import pandas as pd
from histolab.slide import SlideSet
from histolab.tiler import RandomTiler
from sklearn.model_selection import train_test_split

PATIENT_COL_NAME = "case_submitter_id"


def extract_random_tiles(
    dataset_dir: str,
    processed_path: str,
    tile_size: Tuple[int, int],
    n_tiles: int,
    level: int,
    seed: int,
    check_tissue: bool,
) -> None:
    """Save random tiles extracted from WSIs in `dataset_dir` into
    `processed_path`/tiles/

    Parameters
    ----------
    dataset_dir : str
        Path were the WSIs are saved
    processed_path : str
        Path where to store the tiles (will be concatenated with /tiles)
    tile_size : Tuple[int, int]
        width and height of the cropped tiles
    n_tiles : int
        Maximum number of tiles to extract
    level : int
        Magnification level from which extract the tiles
    seed : int
        Seed for RandomState
    check_tissue : bool
        Whether to check if the tile has enough tissue to be saved
    """
    slideset = SlideSet(dataset_dir, processed_path, valid_extensions=[".svs"])

    for slide in tqdm(slideset.slides):
        prefix = f"{slide.name}_"
        random_tiles_extractor = RandomTiler(
            tile_size=tile_size,
            n_tiles=n_tiles,
            level=level,
            seed=seed,
            check_tissue=check_tissue,
            prefix=prefix,
        )

        random_tiles_extractor.extract(slide)


def train_test_df_patient_wise(
    dataset_df: pd.DataFrame,
    patient_col: str,
    label_col: str,
    test_size: float = 0.2,
    seed: int = 1234,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``dataset_df`` into train/test partitions following a patient-wise protocol.

    Parameters
    ----------
    dataset_df : pd.DataFrame
        DataFrame containing the data to split
    patient_col : str
        Name of the patient column in ``dataset_df``
    label_col : str
        Name of the target column in ``dataset_df``
    test_size : float, optional
        Ratio of test set samples over the entire dataset, by default 0.2
    seed : int, optional
        Seed for RandomState, by default 1234

    Returns
    -------
    pd.DataFrame
        Training dataset
    pd.DataFrame
        Test dataset
    """

    patient_with_labels = (
        dataset_df.groupby(patient_col)[label_col].unique().apply(list)
    )
    unique_patients = patient_with_labels.index.values

    train_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=seed
    )

    dataset_train_df = dataset_df.loc[dataset_df[patient_col].isin(train_patients)]
    dataset_test_df = dataset_df.loc[dataset_df[patient_col].isin(test_patients)]

    return dataset_train_df, dataset_test_df


def split_tiles_patient_wise(
    tiles_dir: str,
    clinical_df: pd.DataFrame,
    train_csv_path: str,
    test_csv_path: str,
    label_col: str,
    patient_col: str,
    test_size: float = 0.2,
    seed: int = 1234,
) -> None:
    """Split a tile dataset into train-test following a patient-wise partitioning protocol.

    Save two CSV files containing the train-test partition for the tile dataset.

    Parameters
    ----------
    tiles_dir : str
        Tile dataset directory.
    clinical_df : pd.DataFrame
        CSV of patient clinical data.
    train_csv_path : str
        Path where to save the CSV file for the training set.
    test_csv_path : str
        Path where to save the CSV file for the test set.
    label_col : str
        Name of the target column in ``dataset_df``
    patient_col : str
        Name of the patient column in ``dataset_df``
    test_size : float, optional
        Ratio of test set samples over the entire dataset, by default 0.2
    seed : int, optional
        Seed for RandomState, by default 1234
    """
    tiles_filenames = [
        f for f in os.listdir(tiles_dir) if os.path.splitext(f)[1] == ".png"
    ]
    tiles_filenames_df = pd.DataFrame(
        {
            "tile_filename": tiles_filenames,
            f"{PATIENT_COL_NAME}": [f.split("_")[0] for f in tiles_filenames],
        }
    )

    tiles_clinical = clinical_df.join(
        tiles_filenames_df.set_index(PATIENT_COL_NAME), on=PATIENT_COL_NAME
    )

    train_df, test_df = train_test_df_patient_wise(
        tiles_clinical,
        patient_col,
        label_col,
        test_size,
        seed,
    )

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve a leakage-free dataset of tiles using a collection of WSI"
    )
    parser.add_argument(
        "--clinical_csv",
        type=str,
        default="clinical_csv_example.csv",
        help="CSV with clinical data. Default clinical_csv_example.csv",
    )
    parser.add_argument(
        "--wsi_dataset_dir",
        type=str,
        default="WSI_TCGA",
        help="Path where to save the WSIs. Default WSI_TCGA.",
    )
    parser.add_argument(
        "--tile_dataset_dir",
        type=str,
        default="tiles_TCGA",
        help="Path where to save the WSIs. Default tiles_TCGA.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="width and height of the cropped tiles. Default (512, 512).",
    )
    parser.add_argument(
        "--n_tiles",
        type=int,
        default=100,
        help="Maximum number of tiles to extract. Default 100.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=2,
        help="Magnification level from which extract the tiles. Default 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for RandomState. Default 7.",
    )
    parser.add_argument(
        "--check_tissue",
        type=bool,
        default=True,
        help="Whether to check if the tile has enough tissue to be saved. "
        "Default True.",
    )
    args = parser.parse_args()

    clinical_csv = Path(args.clinical_csv)
    wsi_dataset_dir = Path(args.wsi_dataset_dir)
    tile_dataset_dir = Path(args.tile_dataset_dir)
    tile_size = args.tile_size
    n_tiles = args.n_tiles
    level = args.level
    seed = args.seed
    check_tissue = args.check_tissue

    try:
        tcga_df = pd.read_csv(clinical_csv)
    except FileNotFoundError:
        print(f"Metadata CSV filepath {clinical_csv} does not exist. Please check.")
        return

    print("Extracting Random Tiles...", end=" ")
    extract_random_tiles(
        wsi_dataset_dir, tile_dataset_dir, tile_size, n_tiles, level, seed, check_tissue
    )
    print(f"..saved in {tile_dataset_dir}")

    print("Split Tiles Patient-wise...", end=" ")
    split_tiles_patient_wise(
        tiles_dir=os.path.join(tile_dataset_dir, "tiles"),
        clinical_df=tcga_df,
        train_csv_path=os.path.join(
            tile_dataset_dir, f"train_tiles_PW_{os.path.basename(clinical_csv)}"
        ),
        test_csv_path=os.path.join(
            tile_dataset_dir, f"test_tiles_PW_{os.path.basename(clinical_csv)}"
        ),
        label_col="tissue_or_organ_of_origin",
        patient_col=PATIENT_COL_NAME,
        test_size=0.2,
        seed=1234,
    )
    print("..done!")


if __name__ == "__main__":
    main()
