import argparse
import os
import time
from typing import List, Tuple
from pathlib import Path
from random import randint

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from histolab.slide import SlideSet
from histolab.tiler import RandomTiler

URL_ROOT = "https://brd.nci.nih.gov/brd/imagedownload"


def download_wsi_gtex(dataset_dir: str, sample_ids: List[str]) -> None:
    """Download into ``dataset_dir`` all the GTEx WSIs corresponding to ``sample_ids``

    Parameters
    ----------
    dataset_dir : str
        Path where to save the WSIs
    sample_ids : List[str]
        List of GTEx WSI ids
    """
    dir_files = os.listdir(dataset_dir)
    sample_ids = set(sample_ids)  # avoid any possible repetition
    downloaded = set(filter(lambda sid: f"{sid}.svs" in dir_files, sample_ids))
    to_download = set(sample_ids).difference(downloaded)

    if len(to_download) > 0:
        if len(downloaded) > 0:
            print(f"{len(downloaded)} out of {len(sample_ids)} found. Resuming:")
        else:
            print(
                "downloading GTEx WSI dataset. "
                "This may take several minutes to complete."
            )
        for sample_id in tqdm(
            to_download, initial=len(downloaded), total=len(sample_ids)
        ):
            with requests.get(f"{URL_ROOT}/{sample_id}", stream=True) as request:
                request.raise_for_status()
                with open(
                    os.path.join(dataset_dir, f"{sample_id}.svs"), "wb"
                ) as output_file:
                    for chunk in request.iter_content(chunk_size=8192):
                        output_file.write(chunk)
            time.sleep(randint(3, 7))  # slight delay to avoid over-flooding


def extract_random_tiles(
    dataset_dir: str,
    processed_path: str,
    tile_size: Tuple[int, int],
    n_tiles: int,
    level: int,
    seed: int,
    check_tissue: bool,
) -> None:
    """Save random tiles extracted from WSIs in `dataset_dir` into `processed_path`/tiles

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
    metadata_df: pd.DataFrame,
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
    metadata_df : pd.DataFrame
        CSV of patient metadata.
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
            "Tissue Sample ID": [f.split("_")[0] for f in tiles_filenames],
        }
    )

    tiles_metadata = metadata_df.join(
        tiles_filenames_df.set_index("Tissue Sample ID"), on="Tissue Sample ID"
    )

    train_df, test_df = train_test_df_patient_wise(
        tiles_metadata,
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
        "--metadata_csv",
        type=str,
        default="GTEx_AIDP2021.csv",
        help="CSV with WSI metadata. Default GTEx_AIDP2021.csv.",
    )
    parser.add_argument(
        "--wsi_dataset_dir",
        type=str,
        default="WSI_GTEx",
        help="Path where to save the WSIs. Default WSI_GTEx.",
    )
    parser.add_argument(
        "--tile_dataset_dir",
        type=str,
        default="tiles_GTEx",
        help="Path where to save the WSIs. Default tiles_GTEx.",
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

    metadata_csv = Path(args.metadata_csv)
    wsi_dataset_dir = Path(args.wsi_dataset_dir)
    tile_dataset_dir = Path(args.tile_dataset_dir)
    tile_size = args.tile_size
    n_tiles = args.n_tiles
    level = args.level
    seed = args.seed
    check_tissue = args.check_tissue

    try:
        gtex_df = pd.read_csv(metadata_csv)
    except FileNotFoundError:
        print(f"Metadata CSV filepath {metadata_csv} does not exist. Please check.")
        return
    else:
        sample_ids = gtex_df["Tissue Sample ID"].tolist()

    os.makedirs(wsi_dataset_dir, exist_ok=True)
    print("Check GTEX dataset...")
    download_wsi_gtex(wsi_dataset_dir, sample_ids)
    print("done.")

    print("Extracting Random Tiles...", end=" ")
    extract_random_tiles(
        wsi_dataset_dir, tile_dataset_dir, tile_size, n_tiles, level, seed, check_tissue
    )
    print(f"..saved in {tile_dataset_dir}")

    print("Split Tiles Patient-wise...", end=" ")
    split_tiles_patient_wise(
        tiles_dir=os.path.join(tile_dataset_dir, "tiles"),
        metadata_df=gtex_df,
        train_csv_path=os.path.join(
            tile_dataset_dir, f"train_tiles_PW_{os.path.basename(metadata_csv)}"
        ),
        test_csv_path=os.path.join(
            tile_dataset_dir, f"test_tiles_PW_{os.path.basename(metadata_csv)}"
        ),
        label_col="Tissue",
        patient_col="Subject ID",
        test_size=0.2,
        seed=1234,
    )
    print("..done!")


if __name__ == "__main__":
    main()
