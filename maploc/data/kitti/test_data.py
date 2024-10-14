import argparse
from pathlib import Path
import shutil
import zipfile

import numpy as np
from tqdm.auto import tqdm

from ... import logger
from ...osm.tiling import TileManager
from ...osm.viz import GeoPlotter
from ...utils.geo import BoundaryBox, Projection
from ...utils.io import download_file, DATA_URL
from .utils import parse_gps_file
from .dataset import KittiDataModule

split_files = ["test1_files.txt", "test2_files.txt", "train_files.txt"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, default=Path(KittiDataModule.default_cfg["data_dir"])
    )
    parser.add_argument("--pixel_per_meter", type=int, default=2)
    parser.add_argument("--generate_tiles", action="store_true")
    args = parser.parse_args()
    
    
    tiles_path = args.data_dir / KittiDataModule.default_cfg["tiles_filename"]
    
    from IPython import embed;embed()



