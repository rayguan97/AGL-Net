## clean up data that are not used, 
# after splits_carla_loc is created.
import asyncio
import argparse
from collections import defaultdict
import json
import shutil
from pathlib import Path
from typing import List
import os 
from PIL import Image
import matplotlib.image as mpimg
import math
import numpy as np
import cv2
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import glob

data_dir = "/home/rayguan/scratch2/carla"
split_filename = "splits_carla_loc.json"

if __name__ == "__main__":

    with open(os.path.join(data_dir, split_filename), 'r') as file:
        train_test_split = json.load(file)


    ## remove dir
    keep_lst = []

    for split in ["train", "val", "test"]:
        keep_lst.extend([os.path.join(data_dir, p) for p in train_test_split[split].keys()])

    all_lst = []

    scene_lst = [s for s in os.listdir(data_dir) if ".json" not in s ]

    for s in scene_lst:
        # all_lst.extend(glob.glob(os.path.join(data_dir, s, "*")))
        all_lst.extend(glob.glob(os.path.join(data_dir, s, "*")))
    
    for d in tqdm(all_lst):
        if d not in keep_lst:
            shutil.rmtree(d)

    ## remove file
    # from IPython import embed;embed()
    
    keep_topdown = []
    keep_lidar = []

    for split in ["train", "val", "test"]:
        for k, v in tqdm(train_test_split[split].items()):
            temp_keep = []
            for kk, vv in v.items():
                keep_topdown.append(os.path.join(data_dir, k, kk))
                temp_keep.extend([os.path.join(data_dir, k, "lidar", nn + ".npy") for nn in vv])

            for l in temp_keep:
                if l not in keep_lidar:
                    keep_lidar.append(l)

            # keep_topdown.extend([os.path.join(data_dir, k, kk) for kk in v.keys()])
            # keep_lidar.extend([os.path.join(data_dir, k, "lidar", nn + ".npy") for nn in v.keys()])
            # from IPython import embed;embed()


    for d in tqdm(keep_lst):
        all_topdown = []
        all_topdown.extend(glob.glob(os.path.join(d, "topdown_rgb", "*")))

        for topdown_path in all_topdown:
            if topdown_path not in keep_topdown and os.path.exists(topdown_path):
                os.remove(topdown_path)

        all_lidar = []
        all_lidar.extend(glob.glob(os.path.join(d, "lidar", "*")))

        for lidar_path in all_lidar:
            if lidar_path not in keep_lidar and os.path.exists(lidar_path):
                os.remove(lidar_path)
