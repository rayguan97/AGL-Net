
import collections
import collections.abc
from collections import defaultdict
from pathlib import Path
from typing import Optional
import os, json 
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from ... import logger, DATASETS_PATH
from ...rgbmap.tiling import RGBTileManager
from ..dataset import MapLocLiDarDataset
from ..sequential import chunk_sequence
from ..torch import collate, worker_init_fn

from .utils import ( 
    load_point_cloud, 
    get_transformation_from_ego, 
    get_cam_intrinsic,
    lidar_param, 
    cam_extrinsic_param, 
    cam_intrinsic_param
)

def pack_dump_dict(dump):

    return dump

class CarlaDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocLiDarDataset.default_cfg,
        "name": "carla",
        # paths and fetch
        "data_dir": DATASETS_PATH / "carla",
        "tiles_filename": "tiles.pkl",
        # "dump_filename": "dump.json",
        "dump_filename": "dump_gt.json",
        "split": "???",
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": None,
        # "max_num_val": 500,
        "selection_subset_val": "random",
        # "selection_subset_val": "furthest",
        "skip_frames": 1,
        # being overwritten
        "crop_size_meters": 128,
        "max_init_error": 64,
        "max_init_error_rotation": 4,
        "add_map_mask": "???",
        "mask_pad": "???",
        "oc_intrinsic_diff": False,
        "pre_load_points": False,
        "pre_load_tiles": False,
        "max_active_managers": 100,
        "overhead_map_size": 5000,
        "max_num_pts": 4096,
        "lidar_height": lidar_param[2],
        "lidar_forward": -1 * lidar_param[0],
        "filter_len": 0.1,
        "filter_up": True,
        "filter_down": True,
        "aug_scale": "???",
    }
    

    def __init__(self, cfg):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        splits = json.load(open(os.path.join(self.cfg.data_dir, self.cfg.split)))
        self.splits = {}
        for key in ["train", "val", "test"]:
            self.splits[key] = list(splits[key].keys())
        self.test_shift = splits["test_shifts"]

        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        # assert self.cfg.selection_subset_val in ["random", "furthest"]
        assert self.cfg.selection_subset_val in ["random"]
        if self.cfg.dump_filename == "dump_gt.json":
            assert not self.cfg.pre_load_points

        self.cfg.pixel_per_meter = 1

        self.tile_managers = {}
        self.active_managers = []
        # self.dump_dirs = {}
        # self.dump_dicts = {}

        self.data = {}
        self.names = {}
        assert not self.cfg.oc_intrinsic_diff
        if self.cfg.oc_intrinsic_diff:
            raise NotImplementedError
        else:
            self.oc_intrinsic = get_cam_intrinsic(*cam_intrinsic_param)


    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]

        for stage in stages:

            # with (self.root / Path(stage + "_" + self.cfg.dump_filename)).open("r") as fp:
            #     dump_dict = pack_dump_dict(json.load(fp))
            # self.dump_dicts[stage] = dump_dict
        
            self.names[stage] = []
            self.data[stage] = {
                "t_l2oc": [],
                "uv_gt": [],
                "theta_gt": []
            }
            if self.cfg.pre_load_points:
                self.data[stage]["pts_in_uv"] = []
                self.data[stage]["uv_idx"] = []
                # self.data[stage]["reverse_idx"] = []
                self.data[stage]["scale_gt"] = []

            split = self.splits[stage]

            do_val_subset = "val" == stage and self.cfg.max_num_val is not None
            if do_val_subset and self.cfg.selection_subset_val == "random":
                select = np.random.RandomState(self.cfg.seed).choice(
                    len(self.splits["val"]), self.cfg.max_num_val, replace=False
                )
                self.splits["val"] = [self.splits["val"][i] for i in select]
            
            if stage == "test":
                self.data[stage]["shifts"] = []
                shift_data = self.test_shift
            else:
                shift_data = None

            for route_dir in tqdm(split):

                # self.pack_data(stage, dump_dict[route_dir], route_dir, shift_data)
                self.pack_data(stage, route_dir, shift_data)
                

    # def pack_data(self, stage, dump_dict, route_dir, shift_data=None):
    def pack_data(self, stage, route_dir, shift_data=None):

        with (self.root / route_dir / self.cfg.dump_filename).open("r") as fp:
            dump_dict = pack_dump_dict(json.load(fp))

        # from IPython import embed;embed()

        for aerial_idx, ground_data in dump_dict.items():
                
            if self.cfg.pre_load_tiles and len(self.active_managers) < self.cfg.max_active_managers:
                if route_dir in self.tile_managers:
                    self.tile_managers[route_dir][aerial_idx] = RGBTileManager.load(Path(os.path.join(self.root, route_dir, aerial_idx + "_" + self.cfg.tiles_filename)))
                else:
                    self.tile_managers[route_dir] = {}
                    self.tile_managers[route_dir][aerial_idx] = RGBTileManager.load(Path(os.path.join(self.root, route_dir, aerial_idx + "_" + self.cfg.tiles_filename)))
                
                self.active_managers.append((route_dir, aerial_idx))

            # else: # test

            #     self.tile_managers[self.active_managers[0][0]][self.active_managers[0][1]] = None
            #     self.active_managers = self.active_managers[1:]
            #     if route_dir in self.tile_managers:
            #         self.tile_managers[route_dir][aerial_idx] = RGBTileManager.load(Path(os.path.join(self.root, route_dir, aerial_idx + "_" + self.cfg.tiles_filename)))
            #     else:
            #         self.tile_managers[route_dir] = {}
            #         self.tile_managers[route_dir][aerial_idx] = RGBTileManager.load(Path(os.path.join(self.root, route_dir, aerial_idx + "_" + self.cfg.tiles_filename)))
                
            #     self.active_managers.append((route_dir, aerial_idx))



            for ground_idx, ground_dict in ground_data.items():


                self.names[stage].append((route_dir, aerial_idx, ground_idx))
                self.data[stage]["t_l2oc"].append(np.array(ground_dict["t_l2oc"]))
                self.data[stage]["uv_gt"].append(np.array(ground_dict["uv_gt"]))
                self.data[stage]["theta_gt"].append(ground_dict["theta_gt"])


                if self.cfg.pre_load_points:
                    self.data[stage]["scale_gt"].append(ground_dict["scale_gt"])
                    pts_in_uv, uv_idx, reverse_idx = np.unique(np.array(ground_dict["pts_in_uv"]).astype(np.int_), return_index=True, return_inverse=True, axis=0)
                    self.data[stage]["pts_in_uv"].append(pts_in_uv)
                    self.data[stage]["uv_idx"].append(uv_idx)
                    # self.data[stage]["reverse_idx"].append(reverse_idx)

 
                if shift_data is not None:
                    self.data[stage]["shifts"].append(shift_data[route_dir]["topdown_rgb/" + aerial_idx + ".png"][ground_idx])

                # if self.oc_intrinsic is None:
                #     self.oc_intrinsic = np.array(ground_dict["oc_intrinsic"])


    def dataset(self, stage: str, ori_uv_=False):
        return MapLocLiDarDataset(
            stage,
            self.cfg,
            root_dir=self.root,
            names=self.names[stage],
            data=self.data[stage],
            tile_managers=self.tile_managers,
            active_managers=self.active_managers,
            max_active_managers=self.cfg.max_active_managers,
            pre_load_points=self.cfg.pre_load_points,
            oc_intrinsic = self.oc_intrinsic,
            max_num_pts=self.cfg.max_num_pts,
            tiles_filename = self.cfg.tiles_filename,
            image_ext=".png",
            ori_uv_=ori_uv_,
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
        ori_uv_=False,
    ):
        dataset = self.dataset(stage, ori_uv_=ori_uv_)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)
    
