# Copyright (c) Meta Platforms, Inc. and affiliates.

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
import os

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from omegaconf import DictConfig, OmegaConf
import fpsample
from mmcv.ops import Voxelization
import matplotlib.pyplot as plt
import cv2


from ..models.utils import deg2rad, rotmat2d
from ..osm.tiling import TileManager
from ..rgbmap.tiling import RGBTileManager
from ..utils.geo import BoundaryBox
from ..utils.io import read_image
from ..utils.wrappers import Camera
from .image import pad_image, rectify_image, resize_image
from .utils import decompose_rotmat, random_flip, random_rot90, random_flip_rgb, random_rot90_rgb
from .carla.utils import ( 
    load_point_cloud, 
    load_point_cloud_kitti,
    get_transformation_from_ego, 
    get_cam_intrinsic,
    overlay_points_on_image,
    lidar_param, 
    cam_extrinsic_param, 
    cam_intrinsic_param,
    carla_trans,
    to_normal_pc_trans, # x, y, z - r, g, b - front, left, top
)

# def overlay_points_on_image(image, points_2d, color=((255, 0, 0))):
#     # Overlay the 2D points on the satellite image
#     # For simplicity, let's just draw points on the image
#     # lst = []
#     im = image.copy().astype(np.int32)
#     for point in points_2d:
#         if 0 <= int(point[0]) < image.shape[0] and 0 <= int(point[1]) < image.shape[1]:
#             # from IPython import embed;embed()   
#             cv2.circle(im, (int(point[0]), int(point[1])), 1, color, -1)
#             # try:
#             #     cv2.circle(im, (int(point[0]), int(point[1])), 1, color, -1)
#             # except:
#             #     from IPython import embed;embed()

#     # print(lst)
#     # from IPython import embed;embed()
#     return im

def to_valid_scale(uv_init, crop_size, scale, map_size):
    y_min, x_min = uv_init - crop_size * scale
    y_max, x_max = uv_init + crop_size * scale

    if x_min < 0:
        x_max = x_max - x_min
        x_min = 0

    if y_min < 0:
        y_max = y_max - y_min
        y_min = 0

    if x_max > map_size:
        x_min = x_min - (x_max - map_size + 1)
        x_max = map_size - 1

    if y_max > map_size:
        y_min = y_min - (y_max - map_size + 1)
        y_max = map_size - 1


    assert x_min >= 0
    assert y_min >= 0
    assert x_max < map_size
    assert y_max < map_size
    assert x_min < x_max
    assert y_min < y_max

    uv_init = np.array([y_min + crop_size * scale, x_min + crop_size * scale]).flatten()
    return uv_init


class MapLocLiDarDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "random": True,
        "num_threads": None,
        # map
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        # "init_from_gps": False,
        # "return_gps": False,
        # pose priors
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 2,
        "prior_range_rotation": None,
        # image preprocessing
        "pad_to_multiple": 32,
        "augmentation": {
            "rot90": False,
            "flip": False,
        },
        "point_cloud_range": "???",
        "use_voxel": "???",
        "voxel_size": [0.5, 0.5, 0.5],
        "pts_per_voxel": 35,
        "max_voxels": 2048,
        "aug_scale": "???",
    }

    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        root_dir: str,
        names: List[Any],
        data: Dict[str, Any],
        tile_managers: Dict[str, RGBTileManager],
        active_managers: List[Any],
        max_active_managers: int,
        pre_load_points: bool,
        oc_intrinsic: Any,
        max_num_pts: int = 4096,
        tiles_filename: str = "tiles.pkl",
        image_ext: str = "",
        ori_uv_ = False,
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.root_dir = root_dir
        self.names = names
        self.data = data
        self.tile_managers = tile_managers
        self.active_managers = active_managers
        self.max_active_managers = max_active_managers
        self.pre_load_points = pre_load_points
        self.tiles_filename = tiles_filename
        self.image_ext = image_ext
        self.oc_intrinsic = oc_intrinsic
        if self.cfg.use_voxel:
            self.voxel_gen = Voxelization(self.cfg.voxel_size, self.cfg.point_cloud_range, self.cfg.pts_per_voxel, self.cfg.max_voxels)
            # self.voxel_sampler = VoxelBasedPointSampler(
            #     {"voxel_size": self.cfg.voxel_size, 
            #     "point_cloud_range": self.cfg.point_cloud_range,
            #     "max_num_points": self.cfg.max_num_pts}
            #     )
        self.lidar_forward = cfg.lidar_forward
        if cfg.filter_up:
            self.filter_up = cfg.filter_len - cfg.lidar_height
        else:
            self.filter_up = None

        if cfg.filter_down:
            self.filter_down = -1 * cfg.filter_len - cfg.lidar_height
        else:
            self.filter_down = None

        self.aug_scale = self.cfg.aug_scale
        self.ori_uv_ = ori_uv_


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        route_dir, aerial_idx, ground_idx = self.names[idx]
        # if self.cfg.init_from_gps:
        #     # latlon_gps = self.data["gps_position"][idx][:2].clone().numpy()
        #     # xy_w_init = self.tile_managers[scene].projection.project(latlon_gps)
        #     raise NotImplementedError
        # else:
            # uv_gt = self.data["uv_gt"][idx].clone().double().numpy()
    
        uv_gt = self.data["uv_gt"][idx]

        if "shifts" in self.data:
            assert self.stage == "test"
            error = np.array(self.data["shifts"][idx][:2])
        else:
            error = np.random.RandomState(seed).uniform(-1, 1, size=2)
        uv_init = uv_gt + error * self.cfg.max_init_error

        adjust_scale = (np.random.random() < self.aug_scale)

        if adjust_scale or self.stage == "test":
            rad_num = abs(self.data["shifts"][idx][0]) if self.stage == "test" else np.random.random()
            if rad_num > 0.5:
                patch_scale = np.random.RandomState(seed + 1).uniform(
                    max(0.5, np.max(np.abs(error) * self.cfg.max_init_error) * 2 / self.cfg.crop_size_meters ), 1, size=1)[0]
            else:
                patch_scale = np.random.RandomState(seed + 1).uniform(1, self.cfg.overhead_map_size / (4 * self.cfg.crop_size_meters), size=1)[0]
        else:
            patch_scale = 1

        # x_min, y_min = uv_init[::-1] - self.cfg.crop_size_meters
        # x_max, y_max = uv_init[::-1] + self.cfg.crop_size_meters

        uv_init = to_valid_scale(uv_init, self.cfg.crop_size_meters, patch_scale, self.cfg.overhead_map_size)

        bbox_tile = BoundaryBox(
            uv_init[::-1] - self.cfg.crop_size_meters,
            uv_init[::-1] + self.cfg.crop_size_meters
            )


        tmp =  uv_gt - bbox_tile.min_[::-1] - (1-patch_scale) * self.cfg.crop_size_meters
        # if not ( (tmp[0] > 0) and (tmp[0] < 1024) and \
        #         (tmp[1] > 0) and (tmp[1] < 1024) ):
        # from IPython import embed;embed()
        # try:
        assert (tmp[0] > 0) and (tmp[0] < 2 * patch_scale * self.cfg.crop_size_meters) and \
                (tmp[1] > 0) and (tmp[1] < 2 * patch_scale * self.cfg.crop_size_meters) 
        # except:
        #     from IPython import embed;embed()

        # if min(uv_init - self.cfg.crop_size_meters) < 0:
        #     from IPython import embed;embed()
        return self.get_data(idx, route_dir, aerial_idx, ground_idx, seed, uv_gt, bbox_tile, patch_scale)

    def get_data(self, idx, route_dir, aerial_idx, ground_idx, seed, uv_gt, bbox_tile, patch_scale):
        data = {
            "index": idx,
            "route_dir": route_dir,
            "aerial_idx": aerial_idx,
            "ground_idx": ground_idx,
        }
        
        # raster extraction
        if route_dir in self.tile_managers and aerial_idx in self.tile_managers and \
            self.tile_managers[route_dir][aerial_idx] is not None:

            canvas = self.tile_managers[route_dir][aerial_idx].query(bbox_tile, patch_scale)
        else:
            if len(self.active_managers) >= self.max_active_managers:
                assert len(self.active_managers) == self.max_active_managers
                # del self.tile_managers[self.active_managers[0][0]][self.active_managers[0][1]]
                self.tile_managers[self.active_managers[0][0]][self.active_managers[0][1]] = None
                self.active_managers = self.active_managers[1:]

            if route_dir not in self.tile_managers:
                self.tile_managers[route_dir] = {}
            map_data = None
            self.tile_managers[route_dir][aerial_idx] = RGBTileManager.load(Path(os.path.join(self.root_dir, route_dir, aerial_idx + "_" + self.tiles_filename)))
            self.active_managers.append((route_dir, aerial_idx))
            canvas = self.tile_managers[route_dir][aerial_idx].query(bbox_tile, patch_scale)

        uv_init = bbox_tile.center[::-1]
        raster = canvas.raster  # H, W, 3
        theta_gt = self.data["theta_gt"][idx]
        t_l2oc = self.data["t_l2oc"][idx]

        point_cloud = load_point_cloud(
            os.path.join(self.root_dir, route_dir, "lidar", ground_idx + ".npy"),
            self.filter_up, self.filter_down, self.lidar_forward)

        point_cloud.transform(carla_trans)
        point_cloud.transform(to_normal_pc_trans)

        pts = np.asarray(point_cloud.points).copy()
        
        if self.cfg.use_voxel:
            
            voxels, coors, num_points_per_voxel = self.voxel_gen(torch.tensor(pts, dtype=torch.float))
            if voxels.shape[0] < max(self.voxel_gen.max_voxels):
                padding_points = np.zeros([
                    max(self.voxel_gen.max_voxels) - voxels.shape[0], 
                    self.voxel_gen.max_num_points, 3],
                    dtype=pts.dtype)
                padding_points[:] = voxels[0]
                voxels = torch.from_numpy(np.concatenate([voxels, padding_points], axis=0))

                padding_coors = np.zeros([
                    max(self.voxel_gen.max_voxels) - coors.shape[0], 3], 
                    dtype=np.int_)
                padding_coors[:] = coors[0]
                coors = torch.from_numpy(np.concatenate([coors, padding_coors], axis=0))

                padding_num = np.zeros(padding_coors.shape[0], dtype=np.int_)
                num_points_per_voxel = torch.from_numpy(np.concatenate([num_points_per_voxel, padding_num], axis=0))

        if self.pre_load_points:
            pts_in_uv = self.data["pts_in_uv"][idx]
            uv_idx = self.data["uv_idx"][idx]
            # reverse_idx = self.data["reverse_idx"][idx]
            scale_gt = self.data["scale_gt"][idx]
            adjust_pixels_per_meter = -1
            
            raise NotImplementedError

            # pts = np.load(os.path.join(self.root, route_dir, "lidar", ground_idx + ".npy"), allow_pickle=True)[1][:,:3]
        else:
            point_cloud.transform(t_l2oc)

            pts_3d = np.asarray(point_cloud.points)
            points_2d_homogeneous = self.oc_intrinsic @ pts_3d.T
            points_2d =  points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
            pts_in_uv = points_2d.T

            # ppm_lst = ( np.sqrt(np.sum((pts_in_uv[1:,:] - pts_in_uv[0,:])**2, 1)) / np.sqrt(np.sum((pts[1:,:2] - pts[0,:2])**2, 1)) ).flatten()
            # ppm_lst = ppm_lst[~np.isnan(ppm_lst)]
            # adjust_pixels_per_meter = np.average(ppm_lst)

            pts_in_uv, uv_idx, reverse_idx = np.unique(pts_in_uv.astype(np.int_), return_index=True, return_inverse=True, axis=0)

            # scale_gt = np.max(pts) - np.min(pts)

        # point_cloud_norm = pts / scale_gt        
        
        if self.ori_uv_:
            ori_uv_pts = pts_in_uv.copy()
            data["ori_uv_pts"] = ori_uv_pts

        if len(pts_in_uv) < self.cfg.max_num_pts:
            idxs = np.concatenate([np.arange(len(pts_in_uv)), np.zeros(self.cfg.max_num_pts - len(pts_in_uv)).astype(np.int_)])
        else:
            idxs = fpsample.bucket_fps_kdline_sampling(pts_in_uv, self.cfg.max_num_pts, h=7)


        uv_offset = canvas.bbox.min_[::-1]

        pts_in_uv = ( (pts_in_uv - canvas.bbox.center[::-1]) / patch_scale ) + canvas.bbox.center[::-1]
        uv_gt = ( (uv_gt - canvas.bbox.center[::-1]) / patch_scale ) + canvas.bbox.center[::-1]

        pts_in_uv_sub = pts_in_uv[idxs]
        uv_idx_sub = uv_idx[idxs]

        # point_cloud_norm_sub = point_cloud_norm[uv_idx_sub]
        # pc_sub = point_cloud_norm_sub * scale_gt

        pc_sub = pts[uv_idx_sub]

        # ### using cv2.circle (u, v) // using array[v, u]
        # aa = np.ascontiguousarray(raster).copy()

        # # overlay_image = overlay_points_on_image(aa, (pts_in_uv - canvas.bbox.min_[::-1]).astype(np.int_))
        # overlay_image = overlay_points_on_image(aa, (pts_in_uv - canvas.bbox.min_[::-1]).astype(np.int_))
        # overlay_image = overlay_points_on_image(overlay_image, [(np.array(uv_gt - canvas.bbox.min_[::-1])).astype(np.int_)], (0, 0, 255))

        # for p in pts_in_uv - canvas.bbox.min_[::-1]:
        #     if 0 <= int(p[0]) < aa.shape[0] and 0 <= int(p[1]) < aa.shape[1]:
        #         aa[int(p[1]), int(p[0])][0] = 255
        #         aa[int(p[1]), int(p[0])][1] = 0
        #         aa[int(p[1]), int(p[0])][2] = 0
        # # # aa[5, -5][0] = 0
        # # # aa[5, -5][1] = 0
        # # # aa[5, -5][2] = 255
        # # # cv2.circle(aa, (100, 50), 1, [255, 0, 0], -1)
        # # aa[50, 100][0] = 255
        # # aa[50, 100][1] = 0
        # # aa[50, 100][2] = 0

        # from IPython import embed;embed()
 
        # plt.imshow(overlay_image)
        # plt.show()

        # plt.imshow(aa)
        # plt.show()

        # ### 

        # (pts_in_uv_sub[1,:] - pts_in_uv_sub[0,:]) / (pc_sub[1,:2] - pc_sub[0,:2])
        # from IPython import embed;embed()

        # from IPython import embed;embed()


        ppm_lst = ( np.sqrt(np.sum((pts_in_uv_sub[1:50,:] - pts_in_uv_sub[0,:])**2, 1)) / np.sqrt(np.sum((pc_sub[1:50,:2] - pc_sub[0,:2])**2, 1)) ).flatten()
        # ppm_lst = ( np.sqrt(np.sum((pts_in_uv_sub[1:50,::-1] - pts_in_uv_sub[0,::-1])**2, 1)) / np.sqrt(np.sum((pc_sub[1:50,:2] - pc_sub[0,:2])**2, 1)) ).flatten()
        ppm_lst = ppm_lst[~np.isnan(ppm_lst)]
        adjust_pixels_per_meter = np.average(ppm_lst)

        assert not np.isnan(adjust_pixels_per_meter)
        
        # Map augmentations -- need to modify point cloud, uv_pts and t_l2oc transformation!
        # heading = np.deg2rad(90 - yaw)  # fixme
        # if self.stage == "train":
        #     if self.cfg.augmentation.rot90:
        #         raster, uv_gt, heading = random_rot90_rgb(raster, uv_gt, heading, seed)
        #     if self.cfg.augmentation.flip:
        #         image, raster, uv_gt, heading = random_flip_rgb(
        #             image, raster, uv_gt, heading, seed
        #         )
        # yaw = 90 - np.rad2deg(heading)  # fixme

        # Create the mask for prior location
        if self.cfg.add_map_mask:
            data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas, patch_scale, uv_gt-uv_offset))

        if self.cfg.max_init_error_rotation is not None:
            if "shifts" in self.data:
                error = torch.tensor(self.data["shifts"][idx][-1],  dtype=torch.float)
            else:
                error = np.random.RandomState(seed + 1).uniform(-1, 1)
                error = torch.tensor(error, dtype=torch.float)
            theta_init = theta_gt + error * self.cfg.max_init_error_rotation
            range_ = self.cfg.max_init_error_rotation
            if range_ < 3:
                data["yaw_prior"] = torch.stack([theta_init * 180 / np.pi, torch.tensor(range_* 180 / np.pi)])
        # from IPython import embed;embed()

        lidar_point_mask_gt = torch.cat([
            torch.zeros((raster.shape[0], raster.shape[1], 1)),
            torch.ones((raster.shape[0], raster.shape[1], 1)),
            ], dim=-1) 
        uv_pts_all = (pts_in_uv-uv_offset).astype(np.int_)
        
        u_valid_pts_idx = np.where(np.logical_and(uv_pts_all[:,0] > 0, uv_pts_all[:,0] < raster.shape[0]))
        v_valid_pts_idx = np.where(np.logical_and(uv_pts_all[:,1] > 0, uv_pts_all[:,1] < raster.shape[1]))
        valid_idx = np.intersect1d(u_valid_pts_idx, v_valid_pts_idx)
        valid_pts = uv_pts_all[valid_idx]

        lidar_point_mask_gt[valid_pts[:, 1], valid_pts[:,0], torch.zeros(valid_pts[:,0].shape).int()] = 1
        lidar_point_mask_gt[valid_pts[:, 1], valid_pts[:,0], torch.ones(valid_pts[:,0].shape).int()] = 0
        
        lidar_coverage_mask_gt = torch.zeros((raster.shape[0], raster.shape[1]))
        v_min, u_min = np.min(uv_pts_all, 0)
        v_max, u_max = np.max(uv_pts_all, 0)
        v_min = max(0, v_min)
        v_max = min(raster.shape[0], v_max)
        u_min = max(0, u_min)
        u_max = min(raster.shape[1], u_max)
        lidar_coverage_mask_gt[v_min:v_max, u_min:u_max] = torch.ones((v_max - v_min, u_max - u_min))

        lidar_shape_len = np.max(np.max(uv_pts_all, 0) - np.min(uv_pts_all, 0))

        return {
            **data,
            "canvas": canvas,
            "map": torch.from_numpy(np.ascontiguousarray(raster)).long(),
            "point_cloud": torch.from_numpy(pc_sub).float(),  
            "lidar_coverage_mask_gt": lidar_coverage_mask_gt.bool(),
            "lidar_point_mask_gt": lidar_point_mask_gt.float().moveaxis(-1, 0),
            "lidar_shape_len": torch.tensor(lidar_shape_len),
            # "point_cloud_norm": torch.from_numpy(point_cloud_norm_sub * scale_gt).float(),  
            "voxels": voxels.float(),  
            "coors": coors.int(),  
            "num_points_per_voxel": num_points_per_voxel.int(),
            "uv_offset": torch.from_numpy(uv_offset.copy()).int(),
            "patch_scale": torch.tensor(patch_scale).float(),
            "pts_in_uv": torch.from_numpy(pts_in_uv_sub - uv_offset).int(),  
            "uv_idx": torch.from_numpy(uv_idx_sub).int(),  
            # "reverse_idx": torch.from_numpy(reverse_idx).int(),  
            "uv_init": torch.from_numpy(uv_init.copy()).int(),  
            "uv_gt": torch.from_numpy(uv_gt - uv_offset).int(),  
            "theta_init": (theta_init.float()) % (2 * np.pi),  
            "theta_gt": torch.tensor((theta_gt) % (2 * np.pi), dtype=torch.float),  
            # "scale_gt": torch.tensor(scale_gt, dtype=torch.float),  
            # "t_l2oc": torch.from_numpy(t_l2oc).float(),   need to modify this to adjust the meters
            "pixels_per_meter": torch.tensor(canvas.ppm).float(),
            "adjust_pixels_per_meter": torch.tensor(adjust_pixels_per_meter),
        }

    def create_map_mask(self, canvas, patch_scale, uv_gt):
        map_mask = np.zeros((canvas.h, canvas.w), bool)
        radius = self.cfg.max_init_error / patch_scale
        mask_min, mask_max = np.round(
            [int(canvas.h/2), int(canvas.w/2)]
            + np.array([[-1], [1]]) * (radius + self.cfg.mask_pad) * canvas.ppm
        ).astype(int)
        if not map_mask[int(uv_gt[1]), int(uv_gt[0])]:
            mask_min[0] = max(0, min(mask_min[0], int(uv_gt[1]) - self.cfg.mask_pad))
            mask_min[1] = max(0 ,min(mask_min[1], int(uv_gt[0]) - self.cfg.mask_pad))
            mask_max[0] = min(canvas.h-1, max(mask_max[0], int(uv_gt[1]) + self.cfg.mask_pad))
            mask_max[1] = min(canvas.w-1, max(mask_max[1], int(uv_gt[0]) + self.cfg.mask_pad))
        map_mask[mask_min[0] : mask_max[0], mask_min[1] : mask_max[1]] = True

        map_mask[int(uv_gt[1]), int(uv_gt[0])] = True
        # assert map_mask[int(uv_gt[1]), int(uv_gt[0])]
        return map_mask


class MapLocOriLidarDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # map
        "num_classes": None,
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        "init_from_gps": False,
        "return_gps": False,
        "force_camera_height": None,
        # pose priors
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 1,
        "prior_range_rotation": None,
        # image preprocessing
        "target_focal_length": None,
        "reduce_fov": None,
        "resize_image": None,
        "pad_to_square": False,  # legacy
        "pad_to_multiple": 32,
        "rectify_pitch": True,
        "augmentation": {
            "rot90": False,
            "flip": False,
            "image": {
                "apply": False,
                "brightness": 0.5,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.5 / 3.14,
            },
        },
        "max_num_pts": "???",
        "point_cloud_range": "???",
        "voxel_size": [0.5, 0.5, 0.5],
        "pts_per_voxel": 35,
        "max_voxels": 2048,
    }

    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        tile_managers: Dict[str, TileManager],
        image_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.tile_managers = tile_managers
        self.names = names
        self.image_ext = image_ext
        self.voxel_gen = Voxelization(self.cfg.voxel_size, self.cfg.point_cloud_range, self.cfg.pts_per_voxel, self.cfg.max_voxels)

        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))
        self.tfs = tvf.Compose(tfs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        # from IPython import embed;embed()

        scene, seq, name = self.names[idx]
        if self.cfg.init_from_gps:
            latlon_gps = self.data["gps_position"][idx][:2].clone().numpy()
            xy_w_init = self.tile_managers[scene].projection.project(latlon_gps)
        else:
            xy_w_init = self.data["t_c2w"][idx][:2].clone().double().numpy()

        if "shifts" in self.data:
            yaw = self.data["roll_pitch_yaw"][idx][-1]
            R_c2w = rotmat2d((90 - yaw) / 180 * np.pi).float()
            error = (R_c2w @ self.data["shifts"][idx][:2]).numpy()
        else:
            error = np.random.RandomState(seed).uniform(-1, 1, size=2)
        xy_w_init += error * self.cfg.max_init_error

        bbox_tile = BoundaryBox(
            xy_w_init - self.cfg.crop_size_meters,
            xy_w_init + self.cfg.crop_size_meters,
        )
        return self.get_view(idx, scene, seq, name, seed, bbox_tile)

    def get_view(self, idx, scene, seq, name, seed, bbox_tile):
        data = {
            "index": idx,
            "name": name,
            "scene": scene,
            "sequence": seq,
        }
        cam_dict = self.data["cameras"][scene][seq][self.data["camera_id"][idx]]
        cam = Camera.from_dict(cam_dict).float()

        if "roll_pitch_yaw" in self.data:
            roll, pitch, yaw = self.data["roll_pitch_yaw"][idx].numpy()
        else:
            roll, pitch, yaw = decompose_rotmat(self.data["R_c2w"][idx].numpy())
        image = read_image(self.image_dirs[scene] / (name + self.image_ext))
        pc_name_lst = name.split("/")
        pc_name_lst[-3] = "velodyne_points"
        pc_name = "/".join(pc_name_lst).split(".")[0] + ".bin"
        points = load_point_cloud_kitti(os.path.join(self.image_dirs[scene], pc_name))
        pts = np.asarray(points.points).copy()

        voxels, coors, num_points_per_voxel = self.voxel_gen(torch.tensor(pts, dtype=torch.float))
        if voxels.shape[0] < max(self.voxel_gen.max_voxels):
            padding_points = np.zeros([
                max(self.voxel_gen.max_voxels) - voxels.shape[0], 
                self.voxel_gen.max_num_points, 3],
                dtype=pts.dtype)
            padding_points[:] = voxels[0]
            voxels = torch.from_numpy(np.concatenate([voxels, padding_points], axis=0))

            padding_coors = np.zeros([
                max(self.voxel_gen.max_voxels) - coors.shape[0], 3], 
                dtype=np.int_)
            padding_coors[:] = coors[0]
            coors = torch.from_numpy(np.concatenate([coors, padding_coors], axis=0))

            padding_num = np.zeros(padding_coors.shape[0], dtype=np.int_)
            num_points_per_voxel = torch.from_numpy(np.concatenate([num_points_per_voxel, padding_num], axis=0))

        # scale_gt = np.max(pts) - np.min(pts)
        # point_cloud_norm = pts / scale_gt        

        # idxs = fpsample.bucket_fps_kdline_sampling(point_cloud_norm, self.cfg.max_num_pts, h=7)
        # point_cloud_norm_sub = point_cloud_norm[idxs]

        idxs = fpsample.bucket_fps_kdline_sampling(pts, self.cfg.max_num_pts, h=7)
        pts = pts[idxs]

        if "plane_params" in self.data:
            # transform the plane parameters from world to camera frames
            plane_w = self.data["plane_params"][idx]
            data["ground_plane"] = torch.cat(
                [rotmat2d(deg2rad(torch.tensor(yaw))) @ plane_w[:2], plane_w[2:]]
            )
        if self.cfg.force_camera_height is not None:
            data["camera_height"] = torch.tensor(self.cfg.force_camera_height)
        elif "camera_height" in self.data:
            data["camera_height"] = self.data["height"][idx].clone()

        # raster extraction
        canvas = self.tile_managers[scene].query(bbox_tile)
        xy_w_gt = self.data["t_c2w"][idx][:2].numpy()
        uv_gt = canvas.to_uv(xy_w_gt)
        uv_init = canvas.to_uv(bbox_tile.center)
        raster = canvas.raster  # C, H, W

        # Map augmentations
        heading = np.deg2rad(90 - yaw)  # fixme
        if self.stage == "train":
            if self.cfg.augmentation.rot90:
                raster, uv_gt, heading = random_rot90(raster, uv_gt, heading, seed)
            if self.cfg.augmentation.flip:
                image, raster, uv_gt, heading = random_flip(
                    image, raster, uv_gt, heading, seed
                )
        yaw = 90 - np.rad2deg(heading)  # fixme

        image, valid, cam, roll, pitch = self.process_image(
            image, cam, roll, pitch, seed
        )

        # Create the mask for prior location
        if self.cfg.add_map_mask:
            data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas))

        if self.cfg.max_init_error_rotation is not None:
            if "shifts" in self.data:
                error = self.data["shifts"][idx][-1]
            else:
                error = np.random.RandomState(seed + 1).uniform(-1, 1)
                error = torch.tensor(error, dtype=torch.float)
            yaw_init = yaw + error * self.cfg.max_init_error_rotation
            range_ = self.cfg.prior_range_rotation or self.cfg.max_init_error_rotation
            data["yaw_prior"] = torch.stack([yaw_init, torch.tensor(range_)])

        if self.cfg.return_gps:
            gps = self.data["gps_position"][idx][:2].numpy()
            xy_gps = self.tile_managers[scene].projection.project(gps)
            data["uv_gps"] = torch.from_numpy(canvas.to_uv(xy_gps)).float()
            data["accuracy_gps"] = torch.tensor(
                min(self.cfg.accuracy_gps, self.cfg.crop_size_meters)
            )

        if "chunk_index" in self.data:
            data["chunk_id"] = (scene, seq, self.data["chunk_index"][idx])

        return {
            **data,
            "image": image,
            "points": pts,
            "voxels": voxels.float(),  
            "coors": coors.int(),  
            "num_points_per_voxel": num_points_per_voxel.int(),
            "valid": valid,
            "camera": cam,
            "canvas": canvas,
            "map": torch.from_numpy(np.ascontiguousarray(raster)).long(),
            "uv": torch.from_numpy(uv_gt).float(),  # TODO: maybe rename to uv?
            "uv_init": torch.from_numpy(uv_init).float(),  # TODO: maybe rename to uv?
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "pixels_per_meter": torch.tensor(canvas.ppm).float(),
        }

    def process_image(self, image, cam, roll, pitch, seed):
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )
        image, valid = rectify_image(
            image, cam, roll, pitch if self.cfg.rectify_pitch else None
        )
        roll = 0.0
        if self.cfg.rectify_pitch:
            pitch = 0.0

        if self.cfg.target_focal_length is not None:
            # resize to a canonical focal length
            factor = self.cfg.target_focal_length / cam.f.numpy()
            size = (np.array(image.shape[-2:][::-1]) * factor).astype(int)
            image, _, cam, valid = resize_image(image, size, camera=cam, valid=valid)
            size_out = self.cfg.resize_image
            if size_out is None:
                # round the edges up such that they are multiple of a factor
                stride = self.cfg.pad_to_multiple
                size_out = (np.ceil((size / stride)) * stride).astype(int)
            # crop or pad such that both edges are of the given size
            image, valid, cam = pad_image(
                image, size_out, cam, valid, crop_and_center=True
            )
        elif self.cfg.resize_image is not None:
            image, _, cam, valid = resize_image(
                image, self.cfg.resize_image, fn=max, camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                # pad such that both edges are of the given size
                image, valid, cam = pad_image(image, self.cfg.resize_image, cam, valid)

        if self.cfg.reduce_fov is not None:
            h, w = image.shape[-2:]
            f = float(cam.f[0])
            fov = np.arctan(w / f / 2)
            w_new = round(2 * f * np.tan(self.cfg.reduce_fov * fov))
            image, valid, cam = pad_image(
                image, (w_new, h), cam, valid, crop_and_center=True
            )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)
        return image, valid, cam, roll, pitch

    def create_map_mask(self, canvas):
        map_mask = np.zeros(canvas.raster.shape[-2:], bool)
        radius = self.cfg.mask_radius or self.cfg.max_init_error
        mask_min, mask_max = np.round(
            canvas.to_uv(canvas.bbox.center)
            + np.array([[-1], [1]]) * (radius + self.cfg.mask_pad) * canvas.ppm
        ).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask


class MapLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # map
        "num_classes": None,
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        "init_from_gps": False,
        "return_gps": False,
        "force_camera_height": None,
        # pose priors
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 1,
        "prior_range_rotation": None,
        # image preprocessing
        "target_focal_length": None,
        "reduce_fov": None,
        "resize_image": None,
        "pad_to_square": False,  # legacy
        "pad_to_multiple": 32,
        "rectify_pitch": True,
        "augmentation": {
            "rot90": False,
            "flip": False,
            "image": {
                "apply": False,
                "brightness": 0.5,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.5 / 3.14,
            },
        },
    }

    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        tile_managers: Dict[str, TileManager],
        image_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.tile_managers = tile_managers
        self.names = names
        self.image_ext = image_ext

        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))
        self.tfs = tvf.Compose(tfs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        scene, seq, name = self.names[idx]
        if self.cfg.init_from_gps:
            latlon_gps = self.data["gps_position"][idx][:2].clone().numpy()
            xy_w_init = self.tile_managers[scene].projection.project(latlon_gps)
        else:
            xy_w_init = self.data["t_c2w"][idx][:2].clone().double().numpy()

        if "shifts" in self.data:
            yaw = self.data["roll_pitch_yaw"][idx][-1]
            R_c2w = rotmat2d((90 - yaw) / 180 * np.pi).float()
            error = (R_c2w @ self.data["shifts"][idx][:2]).numpy()
        else:
            error = np.random.RandomState(seed).uniform(-1, 1, size=2)
        xy_w_init += error * self.cfg.max_init_error

        bbox_tile = BoundaryBox(
            xy_w_init - self.cfg.crop_size_meters,
            xy_w_init + self.cfg.crop_size_meters,
        )
        return self.get_view(idx, scene, seq, name, seed, bbox_tile)

    def get_view(self, idx, scene, seq, name, seed, bbox_tile):
        data = {
            "index": idx,
            "name": name,
            "scene": scene,
            "sequence": seq,
        }
        cam_dict = self.data["cameras"][scene][seq][self.data["camera_id"][idx]]
        cam = Camera.from_dict(cam_dict).float()

        if "roll_pitch_yaw" in self.data:
            roll, pitch, yaw = self.data["roll_pitch_yaw"][idx].numpy()
        else:
            roll, pitch, yaw = decompose_rotmat(self.data["R_c2w"][idx].numpy())
        image = read_image(self.image_dirs[scene] / (name + self.image_ext))

        if "plane_params" in self.data:
            # transform the plane parameters from world to camera frames
            plane_w = self.data["plane_params"][idx]
            data["ground_plane"] = torch.cat(
                [rotmat2d(deg2rad(torch.tensor(yaw))) @ plane_w[:2], plane_w[2:]]
            )
        if self.cfg.force_camera_height is not None:
            data["camera_height"] = torch.tensor(self.cfg.force_camera_height)
        elif "camera_height" in self.data:
            data["camera_height"] = self.data["height"][idx].clone()

        # raster extraction
        canvas = self.tile_managers[scene].query(bbox_tile)
        xy_w_gt = self.data["t_c2w"][idx][:2].numpy()
        uv_gt = canvas.to_uv(xy_w_gt)
        uv_init = canvas.to_uv(bbox_tile.center)
        raster = canvas.raster  # C, H, W

        # Map augmentations
        heading = np.deg2rad(90 - yaw)  # fixme
        if self.stage == "train":
            if self.cfg.augmentation.rot90:
                raster, uv_gt, heading = random_rot90(raster, uv_gt, heading, seed)
            if self.cfg.augmentation.flip:
                image, raster, uv_gt, heading = random_flip(
                    image, raster, uv_gt, heading, seed
                )
        yaw = 90 - np.rad2deg(heading)  # fixme

        image, valid, cam, roll, pitch = self.process_image(
            image, cam, roll, pitch, seed
        )

        # Create the mask for prior location
        if self.cfg.add_map_mask:
            data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas))

        if self.cfg.max_init_error_rotation is not None:
            if "shifts" in self.data:
                error = self.data["shifts"][idx][-1]
            else:
                error = np.random.RandomState(seed + 1).uniform(-1, 1)
                error = torch.tensor(error, dtype=torch.float)
            yaw_init = yaw + error * self.cfg.max_init_error_rotation
            range_ = self.cfg.prior_range_rotation or self.cfg.max_init_error_rotation
            data["yaw_prior"] = torch.stack([yaw_init, torch.tensor(range_)])

        if self.cfg.return_gps:
            gps = self.data["gps_position"][idx][:2].numpy()
            xy_gps = self.tile_managers[scene].projection.project(gps)
            data["uv_gps"] = torch.from_numpy(canvas.to_uv(xy_gps)).float()
            data["accuracy_gps"] = torch.tensor(
                min(self.cfg.accuracy_gps, self.cfg.crop_size_meters)
            )

        if "chunk_index" in self.data:
            data["chunk_id"] = (scene, seq, self.data["chunk_index"][idx])

        return {
            **data,
            "image": image,
            "valid": valid,
            "camera": cam,
            "canvas": canvas,
            "map": torch.from_numpy(np.ascontiguousarray(raster)).long(),
            "uv": torch.from_numpy(uv_gt).float(),  # TODO: maybe rename to uv?
            "uv_init": torch.from_numpy(uv_init).float(),  # TODO: maybe rename to uv?
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "pixels_per_meter": torch.tensor(canvas.ppm).float(),
        }

    def process_image(self, image, cam, roll, pitch, seed):
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )
        image, valid = rectify_image(
            image, cam, roll, pitch if self.cfg.rectify_pitch else None
        )
        roll = 0.0
        if self.cfg.rectify_pitch:
            pitch = 0.0

        if self.cfg.target_focal_length is not None:
            # resize to a canonical focal length
            factor = self.cfg.target_focal_length / cam.f.numpy()
            size = (np.array(image.shape[-2:][::-1]) * factor).astype(int)
            image, _, cam, valid = resize_image(image, size, camera=cam, valid=valid)
            size_out = self.cfg.resize_image
            if size_out is None:
                # round the edges up such that they are multiple of a factor
                stride = self.cfg.pad_to_multiple
                size_out = (np.ceil((size / stride)) * stride).astype(int)
            # crop or pad such that both edges are of the given size
            image, valid, cam = pad_image(
                image, size_out, cam, valid, crop_and_center=True
            )
        elif self.cfg.resize_image is not None:
            image, _, cam, valid = resize_image(
                image, self.cfg.resize_image, fn=max, camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                # pad such that both edges are of the given size
                image, valid, cam = pad_image(image, self.cfg.resize_image, cam, valid)

        if self.cfg.reduce_fov is not None:
            h, w = image.shape[-2:]
            f = float(cam.f[0])
            fov = np.arctan(w / f / 2)
            w_new = round(2 * f * np.tan(self.cfg.reduce_fov * fov))
            image, valid, cam = pad_image(
                image, (w_new, h), cam, valid, crop_and_center=True
            )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)
        return image, valid, cam, roll, pitch

    def create_map_mask(self, canvas):
        map_mask = np.zeros(canvas.raster.shape[-2:], bool)
        radius = self.cfg.mask_radius or self.cfg.max_init_error
        mask_min, mask_max = np.round(
            canvas.to_uv(canvas.bbox.center)
            + np.array([[-1], [1]]) * (radius + self.cfg.mask_pad) * canvas.ppm
        ).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask
