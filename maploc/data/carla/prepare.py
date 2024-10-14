
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
from omegaconf import DictConfig, OmegaConf
from opensfm.pygeometry import Camera
from opensfm.pymap import Shot
from opensfm.undistort import (
    perspective_camera_from_fisheye,
    perspective_camera_from_perspective,
)

from ... import logger
from ...rgbmap.tiling import RGBTileManager
from ...utils.geo import BoundaryBox
from ...utils.io import write_json, DATA_URL
from ..utils import decompose_rotmat
from .utils import ( 
    load_point_cloud, 
    get_transformation_from_ego, 
    get_cam_intrinsic,
    get_rot_matrix,
    get_trans_matrix,
    lidar_param, 
    cam_extrinsic_param, 
    cam_intrinsic_param,
    carla_trans,
    to_normal_pc_trans
)

from .dataset import CarlaDataModule

default_cfg = OmegaConf.create(
    {
        "max_image_size": 512,
        "do_legacy_pano_offset": True,
        "min_dist_between_keyframes": 4,
        "tiling": {
            "tile_size": 1024,
            "margin": 0,
            "ppm": 1,
        },
    }
)

ego_to_lidar = get_transformation_from_ego(*lidar_param)
ego_to_cur = None
cur_to_overhead_cam = get_transformation_from_ego(*cam_extrinsic_param)
overhead_cam_intrinsic = get_cam_intrinsic(*cam_intrinsic_param)




# def process_location(
#     data_dir: str,
#     route_dir: str,
#     overhead_dict: dict,
#     # overhead_dir: str,
#     # ground_lst: List[str],
#     cfg: DictConfig,
#     generate_tiles: bool = True,
# ):
#     # dump = {
#     #   overead_idx1: {
#     #       ground_idx1: {dir, transformation, uv coordinates, theta (yaw)}
#     #       ground_idx2: {}
#     #       ground_idx3: {}
#     #       ...
#     #      },
#     #   overead_idx2: {
#     #       ...
#     #   },
#     #   ...
#     # }

#     dump = {}

#     for overhead_dir, ground_lst in overhead_dict.items():

#         overhead_dir = os.path.join(data_dir, route_dir, overhead_dir)
#         route_dir = os.path.join(data_dir, route_dir)

#         overead_idx = overhead_dir.split("/")[-1].split(".")[0]

#         map_data = mpimg.imread(overhead_dir)

#         tile_path = os.path.join(route_dir, overead_idx + "_" + CarlaDataModule.default_cfg["tiles_filename"])
#         if generate_tiles:

#             bbox_data = BoundaryBox([0, 0], list(map_data.shape)[:2])
#             bbox_tiling = bbox_data + cfg.tiling.margin

#             tile_manager = RGBTileManager.from_bbox(
#                 bbox_tiling,
#                 cfg.tiling.ppm,
#                 map_data=map_data,
#                 tile_size=cfg.tiling.tile_size,
#                 oc_intrinsic=overhead_cam_intrinsic,
#             )
#             tile_manager.save(tile_path)

#             # overead_dict["tile_manager"] = None
#         else:
#             # overead_dict["tile_manager"] = None
#             raise NotImplementedError

            
#         ground_dict = {}


#         for ground_idx in ground_lst:

#             ground_dict[ground_idx] = {}
            
#             point_cloud = load_point_cloud(os.path.join(route_dir, "lidar", ground_idx + ".npy"))
#             point_cloud.transform(carla_trans)
#             point_cloud.transform(to_normal_pc_trans)

#             pts = np.asarray(point_cloud.points)

#             ground_dict[ground_idx]["scale_gt"] = np.max(pts) - np.min(pts)
#             ground_dict[ground_idx]["normalized_pts"] = pts / ground_dict[ground_idx]["scale_gt"]


#             ego_measurement_path = os.path.join(route_dir, "measurements", ground_idx + ".json")
#             ego_measurement = json.load(open(ego_measurement_path))

#             cam_cur_measurement_path = os.path.join(route_dir, "measurements", overead_idx + ".json")
#             cam_cur_measurement = json.load(open(cam_cur_measurement_path))

#             ego_to_cur = get_transformation_from_ego(cam_cur_measurement["y"] - ego_measurement["y"], -1 * (cam_cur_measurement["x"] - ego_measurement["x"]), 0.0, 0.0, 0.0, -1 * (math.degrees(cam_cur_measurement["theta"]) - math.degrees(ego_measurement["theta"])))

#             ground_dict[ground_idx]["t_l2oc"] = np.matmul(cur_to_overhead_cam, np.matmul(ego_to_cur, np.matmul(np.linalg.inv(ego_to_lidar), np.linalg.inv(to_normal_pc_trans))))
#             ground_dict[ground_idx]["oc_intrinsic"] = overhead_cam_intrinsic


#             point_cloud.transform(ground_dict[ground_idx]["t_l2oc"])

#             pts_3d = np.asarray(point_cloud.points)

#             points_2d_homogeneous = ground_dict[ground_idx]["oc_intrinsic"] @ pts_3d.T

#             points_2d =  points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

#             ground_dict[ground_idx]["pts_in_uv"] = points_2d.T


#             uv_homogeneous = ground_dict[ground_idx]["oc_intrinsic"] @ ground_dict[ground_idx]["t_l2oc"][:3,-1]
#             ground_dict[ground_idx]["uv_gt"] = uv_homogeneous[:2] / uv_homogeneous[2]

#             # (10, 0, 0) x, y, z - r, g, b - front, left, top
#             front = ground_dict[ground_idx]["t_l2oc"] @ np.array([10,0,0,1])
#             uv_front_homogeneous = ground_dict[ground_idx]["oc_intrinsic"] @ front[:3]
#             ground_dict[ground_idx]["uv_gt_front"] = uv_front_homogeneous[:2] / uv_front_homogeneous[2]

#             ground_dict[ground_idx]["theta_gt"] = math.atan2(
#                 ground_dict[ground_idx]["uv_gt_front"][1] - ground_dict[ground_idx]["uv_gt"][1],
#                 ground_dict[ground_idx]["uv_gt_front"][0] - ground_dict[ground_idx]["uv_gt"][0]
#             )

#             # u, v = math.cos(ground_dict[ground_idx]["theta_gt"]), math.sin(ground_dict[ground_idx]["theta_gt"])
#             # assert round(u / v, 5) == round((ground_dict[ground_idx]["uv_gt_front"][0] - ground_dict[ground_idx]["uv_gt"][0]) / (ground_dict[ground_idx]["uv_gt_front"][1] - ground_dict[ground_idx]["uv_gt"][1]), 5)


#         # overead_dict["ground_data"] = ground_dict

#         # dump[overead_idx] = overead_dict

#         dump[overead_idx] = ground_dict

#     write_json(os.path.join(route_dir, "dump.json"), dump)

#     return dump

def process_location_gt(
    data_dir: str,
    route_dir: str,
    overhead_dict: dict,
    # overhead_dir: str,
    # ground_lst: List[str],
    cfg: DictConfig,
    generate_tiles: bool = True,
):
    # dump = {
    #   overead_idx1: {
    #       ground_idx1: {dir, transformation, uv coordinates, theta (yaw)}
    #       ground_idx2: {}
    #       ground_idx3: {}
    #       ...
    #      },
    #   overead_idx2: {
    #       ...
    #   },
    #   ...
    # }

    dump = {}
    new_dict = {}

    for overhead_key, ground_lst in overhead_dict.items():
        new_dict[overhead_key] = []

        overhead_dir = os.path.join(data_dir, route_dir, overhead_key)
        route_dir = os.path.join(data_dir, route_dir)

        overead_idx = overhead_dir.split("/")[-1].split(".")[0]
        

        # map_data = mpimg.imread(overhead_dir) # (generate a float, but we want np.uint8)
        map_data = np.array(Image.open(overhead_dir))

        tile_path = os.path.join(route_dir, overead_idx + "_" + CarlaDataModule.default_cfg["tiles_filename"])
        if generate_tiles:

            bbox_data = BoundaryBox([0, 0], list(map_data.shape)[:2])
            bbox_tiling = bbox_data + cfg.tiling.margin

            tile_manager = RGBTileManager.from_bbox(
                bbox_tiling,
                cfg.tiling.ppm,
                map_data=map_data,
                tile_size=cfg.tiling.tile_size,
                oc_intrinsic=overhead_cam_intrinsic,
            )
            tile_manager.save(tile_path)

            # overead_dict["tile_manager"] = None
        # else:
        #     # overead_dict["tile_manager"] = None
        #     raise NotImplementedError

            
        ground_dict = {}


        for ground_idx in ground_lst:

            ground_dict[ground_idx] = {}
            
            ## process point cloud
            # point_cloud = load_point_cloud(os.path.join(route_dir, "lidar", ground_idx + ".npy"))
            # point_cloud.transform(carla_trans)
            # point_cloud.transform(to_normal_pc_trans)

            # pts = np.asarray(point_cloud.points)

            # ground_dict[ground_idx]["scale_gt"] = np.max(pts)
            # ground_dict[ground_idx]["normalized_pts"] = pts / ground_dict[ground_idx]["scale_gt"]

            ## process transformation
            ego_measurement_path = os.path.join(route_dir, "measurements", ground_idx + ".json")
            ego_measurement = json.load(open(ego_measurement_path))

            cam_cur_measurement_path = os.path.join(route_dir, "measurements", overead_idx + ".json")
            cam_cur_measurement = json.load(open(cam_cur_measurement_path))

            # ego_to_cur = get_transformation_from_ego(cam_cur_measurement["y"] - ego_measurement["y"], -1 * (cam_cur_measurement["x"] - ego_measurement["x"]), 0.0, 0.0, 0.0, -1 * (math.degrees(cam_cur_measurement["theta"]) - math.degrees(ego_measurement["theta"])))
            ego_to_cur1 = get_rot_matrix(ego_measurement["theta"])
            ego_to_cur2 = get_trans_matrix(ego_measurement["y"], -1 * ego_measurement["x"])
            ego_to_cur3 = get_trans_matrix(-1 * cam_cur_measurement["y"], cam_cur_measurement["x"])
            ego_to_cur4 = get_rot_matrix(-1 * cam_cur_measurement["theta"])

            ego_to_cur = np.matmul(ego_to_cur4, 
                                   np.matmul(ego_to_cur3,
                                             np.matmul(ego_to_cur2,
                                                       ego_to_cur1)))

            # ground_dict[ground_idx]["t_l2oc"] = np.matmul(cur_to_overhead_cam, np.matmul(ego_to_cur, np.matmul(np.linalg.inv(ego_to_lidar), np.linalg.inv(to_normal_pc_trans))))
            ground_dict[ground_idx]["t_l2oc"] = np.matmul(cur_to_overhead_cam, 
                np.matmul(ego_to_cur, 
                          np.matmul(np.linalg.inv(ego_to_lidar), 
                                    np.linalg.inv(to_normal_pc_trans))))
            # ground_dict[ground_idx]["oc_intrinsic"] = overhead_cam_intrinsic

            ## process ground point cloud 
            # point_cloud.transform(ground_dict[ground_idx]["t_l2oc"])

            # pts_3d = np.asarray(point_cloud.points)

            # points_2d_homogeneous = ground_dict[ground_idx]["oc_intrinsic"] @ pts_3d.T

            # points_2d =  points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

            # ground_dict[ground_idx]["pts_in_uv"] = points_2d.T


            uv_homogeneous = overhead_cam_intrinsic @ ground_dict[ground_idx]["t_l2oc"][:3,-1]
            ground_dict[ground_idx]["uv_gt"] = uv_homogeneous[:2] / uv_homogeneous[2]

            if not (0 < ground_dict[ground_idx]["uv_gt"][0] < list(map_data.shape)[0]):
                del ground_dict[ground_idx]
                continue

            if not (0 < ground_dict[ground_idx]["uv_gt"][1] < list(map_data.shape)[1]):
                del ground_dict[ground_idx]
                continue

            new_dict[overhead_key].append(ground_idx)
            # (10, 0, 0) x, y, z - r, g, b - front, left, top
            front = ground_dict[ground_idx]["t_l2oc"] @ np.array([10,0,0,1])
            uv_front_homogeneous = overhead_cam_intrinsic @ front[:3]
            uv_gt_front = uv_front_homogeneous[:2] / uv_front_homogeneous[2]

            ground_dict[ground_idx]["theta_gt"] = math.atan2(
                uv_gt_front[1] - ground_dict[ground_idx]["uv_gt"][1],
                uv_gt_front[0] - ground_dict[ground_idx]["uv_gt"][0]
            )


        # overead_dict["ground_data"] = ground_dict

        # dump[overead_idx] = overead_dict

        dump[overead_idx] = ground_dict

        if len(dump[overead_idx].keys()) < 5:
            del dump[overead_idx]
            del new_dict[overhead_key]

    write_json(os.path.join(route_dir, "dump_gt.json"), dump)

    return dump, new_dict


def update_test_shift(test_dict, test_shift_dict):

    new_dict = {}

    for data_path, data_dict in test_dict.items():
        for overhead_path, lidar_path_lst in data_dict.items():
            for lidar_path in lidar_path_lst:
                if data_path not in new_dict:
                    new_dict[data_path] = {}
                if overhead_path not in new_dict[data_path]:
                    new_dict[data_path][overhead_path] = {}

                new_dict[data_path][overhead_path][lidar_path] = test_shift_dict[data_path][overhead_path][lidar_path]

    return new_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_filename", type=str, default="splits_carla_loc.json")
    parser.add_argument(
        "--data_dir", type=Path, default=CarlaDataModule.default_cfg["data_dir"]
    )
    parser.add_argument("dotlist", nargs="*")

    args = parser.parse_args()

    args.data_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(Path(__file__).parent / args.split_filename, args.data_dir)
    cfg_ = OmegaConf.merge(default_cfg, OmegaConf.from_cli(args.dotlist))

    train_test_split = json.load(open(args.data_dir / args.split_filename))

    # for location in args.locations:
    #     logger.info("Starting processing for location %s.", location)
    #     process_location(
    #         location,
    #         args.data_dir,
    #         args.data_dir / args.split_filename,
    #         args.token,
    #         cfg_,
    #         args.generate_tiles,
    #     )
    train_test_split_new = {}


    for split in ["train", "val", "test"]:
    # for split in ["val", "test", "train"]:
    # for split in ["train"]:
        data = train_test_split[split]
        dumps = {}
        dumps_gt = {}
        
        new_data = {}
        
        for route_dir, overhead_dict in tqdm(data.items()):
            # dump = process_location(args.data_dir, route_dir, overhead_dict, cfg_, True)
            # dump_gt, new_dict = process_location_gt(args.data_dir, route_dir, overhead_dict, cfg_, True)
            dump_gt, new_dict = process_location_gt(args.data_dir, route_dir, overhead_dict, cfg_, False)
            # for overhead_dir, ground_lst in overhead_dict.items():
            #     process_location(args.data_dir, route_dir, overhead_dir, ground_lst, cfg_, True)
            # dumps[route_dir] = dump
            dumps_gt[route_dir] = dump_gt

            new_data[route_dir] = new_dict
    
        # write_json(os.path.join(args.data_dir, split + "_dump.json"), dumps)
        write_json(os.path.join(args.data_dir, split + "_dump_gt.json"), dumps_gt)
        train_test_split_new[split] = new_data


    test_shift_new = update_test_shift(train_test_split["test"], train_test_split["test_shifts"])
    train_test_split_new["test_shifts"] = test_shift_new


    with open(Path(args.data_dir) / args.split_filename, "w") as f:
        json.dump(train_test_split_new, f, indent=2)

