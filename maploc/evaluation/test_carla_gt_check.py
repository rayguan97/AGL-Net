from pathlib import Path
import torch
import yaml
from torchmetrics import MetricCollection
from omegaconf import OmegaConf as OC
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pytorch_lightning import seed_everything
import cv2
## main.py
import os
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
import math
import json
from tqdm import tqdm

import maploc
from maploc.data import CarlaDataModule
from maploc.data.torch import unbatch_to_device
from maploc.module import GenericModule
from maploc.models.metrics import Location2DError, AngleError, LateralLongitudinalError
from maploc.evaluation.run import resolve_checkpoint_path
from maploc.evaluation.viz import plot_example_single

from maploc.models.voting import argmax_xyr, fuse_gps
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images, features_to_RGB, save_plot, add_text
from maploc.utils.viz_localization import likelihood_overlay, plot_pose, plot_dense_rotations, add_circle_inset


from maploc.data.carla.utils import( 
    load_point_cloud, 
    overlay_points_on_image,
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


# lidar_param = [1.3, 0.0, 2.5, 0.0, 0.0, -90.0]
# # lidar_param = [0, 0.0, 0, 0.0, 0.0, -90.0]

# # cam_extrinsic_param = [0.0, 0.0, 1200, 0.0, -90.0, 0.0]
# cam_extrinsic_param = [0.0, 0.0, 1200, 0.0, 0.0, 0.0]
# cam_intrinsic_param = [5000, 5000, 20]


# def get_transformation_from_ego(x, y, z, roll, pitch, yaw):
#     # Convert angles from degrees to radians
#     roll = math.radians(roll)
#     pitch = math.radians(pitch)
#     yaw = math.radians(yaw)

#     # Rotation matrices for each axis
#     Rx = np.array([[1, 0, 0],
#                 [0, math.cos(roll), -math.sin(roll)],
#                 [0, math.sin(roll), math.cos(roll)]])

#     Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
#                 [0, 1, 0],
#                 [-math.sin(pitch), 0, math.cos(pitch)]])

#     Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
#                 [math.sin(yaw), math.cos(yaw), 0],
#                 [0, 0, 1]])

#     # Combined rotation matrix
#     R = Rz @ Ry @ Rx

#     # Translation matrix
#     T = np.array([x, y, z])

#     # 4x4 transformation matrix
#     transformation_matrix = np.eye(4)
#     transformation_matrix[:3, :3] = R
#     transformation_matrix[:3, 3] = T

#     return transformation_matrix

# def get_trans_matrix(x, y):
#     return get_transformation_from_ego(x, y, 0, 0, 0, 0)

# def get_rot_matrix(theta):
#     return get_transformation_from_ego(0, 0, 0, 0, 0, math.degrees(theta))


# def get_cam_intrinsic(width, height, fov):
#     f_x = width / (2 * math.tan(math.radians(fov) / 2))
#     f_y = f_x  # Assuming square pixels
#     c_x = width / 2  # Principal point (optical center) x-coordinate
#     c_y = height / 2  # Principal point (optical center) y-coordinate

#     intrinsic_matrix = np.array([[f_x, 0, c_x],
#                                 [0, f_y, c_y],
#                                 [0, 0, 1]])
#     return intrinsic_matrix

# def load_point_cloud(point_cloud_path):
#     # Load a 3D point cloud file
#     # point_cloud = o3d.io.read_point_cloud(point_cloud_path)

#     pts = np.load(point_cloud_path, allow_pickle=True)[1][:,:3]
#     # from IPython import embed;embed()
#     assert pts.shape[1] == 3

#     # Create a PointCloud object from the NumPy array
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(pts)
#     return point_cloud

# def overlay_points_on_image(image, points_2d, color=((255, 0, 0)), r=1):
#     # Overlay the 2D points on the satellite image
#     # For simplicity, let's just draw points on the image
#     # lst = []
#     im = image.copy().astype(np.int32)
#     for point in points_2d:
#         if 0 <= int(point[0]) < image.shape[0] and 0 <= int(point[1]) < image.shape[1]:
#             # from IPython import embed;embed()   
#             cv2.circle(im, (int(point[0]), int(point[1])), r, color, -1)
#             # try:
#             #     cv2.circle(im, (int(point[0]), int(point[1])), 1, color, -1)
#             # except:
#             #     from IPython import embed;embed()

#     # print(lst)
#     # from IPython import embed;embed()
#     return im



torch.set_grad_enabled(False)

conf = OC.load(Path(maploc.__file__).parent / 'conf/data/carla.yaml')
conf = OC.merge(conf, OC.create(yaml.full_load("""
loading:
    val: {batch_size: 1, num_workers: 0}
    train: ${.val}
aug_scale: 0
# max_init_error_rotation: null  # to remove any prior on the rotation
# max_init_error: 64  # default: 20 (meters)
# add_map_mask: false  # remove to search in the entire tile
""")))
OC.resolve(conf)
dataset = CarlaDataModule(conf)
dataset.prepare_data()
dataset.setup()
sampler = None

loader = dataset.dataloader("test", shuffle=sampler is None, sampler=sampler)

for i, item in zip(tqdm(range(len(loader))), loader):
    # continue
    index = item["index"]
    route_dir = item["route_dir"]
    aerial_idx = item["aerial_idx"]
    ground_idx = item["ground_idx"]

    map = item["map"]

    uv_offset = item["uv_offset"]
    pts_in_uv = item["pts_in_uv"]

    uv_init = item["uv_init"]
    theta_init = item["theta_init"]

    uv_gt = item["uv_gt"]
    theta_gt = item["theta_gt"]


    root = "/home/rayguan/scratch_ssd/projects/carla/transfuser/results/"

    sub_folder = route_dir[0]


    # # Main execution
    # point_cloud_path = os.path.join(root, sub_folder, "lidar/" + ground_idx[0] + ".npy")
    # ego_measurement_path = os.path.join(root, sub_folder, "measurements/" + ground_idx[0] + ".json")
    # ego_measurement = json.load(open(ego_measurement_path))


    # satellite_image_path = os.path.join(root, sub_folder, "topdown_rgb/" + aerial_idx[0] + ".png")
    # cam_cur_measurement_path = os.path.join(root, sub_folder, "measurements/" + aerial_idx[0] + ".json")
    # cam_cur_measurement = json.load(open(cam_cur_measurement_path))

    # ego_to_lidar = get_transformation_from_ego(*lidar_param)

    # ego_to_cur1 = get_rot_matrix(ego_measurement["theta"])
    # ego_to_cur2 = get_trans_matrix(ego_measurement["y"], -1 * ego_measurement["x"])
    # ego_to_cur3 = get_trans_matrix(-1 * cam_cur_measurement["y"], cam_cur_measurement["x"])
    # ego_to_cur4 = get_rot_matrix(-1 * cam_cur_measurement["theta"])
    # ego_to_cur = np.matmul(ego_to_cur4, 
    #                         np.matmul(ego_to_cur3,
    #                                     np.matmul(ego_to_cur2,
    #                                             ego_to_cur1)))
    
    # cur_to_cam = get_transformation_from_ego(*cam_extrinsic_param)
    # cam_intrinsic = get_cam_intrinsic(*cam_intrinsic_param)


    # point_cloud = load_point_cloud(point_cloud_path)
    # # https://github.com/carla-simulator/carla/issues/392
    # # carla_trans = np.eye(4)
    # # Transform(Rotation(yaw=90), Scale(z=-1)) 
    # carla_trans = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # # Transform(Rotation(pitch=180)) 
    # to_normal_pc_trans = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # # carla_rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # # carla_scale = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # # carla_trans = np.matmul(carla_scale, carla_rotation)

    # # from IPython import embed;embed()
    # # o3d.visualization.draw_geometries([point_cloud])
    # # viewer = o3d.visualization.Visualizer()
    # # viewer.create_window()
    # # viewer.add_geometry(point_cloud)
    # # opt = viewer.get_render_option()
    # # opt.show_coordinate_frame = True
    # # viewer.run()
    # # viewer.destroy_window()

    # point_cloud.transform(carla_trans)
    # point_cloud.transform(to_normal_pc_trans)

    # point_cloud.transform(np.linalg.inv(to_normal_pc_trans))
    # point_cloud.transform(np.linalg.inv(ego_to_lidar))
    # point_cloud.transform(ego_to_cur)
    # point_cloud.transform(cur_to_cam)

    # final_transformation = np.matmul(cur_to_cam, np.matmul(ego_to_cur, np.matmul(np.linalg.inv(ego_to_lidar), np.linalg.inv(to_normal_pc_trans))))
    # # point_cloud.transform(final_transformation)

    # # from IPython import embed;embed()
    # # point_cloud = np.asarray(point_cloud.points)
    # # lst = [int(a) for a in  point_cloud.T[2]]
    # # np.unique( lst, return_counts=True)

    # point_cloud = np.asarray(point_cloud.points)

    # points_2d_homogeneous = cam_intrinsic @ point_cloud.T

    # points_2d =  points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

    # from IPython import embed;embed()
    overlay_image = overlay_points_on_image(np.array(map[0]), np.array(pts_in_uv[0]))
    overlay_image = overlay_points_on_image(overlay_image, np.array(uv_gt), (0, 255, 0), r=10)
    if not ( (uv_gt[0][0] > 0) and (uv_gt[0][0] < np.array(map[0]).shape[0]) and \
            (uv_gt[0][1] > 0) and (uv_gt[0][1] < np.array(map[0]).shape[1]) ):
        print(uv_gt[0])
        print(route_dir)
        print(aerial_idx)
        print(ground_idx)
        from IPython import embed;embed()
        

    plt.imshow(overlay_image)
    plt.show()



    continue


