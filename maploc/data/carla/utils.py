import os
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
import math
import json


# {
# 'type': 'sensor.lidar.ray_cast',
# 'x': 1.3, 'y': 0.0, 'z': 2.5,
# 'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
# 'rotation_frequency': 20,
# 'points_per_second': 1200000,
# 'id': 'lidar'
# },
# {
#     'type': 'sensor.camera.rgb',
#     'x': 0, 'y': 0.0, 'z':1200,
#     'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
#     'width': 5000, 'height': 5000, 'fov': 20,
#     'id': 'rgb_topdown'
# },

lidar_param = [1.3, 0.0, 2.5, 0.0, 0.0, -90.0]
# lidar_param = [0, 0.0, 0, 0.0, 0.0, -90.0]

# cam_extrinsic_param = [0.0, 0.0, 1200, 0.0, -90.0, 0.0]
cam_extrinsic_param = [0.0, 0.0, 1200, 0.0, 0.0, 0.0]
cam_intrinsic_param = [5000, 5000, 20]

# https://github.com/carla-simulator/carla/issues/392
# carla_trans = np.eye(4)
# Transform(Rotation(yaw=90), Scale(z=-1)) 
carla_trans = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# Transform(Rotation(pitch=180)) x, y, z - r, g, b - front, left, top
to_normal_pc_trans = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def get_transformation_from_ego(x, y, z, roll, pitch, yaw):
    # Convert angles from degrees to radians
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])

    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])

    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Translation matrix
    T = np.array([x, y, z])

    # 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    return transformation_matrix

def get_trans_matrix(x, y):
    return get_transformation_from_ego(x, y, 0, 0, 0, 0)

def get_rot_matrix(theta):
    return get_transformation_from_ego(0, 0, 0, 0, 0, math.degrees(theta))

def get_cam_intrinsic(width, height, fov):
    f_x = width / (2 * math.tan(math.radians(fov) / 2))
    f_y = f_x  # Assuming square pixels
    c_x = width / 2  # Principal point (optical center) x-coordinate
    c_y = height / 2  # Principal point (optical center) y-coordinate

    intrinsic_matrix = np.array([[f_x, 0, c_x],
                                [0, f_y, c_y],
                                [0, 0, 1]])
    return intrinsic_matrix

def load_point_cloud(point_cloud_path, filter_up=None, filter_down=None, forward_off=-1.3):
    # Load a 3D point cloud file
    # point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    pts = np.load(point_cloud_path, allow_pickle=True)[1][:,:3]
    # from IPython import embed;embed()
    assert pts.shape[1] == 3

    center_len = 2.5

    x_pts = pts[:,0]
    y_pts = pts[:,1]
    x_idx1 = np.where(x_pts > center_len)[0]
    x_idx2 = np.where(x_pts < -1 * center_len)[0]
    y_idx1 = np.where(y_pts > forward_off + center_len)[0]
    y_idx2 = np.where(y_pts < forward_off + -1 * center_len)[0]
    xy_idx = np.concatenate([x_idx1, x_idx2, y_idx1, y_idx2]).astype(np.int_)

    pts = pts[xy_idx]

    if filter_up or filter_down:
        z_pts = pts[:,-1]
        if filter_up:
            z_idx1 = np.where(z_pts > filter_up)[0]
        else:
            z_idx1 = np.array([])

        if filter_down:
            z_idx2 = np.where(z_pts < filter_down)[0]
        else:
            z_idx2 = np.array([])
        z_idx = np.concatenate([z_idx1, z_idx2]).astype(np.int_)

        pts = pts[z_idx]

    # Create a PointCloud object from the NumPy array
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    return point_cloud

def load_point_cloud_kitti(point_cloud_path):
    # Load a 3D point cloud file
    # point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
    pts = points[:, :3]  # exclude luminance

    # Create a PointCloud object from the NumPy array
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    return point_cloud

def overlay_points_on_image(image, points_2d, color=((255, 0, 0)), r=1):
    # Overlay the 2D points on the satellite image
    # For simplicity, let's just draw points on the image
    # lst = []
    im = image.copy().astype(np.int32)
    # from IPython import embed;embed()   
    for point in points_2d:
        if 0 <= int(point[0]) < image.shape[0] and 0 <= int(point[1]) < image.shape[1]:
            # from IPython import embed;embed()   
            cv2.circle(im, (int(point[0]), int(point[1])), r, color, -1)
            # try:
            #     cv2.circle(im, (int(point[0]), int(point[1])), 1, color, -1)
            # except:
            #     from IPython import embed;embed()

    # print(lst)
    # from IPython import embed;embed()
    return im

def overlay_arrow_on_image(image, start, end, color=((255, 0, 0)), r=1):
    # Overlay the 2D points on the satellite image
    # For simplicity, let's just draw points on the image
    # lst = []
    im = image.copy().astype(np.int32)

    if 0 <= int(start[0]) < image.shape[0] and 0 <= int(start[1]) < image.shape[1] and \
        0 <= int(end[0]) < image.shape[0] and 0 <= int(end[1]) < image.shape[1]:
        # from IPython import embed;embed()   
        cv2.arrowedLine(im, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, r)


    # print(lst)
    # from IPython import embed;embed()
    return im

