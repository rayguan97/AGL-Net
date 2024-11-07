from .kitti.dataset import KittiDataModule, KittiLidarModule
from .carla.dataset import CarlaDataModule

modules = {"kitti": KittiDataModule, 
           "kitti_lidar": KittiLidarModule, 
           "carla": CarlaDataModule}
