from .kitti.dataset import KittiDataModule, KittiLidarModule
from .mapillary.dataset import MapillaryDataModule
from .carla.dataset import CarlaDataModule

modules = {"mapillary": MapillaryDataModule, 
           "kitti": KittiDataModule, 
           "kitti_lidar": KittiLidarModule, 
           "carla": CarlaDataModule}
