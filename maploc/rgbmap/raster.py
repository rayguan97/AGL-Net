# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
import torch

from ..utils.geo import BoundaryBox



class RGBCanvas:
    def __init__(self, bbox: BoundaryBox, ppm: float):
        self.bbox = bbox
        self.ppm = ppm
        self.scaling = bbox.size * ppm
        self.h, self.w = np.ceil(self.scaling).astype(int)
        self.clear()

    def clear(self):
        self.raster = np.zeros((self.h, self.w, 3), np.uint8)

