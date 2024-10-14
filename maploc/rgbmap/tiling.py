
import io
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import rtree
import matplotlib.image as mpimg

from ..utils.geo import BoundaryBox
from .raster import RGBCanvas
from skimage.transform import resize

def bbox_to_slice(bbox: BoundaryBox, off_set=[0, 0]):
    slice_ = (slice(int(bbox.min_[0] - off_set[0]), int(bbox.max_[0] - off_set[0])), 
              slice(int(bbox.min_[1] - off_set[1]), int(bbox.max_[1] - off_set[1])))
    return slice_


def round_bbox(bbox: BoundaryBox, origin: np.ndarray, ppm: int):
    bbox = bbox.translate(-origin)
    bbox = BoundaryBox(np.round(bbox.min_ * ppm) / ppm, np.round(bbox.max_ * ppm) / ppm)
    return bbox.translate(origin)

class RGBTileManager:
    def __init__(
        self,
        tiles: Dict,
        bbox: BoundaryBox,
        tile_size: int,
        ppm: int = 1,
        map_data: Optional[Image.Image] = None,
        oc_intrinsic=None,
    ):
        self.origin = bbox.min_
        self.bbox = bbox
        self.tiles = tiles
        self.tile_size = tile_size
        self.ppm = ppm
        self.map_data = map_data
        self.oc_intrinsic = oc_intrinsic
        assert np.all(tiles[0, 0].bbox.min_ == self.origin)
        for tile in tiles.values():
            assert bbox.contains(tile.bbox)

    @classmethod
    def from_bbox(
        cls,
        bbox: BoundaryBox,
        ppm: int,
        map_data: np.ndarray,
        tile_size: int = 128,
        oc_intrinsic=None,
    ):
        # if path is not None and path.is_file():
        
        bounds_x, bounds_y = [
            np.r_[np.arange(min_, max_, tile_size), max_]
            for min_, max_ in zip(bbox.min_, bbox.max_)
        ]
        bbox_tiles = {}
        for i, xmin in enumerate(bounds_x[:-1]):
            for j, ymin in enumerate(bounds_y[:-1]):
                bbox_tiles[i, j] = BoundaryBox(
                    [xmin, ymin], [bounds_x[i + 1], bounds_y[j + 1]]
                )

        tiles = {}
        for ij, bbox_tile in bbox_tiles.items():
            # from IPython import embed;embed()
            canvas = RGBCanvas(bbox_tile, ppm)
            x_min, y_min = bbox_tile.min_[0], bbox_tile.min_[1]
            x_max, y_max = bbox_tile.max_[0], bbox_tile.max_[1]
            canvas.raster = map_data[x_min:x_max, y_min:y_max, :].copy()
            tiles[ij] = canvas

        return cls(tiles, bbox, tile_size, ppm, map_data, oc_intrinsic)

    def query(self, bbox: BoundaryBox, patch_scale=1) -> RGBCanvas:



        # try:
        # except:
        #     from IPython import embed;embed()
        #     print(bbox)
        #     print(self.bbox)
        #     print(bbox_all)
        if patch_scale == 1:

            bbox = round_bbox(bbox, self.bbox.min_, self.ppm)
            canvas = RGBCanvas(bbox, self.ppm)
            raster = np.zeros((canvas.h, canvas.w, 3), np.uint8)
            bbox_all = bbox & self.bbox

            ij_min = np.floor((bbox_all.min_ - self.origin) / self.tile_size).astype(int)
            ij_max = np.ceil((bbox_all.max_ - self.origin) / self.tile_size).astype(int) - 1
            for i in range(ij_min[0], ij_max[0] + 1):
                for j in range(ij_min[1], ij_max[1] + 1):
                    tile = self.tiles[i, j]
                    bbox_select = tile.bbox & bbox
                    slice_query = bbox_to_slice(bbox_select, bbox.min_)
                    slice_tile = bbox_to_slice(bbox_select, [i * self.tile_size, j * self.tile_size])
                    # from IPython import embed;embed()
                    raster[slice_query + (slice(None),)] = tile.raster[slice_tile + (slice(None),)]
        else:
            # print(patch_scale)
            bbox = round_bbox(bbox, self.bbox.min_, self.ppm)
            special_box = BoundaryBox(
                bbox.center - patch_scale * bbox.size / 2,
                bbox.center + patch_scale * bbox.size / 2
                )
            special_box = round_bbox(special_box, self.bbox.min_, self.ppm)
            canvas = RGBCanvas(bbox, self.ppm)

            bbox_all = special_box & self.bbox

            h, w = np.ceil(bbox.size * patch_scale).astype(int)
            raster = np.zeros((canvas.h, canvas.w, 3), np.uint8)

            raster_scaled = np.zeros((h, w, 3), np.uint8)

            # try:
            ij_min = np.floor((bbox_all.min_ - self.origin) / self.tile_size).astype(int)
            ij_max = np.ceil((bbox_all.max_ - self.origin) / self.tile_size).astype(int) - 1
            for i in range(ij_min[0], ij_max[0] + 1):
                for j in range(ij_min[1], ij_max[1] + 1):
                    tile = self.tiles[i, j]
                    bbox_select = tile.bbox & special_box
                    slice_query = bbox_to_slice(bbox_select, bbox_all.min_)
                    slice_tile = bbox_to_slice(bbox_select, [i * self.tile_size, j * self.tile_size])
                    # from IPython import embed;embed()
                    raster_scaled[slice_query + (slice(None),)] = tile.raster[slice_tile + (slice(None),)]
            # except:
            #     from IPython import embed;embed()        
            
            raster[:,:] = resize(raster_scaled, (canvas.h, canvas.w), preserve_range=True, anti_aliasing=True)

        canvas.raster = raster
        return canvas

    def save(self, path: Path):
        dump = {
            "bbox": self.bbox.format(),
            "tile_size": self.tile_size,
            "ppm": self.ppm,
            "tiles_bbox": {},
            "tiles_raster": {},
            "oc_intrinsic": self.oc_intrinsic
        }
        for ij, canvas in self.tiles.items():
            dump["tiles_bbox"][ij] = canvas.bbox.format()
            raster_bytes = io.BytesIO()
            raster = Image.fromarray(canvas.raster.astype(np.uint8))
            raster.save(raster_bytes, format="PNG")
            dump["tiles_raster"][ij] = raster_bytes
        # map_data_bytes = io.BytesIO()
        # mpimg.imsave(map_data_bytes, self.map_data)
        # dump["map_data"] = map_data_bytes
        with open(path, "wb") as fp:
            pickle.dump(dump, fp)

    @classmethod
    def load(cls, path: Path):
        with path.open("rb") as fp:
            dump = pickle.load(fp)
        tiles = {}
        for ij, bbox in dump["tiles_bbox"].items():
            tiles[ij] = RGBCanvas(BoundaryBox.from_string(bbox), dump["ppm"])
            tiles[ij].raster = np.asarray(Image.open(dump["tiles_raster"][ij])).copy()
        # map_data = mpimg.imread(dump["map_data"])
        return cls(
            tiles,
            BoundaryBox.from_string(dump["bbox"]),
            dump["tile_size"],
            dump["ppm"],
            # map_data,
            oc_intrinsic=np.array(dump["oc_intrinsic"])
        )


