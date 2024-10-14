# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from pathlib import Path
from typing import Optional, Tuple
import yaml

from omegaconf import OmegaConf, DictConfig
import maploc

from .. import logger
from ..data import CarlaDataModule
from .run import evaluate_carla


default_cfg_single = OmegaConf.create({})
default_cfg_sequential = OmegaConf.create({})
# For the sequential evaluation, we need to center the map around the GT location,
# since random offsets would accumulate and leave only the GT location with a valid mask.
# This should not have much impact on the results.
# default_cfg_sequential = OmegaConf.create(
#     {
#         "data": {
#             "mask_radius": CarlaDataModule.default_cfg["max_init_error"],
#             "prior_range_rotation": CarlaDataModule.default_cfg[
#                 "max_init_error_rotation"
#             ]
#             + 1,
#             "max_init_error": 0,
#             "max_init_error_rotation": 0,
#         },
#         "chunking": {
#             "max_length": 100,  # about 10s?
#         },
#     }
# )


def run(
    split: str,
    experiment: str,
    cfg: Optional[DictConfig] = None,
    cfg_path: str = 'carla.yaml',
    sequential: bool = False,
    thresholds: Tuple[int] = (1, 3, 5),
    bs=1,
    **kwargs,
):
    # cfg = cfg or {}
    # if isinstance(cfg, dict):
    #     cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.load(Path(maploc.__file__).parent / "conf/data" / cfg_path)
    OmegaConf.resolve(cfg)
    default = default_cfg_sequential if sequential else default_cfg_single
    cfg = OmegaConf.merge(default, cfg)
    cfg.loading["test"] = {"batch_size": bs, "num_workers": bs}
    # cfg.test.batch_size = bs
    # cfg.loading.test.num_workers = bs
    dataset = CarlaDataModule(cfg)

    metrics = evaluate_carla(
        experiment,
        cfg,
        dataset,
        split=split,
        sequential=sequential,
        viz_kwargs=dict(show_dir_error=True, show_masked_prob=False),
        **kwargs,
    )

    keys = ["xy_max_error", "xy_expectation_error", "yaw_max_error", "directional_error"]
    for k in keys:
        rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
        if k == "directional_error":
            rec = rec[::-1]
        logger.info("Recall %s: %s at %s m/°", k, rec, thresholds)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--data_cfg", type=str, default="carla.yaml")
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "val", "train"]
    )
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num", type=int)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.from_cli(args.dotlist)
    run(
        args.split,
        args.experiment,
        cfg,
        cfg_path=args.data_cfg,
        sequential=args.sequential,
        output_dir=args.output_dir,
        bs=args.bs,
        num=args.num,
    )
