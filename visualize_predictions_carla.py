#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import torch
import yaml
from torchmetrics import MetricCollection
from omegaconf import OmegaConf as OC
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pytorch_lightning import seed_everything
import sys
import os
from tqdm import tqdm 

import maploc
from maploc.data import CarlaDataModule
from maploc.data.torch import unbatch_to_device
from maploc.module import GenericModule
from maploc.models.metrics import Location2DError, AngleError, LateralLongitudinalError
from maploc.evaluation.run import resolve_checkpoint_path
from maploc.evaluation.viz import plot_example_single_rgb, plot_example_single_rgb_more, plot_example_single_rgb_paper_fig

from maploc.models.voting import argmax_xyr, fuse_gps
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images, features_to_RGB, save_plot, add_text
from maploc.utils.viz_localization import likelihood_overlay, plot_pose, plot_dense_rotations, add_circle_inset

torch.set_grad_enabled(False);
plt.rcParams.update({'figure.max_open_warning': 0})

experiment = "xxxxx/checkpoint-epoch=04.ckpt"
save_dir = "./saved_figs/Orienter_base_carla_x4/"

conf = OC.load(Path(maploc.__file__).parent / 'conf/data/carla.yaml')
# conf = OC.load(Path(maploc.__file__).parent / 'conf/data/carla_small.yaml')
conf = OC.merge(conf, OC.create(yaml.full_load("""
loading:
    val: {batch_size: 1, num_workers: 0}
    train: ${.val}
# max_init_error_rotation: null  # to remove any prior on the rotation
# max_init_error: 64  # default: 20 (meters)
# add_map_mask: false  # remove to search in the entire tile
""")))
OC.resolve(conf)
dataset = CarlaDataModule(conf)
dataset.prepare_data()
dataset.setup()
sampler = None


# experiment = "nexus/Orienter_model1_carla_x4_small_attn_scaleaug7/checkpoint-epoch=13.ckpt"
# experiment = "orienternet_mgl.ckpt"
# experiment = "experiment_name"  # find the best checkpoint
# experiment = "experiment_name/checkpoint-step=N.ckpt"  # a given checkpoint
path = resolve_checkpoint_path(experiment)
print(path)
# cfg = {'model': {"num_rotations": 360, "apply_map_prior": True}}
cfg = {}
model = GenericModule.load_from_checkpoint(
    path, strict=True, find_best=not experiment.endswith('.ckpt'), cfg=cfg)
model = model.eval().cuda()




out_dir = save_dir
if (out_dir is not None) and (not os.path.isdir(save_dir)):
    os.makedirs(save_dir)
    
seed_everything(25) # best = 25
loader = dataset.dataloader("test", shuffle=sampler is None, sampler=sampler)
metrics = MetricCollection(model.model.metrics()).to(model.device)
# metrics["loc_max_error"] = Location2DError("uv_max", model.cfg.data.pixel_per_meter, "uv_gt")
# metrics["loc_ave_error"] = Location2DError("uv_expectation", model.cfg.data.pixel_per_meter, "uv_gt")
# metrics["directional_error"] = AngleError("yaw_max", "theta_gt")
# for i, batch in zip(range(5), loader):
for i, batch in zip(tqdm(range(len(loader))), loader):
    lp_uvt = lp_uv = loss = results = pred = data = batch_ = None
    
    batch_ = model.transfer_batch_to_device(batch, model.device, i)
    pred = model(batch_)
    loss = model.model.loss(pred, batch_)
    results = metrics(pred, batch_)
    # results.pop("xy_expectation_error")
    for k in list(results):
        if "recall" in k:
            results.pop(k)
    print(f'{i} {loss["total"].item():.2f}', {k: np.round(v.cpu().numpy(), 2) for k, v in results.items()})
#     if results["xy_max_error"] < 5:
#         continue

    pred = unbatch_to_device(pred)
    data = unbatch_to_device(batch)
    
    # plot_example_single_rgb(i, model, pred, data, "/home/rayguan/scratch_ssd/projects/carla/transfuser/results/", out_dir=out_dir)
    plot_example_single_rgb_more(i, model, pred, data, results, out_dir=Path(out_dir), vis=False, draw_scale=0.5)
    # plot_example_single_rgb_paper_fig(i, model, pred, data, results, out_dir=Path(out_dir), vis=True, draw_scale=0.5)
    # plot_example_single(i, model, pred, data, results, plot_bev=True, out_dir=out_dir, show_dir_error=True)
    break