# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize, interpolate
from ast import literal_eval as make_tuple
import omegaconf
from omegaconf import OmegaConf

from mmdet3d.models.backbones.pointnet2_sa_ssg import PointNet2SASSG
from mmdet3d.models.voxel_encoders.pillar_encoder import PillarFeatureNet
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter

from mmcv.ops import DeformConv2dPack as DeformConv2d


from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .voting import (
    argmax_xyr,
    argmax_xyr_radius,
    expectation_xyr,
    expectation_xyr_radius,
    conv2d_fft_batchwise,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_radius,
    nll_loss_xyr_smoothed,
    nll_loss_xyr_smoothed_radius,
    TemplateSampler,
    TemplateSamplerRGB,
)
from .map_encoder import MapEncoder
from .map_encoder_rgb import MapEncoderRGB
from .utils import make_grid

from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from .attention import TransformerBlock

class AGLNet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "point_encoder": {
            "in_channels": 3,
            "num_points": "???",
            "radius": "???",
            "num_samples": "???",
            "sa_channels": "???",
            "fp_channels": "???",
            "norm_cfg": dict(type='BN2d'),
            "sa_cfg": dict(type='PointSAModule',
                           pool_mod='max',
                           use_xyz=True,
                           normalize_xyz=True),
        },
        "voxel_encoder": {
            "in_channels": 3,
            "feat_channels": "???",
            "point_cloud_range": "???",
            "with_cluster_center": False,
            "with_voxel_center": False,
        },
        "attn": {
            "num_attn_layers": 2,
            "global_channels": "???",
            "local_channels": 0,
            "num_centers": [32, 16],
            "num_heads": 4,
        },
        "bev_net": "???",
        "latent_dim": "???",
        "lidar_dim": "???",
        "matching_dim": "???",
        "scale_range": [0.5, 10],
        "num_scale_bins": "???",
        "v_min": None,
        "v_max": "???",
        "u_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "crop_size_meters": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 0.01,
        "sigma_r": 0.2,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        self.map_encoder = MapEncoderRGB(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        # from IPython import embed;embed()

        # point_encoder_conf = dict(conf.point_encoder)
        # point_encoder_conf["num_points"] = make_tuple(point_encoder_conf["num_points"])
        # point_encoder_conf["norm_cfg"] = dict(point_encoder_conf["norm_cfg"])
        # point_encoder_conf["sa_cfg"] = dict(point_encoder_conf["sa_cfg"])

        # self.point_encoder = PointNet2SASSG(**point_encoder_conf)

        voxel_encoder_conf = dict(conf.voxel_encoder)
        voxel_encoder_conf["feat_channels"] = make_tuple(voxel_encoder_conf["feat_channels"])

        self.voxel_encoder = PillarFeatureNet(**voxel_encoder_conf)
        self.pillar_scatter = PointPillarsScatter(in_channels=voxel_encoder_conf["feat_channels"][0], 
                                                  output_shape=(conf.crop_size_meters*2, conf.crop_size_meters*2))
        self.attn = TransformerBlock(conf.attn)
        # delta = 1 / ppm
        # grid_xz = make_grid(
        #     conf.u_max * 2 + delta, conf.v_max * 2 + delta, step_y=delta, step_x=delta, orig_x=-conf.u_max, orig_y=-conf.v_max, y_up=False
        # )
        # self.register_buffer("grid_xz", grid_xz, persistent=False)

        # from IPython import embed;embed()

        self.template_sampler = TemplateSamplerRGB(
            max(conf.u_max, conf.v_max) * 2 + 1, ppm, conf.num_rotations, optimize=True
        )

        self.skeleton_mask_classifier = torch.nn.Sequential(
                DeformConv2d(conf.matching_dim, conf.matching_dim * 4, kernel_size=3, padding=1),
                DeformConv2d(conf.matching_dim * 4, conf.matching_dim * 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                DeformConv2d(conf.matching_dim * 4, conf.matching_dim, kernel_size=3, padding=1),
            )
        self.skeleton_mask_classifier_head = torch.nn.Linear(conf.matching_dim, 2)

        self.lidar_mask_classifier = torch.nn.Sequential(
                DeformConv2d(conf.matching_dim, conf.matching_dim * 4, kernel_size=3, padding=1),
                DeformConv2d(conf.matching_dim * 4, conf.matching_dim * 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                DeformConv2d(conf.matching_dim * 4, conf.matching_dim, kernel_size=3, padding=1),
            )
        self.lidar_mask_classifier_head = torch.nn.Linear(conf.matching_dim, 2)

        self.scale_range = conf.scale_range
        self.mask_scale_classifier = torch.nn.Sequential(
            DeformConv2d(conf.matching_dim, conf.latent_dim, kernel_size=4, stride=4),
            DeformConv2d(conf.latent_dim, conf.latent_dim, kernel_size=4, stride=4),
            torch.nn.ReLU(),
            DeformConv2d(conf.latent_dim, conf.latent_dim, kernel_size=4, stride=4),
            torch.nn.ReLU(),
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(conf.crop_size_meters * conf.crop_size_meters * conf.latent_dim // (4**5), conf.latent_dim),
            torch.nn.Linear(conf.latent_dim, conf.num_scale_bins),
        )
        # self.num_scale_bins = conf.num_scale_bins
        self.scale_bins = torch.nn.parameter.Parameter(
            torch.arange(self.scale_range[0], self.scale_range[1], (self.scale_range[1] - self.scale_range[0])/conf.num_scale_bins).unsqueeze(-1),
            requires_grad=False)
        # self.scale_bins = torch.nn.parameter.Parameter(
        #     torch.arange(1, 0, -1/conf.num_scale_bins),
        #     requires_grad=False)       
        

        # self.register_buffer("scale_bins", scale_bins)

        self.scale_mse_loss = torch.nn.MSELoss(reduction='none')
        self.skeleton_binary_ce_loss = torch.nn.BCELoss(reduction='none')

        # self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)

    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels
        # in each template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores

    def _forward(self, data):

    
        pred = {}

        level = 0
        f_init_map = self.image_encoder({"image": data["map"].permute(0, 3, 1, 2)})["feature_maps"][level]
        pred_map = pred["map"] = self.map_encoder(f_init_map)
        f_map = pred_map["map_features"][0]
        # batch_size, output_dim, map_shape, map_shape
        # prev v.s. now
        # torch.Size([9, 8, 256, 256]) v.s. torch.Size([9, 128, 256, 256])

        # f_pts = self.point_encoder(data["point_cloud_norm"])
        # ['fp_xyz', 'fp_features', 'fp_indices', 'sa_xyz', 'sa_features', 'sa_indices']

        # from IPython import embed;embed()

        bs, num_voxel, num_pts, f_c = data["voxels"].shape
        # f_bev = self.voxel_encoder(f_pts["fp_features"][-2], f_pts["fp_features"][-2].shape[1], f_pts["fp_xyz"][-2])
        f_bev = self.voxel_encoder(data["voxels"].reshape(-1, num_pts, 3), 
                                   data["num_points_per_voxel"].reshape(-1), 
                                   data["coors"].reshape(-1, 3))
        f_bev = f_bev.reshape(bs, num_voxel, -1)
        coors = data["coors"]

        f_bev = self.attn(f_bev.permute(0,2,1)).permute(0, 2, 1)

        coor_batch_idx = torch.arange(bs, device=coors.device).view(bs, 1, 1)
        coor_batch_idx = coor_batch_idx.expand(-1, num_voxel, 1)

        coors_with_idx = torch.cat((coor_batch_idx, coors), dim=2)
        coors_with_idx = coors_with_idx.view(-1, 4)

        f_bev_flat = f_bev.reshape(-1, f_bev.shape[-1])

        f_bev = self.pillar_scatter(f_bev_flat, coors_with_idx, bs)

        # from IPython import embed;embed()

        ## TODO ADD style transfer


        pred_bev = {}
        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]
        
        valid_bev = torch.ones((bs, f_bev.shape[-2], f_bev.shape[-1]), dtype=torch.bool, device=f_bev.device)

        # f_bev -- B, N, H, W
        # torch.Size([9, 8, 64, 129])
        # f_map -- B, N, map_H, map_W
        # valid_bev == pred_bev.get("confidence") -- B, H, W
        scores = self.exhaustive_voting(
            f_bev, f_map, valid_bev, pred_bev.get("confidence")
        )
        scores = scores.moveaxis(1, -1)  # B,map_H,map_W,n_rot
        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        # pred["scores_unmasked"] = scores.clone()
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)

        # from IPython import embed;embed()

        # with torch.no_grad():
        #     uvr_max = argmax_xyr_radius(scores).to(scores)

        # two stages
        # use point net to learn scales v.s. voxels
        # points = data["point_cloud"]
        # f_pts = self.point_encoder(points)
        # ['fp_xyz', 'fp_features', 'fp_indices', 'sa_xyz', 'sa_features', 'sa_indices']

        # from IPython import embed;embed()
        # indices = scores.flatten(-3).max(-1).indices
        # width, num_rotations = scores.shape[-2:]
        # wr = width * num_rotations
        # y = torch.div(indices, wr, rounding_mode="floor")
        # x = torch.div(indices % wr, num_rotations, rounding_mode="floor")
        # angle_index = indices % num_rotations
        # scores[:,y,x,angle_index]
        # from IPython import embed;embed()

        f_map_skeleton_mask1 = self.skeleton_mask_classifier(f_map)
        f_map_skeleton_mask = self.skeleton_mask_classifier_head(f_map_skeleton_mask1.moveaxis(1, -1))
        f_map_skeleton_mask_pred = torch.sigmoid(f_map_skeleton_mask)
        f_map_skeleton_mask_pred = f_map_skeleton_mask_pred.moveaxis(-1, 1)
        # skeleton_mask_pred = torch.argmax(f_map_skeleton_mask, -1)
        # scales = torch.argmax(self.mask_scale_classifier(f_map_skeleton_mask1), -1)
        scales = torch.softmax(self.mask_scale_classifier(f_map_skeleton_mask1), -1)
        # from IPython import embed;embed()
        scale_f_map = torch.matmul(scales, self.scale_bins)
        # scale_f_map = torch.tensor([0.9 for _ in self.scale_bins[scales]], device=scales.device)

        f_bev_skeleton_mask = self.lidar_mask_classifier(f_bev)
        f_bev_skeleton_mask = self.lidar_mask_classifier_head(f_bev_skeleton_mask.moveaxis(1, -1))
        f_bev_skeleton_mask_pred = torch.sigmoid(f_bev_skeleton_mask)
        # f_bev_skeleton_mask_pred = torch.argmax(f_bev_skeleton_mask, -1)
        # scale_f_bev = self.mask_scale_classifier(f_bev_skeleton_mask_pred)

        # #####
        # valid_bev2 = torch.zeros(f_bev_skeleton_mask_pred.shape[:-1], dtype=torch.bool, device=f_bev_skeleton_mask_pred.device)
        # h, w = f_bev_skeleton_mask_pred.shape[1:3]
        # f_bev_skeleton_mask_pred_scaled = f_bev_skeleton_mask_pred.moveaxis(-1, 1)
        # for b_idx in range(bs):
        #     new_h, new_w = int(h / scale_f_map[b_idx]), int(w / scale_f_map[b_idx])
        #     # try:
        #     if scale_f_map[b_idx] < 1:
        #         s_h = int((new_h - h) / 2)
        #         s_w = int((new_w - w) / 2)
        #         # f_bev_skeleton_mask_pred_scaled[b_idx, :, :, :] = Resize((int(new_h), int(new_w)), antialias=True)(f_bev_skeleton_mask_pred[b_idx])[:, s_h: s_h + h, s_w: s_w + w]
        #         valid_bev2[b_idx] = torch.ones((h, w), dtype=torch.bool, device=f_bev_skeleton_mask_pred.device)
        #     else:
        #         s_h = int((h - new_h) / 2)
        #         s_w = int((w - new_w) / 2)
        #         # f_bev_skeleton_mask_pred_scaled[b_idx, :, s_h: s_h + new_h, s_w: s_w + new_w] = Resize((int(new_h), int(new_w)), antialias=True)(f_bev_skeleton_mask_pred[b_idx])
        #         valid_bev2[b_idx, s_h: s_h + new_h, s_w: s_w + new_w] = torch.ones((new_h, new_w), dtype=torch.bool, device=f_bev_skeleton_mask_pred.device)
        #     # except:
        #     #     from IPython import embed;embed()
        # #####


        # #####
        # valid_bev2 = torch.zeros(f_bev_skeleton_mask_pred.shape[:-1], dtype=torch.bool, device=f_bev_skeleton_mask_pred.device)
        # h, w = f_bev_skeleton_mask_pred.shape[1:3]
        # f_bev_skeleton_mask_pred = f_bev_skeleton_mask_pred.moveaxis(-1, 1)
        # f_bev_skeleton_mask_pred_scaled = torch.zeros(f_bev_skeleton_mask_pred.shape, device=f_bev_skeleton_mask_pred.device)
        # for b_idx in range(bs):
        #     new_h, new_w = int(h / scale_f_map[b_idx]), int(w / scale_f_map[b_idx])
        #     # try:
        #     if scale_f_map[b_idx] < 1:
        #         s_h = int((new_h - h) / 2)
        #         s_w = int((new_w - w) / 2)
        #         # f_bev_skeleton_mask_pred_scaled[b_idx, :, :, :] = Resize((int(new_h), int(new_w)), antialias=True)(f_bev_skeleton_mask_pred[b_idx])[:, s_h: s_h + h, s_w: s_w + w]
        #         f_bev_skeleton_mask_pred_scaled[b_idx, :, :, :] = interpolate(f_bev_skeleton_mask_pred[b_idx:b_idx+1], size=(int(new_h), int(new_w)), mode='nearest')[0, :, s_h: s_h + h, s_w: s_w + w]
        #         valid_bev2[b_idx] = torch.ones((h, w), dtype=torch.bool, device=f_bev_skeleton_mask_pred.device)
        #     else:
        #         s_h = int((h - new_h) / 2)
        #         s_w = int((w - new_w) / 2)
        #         # f_bev_skeleton_mask_pred_scaled[b_idx, :, s_h: s_h + new_h, s_w: s_w + new_w] = Resize((int(new_h), int(new_w)), antialias=True)(f_bev_skeleton_mask_pred[b_idx])
        #         f_bev_skeleton_mask_pred_scaled[b_idx, :, s_h: s_h + new_h, s_w: s_w + new_w] = interpolate(f_bev_skeleton_mask_pred[b_idx:b_idx+1], size=(int(new_h), int(new_w)), mode='nearest')[0]
        #         valid_bev2[b_idx, s_h: s_h + new_h, s_w: s_w + new_w] = torch.ones((new_h, new_w), dtype=torch.bool, device=f_bev_skeleton_mask_pred.device)
        #     # except:
        #     #     from IPython import embed;embed()
        # #####

        # # f_bev_skeleton_mask_pred_scaled = torch.clamp(f_bev_skeleton_mask_pred_scaled)
        # scores2 = self.exhaustive_voting(
        #     f_bev_skeleton_mask_pred_scaled, f_map_skeleton_mask_pred, valid_bev2, pred_bev.get("confidence")
        #     # f_bev scaled down, skeleton_mask_pred, valid_bev, pred_bev.get("confidence")
        # )
        # scores2 = scores2.moveaxis(1, -1)  # B,map_H,map_W,n_rot
        # scores = 0.5 * scores + scores2

        #####
        with torch.autocast("cuda", enabled=False):
            h, w = f_bev_skeleton_mask_pred.shape[1:3]
            f_bev_skeleton_mask_pred = f_bev_skeleton_mask_pred.moveaxis(-1, 1)
            f_bev_skeleton_mask_pred_scaled = torch.zeros(f_bev_skeleton_mask_pred.shape, device=f_bev_skeleton_mask_pred.device)
            for b_idx in range(bs):
                new_h, new_w = int(h / scale_f_map[b_idx]), int(w / scale_f_map[b_idx])
                # try:
                if scale_f_map[b_idx] < 1:
                    s_h = int((new_h - h) / 2)
                    s_w = int((new_w - w) / 2)
                    # f_bev_skeleton_mask_pred_scaled[b_idx, :, :, :] = Resize((int(new_h), int(new_w)), antialias=True)(f_bev_skeleton_mask_pred[b_idx])[:, s_h: s_h + h, s_w: s_w + w]
                    f_bev_skeleton_mask_pred_scaled[b_idx, :, :, :] = interpolate(f_bev_skeleton_mask_pred[b_idx:b_idx+1], size=(int(new_h), int(new_w)), mode='nearest')[0, :, s_h: s_h + h, s_w: s_w + w]
                    # valid_bev2[b_idx] = torch.ones((h, w), dtype=torch.bool, device=f_bev_skeleton_mask_pred.device)
                else:
                    s_h = int((h - new_h) / 2)
                    s_w = int((w - new_w) / 2)
                    # f_bev_skeleton_mask_pred_scaled[b_idx, :, s_h: s_h + new_h, s_w: s_w + new_w] = Resize((int(new_h), int(new_w)), antialias=True)(f_bev_skeleton_mask_pred[b_idx])
                    f_bev_skeleton_mask_pred_scaled[b_idx, :, s_h: s_h + new_h, s_w: s_w + new_w] = interpolate(f_bev_skeleton_mask_pred[b_idx:b_idx+1], size=(int(new_h), int(new_w)), mode='nearest')[0]
                    # valid_bev2[b_idx, s_h: s_h + new_h, s_w: s_w + new_w] = torch.ones((new_h, new_w), dtype=torch.bool, device=f_bev_skeleton_mask_pred.device)
                # except:
                #     from IPython import embed;embed()
            #####

        # f_bev_skeleton_mask_pred_scaled = torch.clamp(f_bev_skeleton_mask_pred_scaled)
        # f_bev_skeleton_mask_pred = f_bev_skeleton_mask_pred.moveaxis(-1, 1)
        scores2 = self.exhaustive_voting(
            f_bev_skeleton_mask_pred, f_map_skeleton_mask_pred, valid_bev, pred_bev.get("confidence")
            # f_bev scaled down, skeleton_mask_pred, valid_bev, pred_bev.get("confidence")
        )
        scores2 = scores2.moveaxis(1, -1)  # B,map_H,map_W,n_rot
        scores = 0.5 * scores + scores2

        # use points with attention! 

        ## TODO or use linear to predict offset
        ## TODO iterative between location and scale 

        # TODO How to solve for scale? coverage mask label? 

        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            uvr_max = argmax_xyr_radius(scores).to(scores)
            uvr_avg, _ = expectation_xyr_radius(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "scale_f_map": scale_f_map,
            # "scale_f_bev": scale_f_bev,
            "f_map_skeleton_mask_pred":f_map_skeleton_mask_pred,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_bev": f_bev,
            "valid_bev": valid_bev.squeeze(1),
        }

    def loss(self, pred, data):
        # xy_gt = data["uv_gt"] - data["uv_offset"]
        xy_gt = data["uv_gt"]
        yaw_gt = data["theta_gt"]
        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed_radius(
                pred["log_probs"],
                xy_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr_radius(pred["log_probs"], xy_gt, yaw_gt)

        # data["map"].shape[1] * pred["scale_f_bev"] / pred["scale_f_map"]

        scale_loss = self.scale_mse_loss(data["map"].shape[1] / pred["scale_f_map"].flatten(), data["lidar_shape_len"].float())
        skeleton_pred_loss = self.skeleton_binary_ce_loss(
            pred["f_map_skeleton_mask_pred"].masked_fill(~data["lidar_coverage_mask_gt"].unsqueeze(1), 0.0), 
            data["lidar_point_mask_gt"].masked_fill(~data["lidar_coverage_mask_gt"].unsqueeze(1), 0.0).float()).mean((1,2,3), dtype=torch.float32)

        # from IPython import embed;embed()

        loss = {"total": nll + (10**-4) * scale_loss, "nll": nll, "scale_loss": scale_loss, "skeleton_pred_loss":skeleton_pred_loss}
        # loss = {"total": nll, "nll": nll, "skeleton_pred_loss":skeleton_pred_loss}
        # loss = {"total": nll, "nll": nll, "scale_loss": scale_loss}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter, "uv_gt"),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter, "uv_gt"
            ),
            "yaw_max_error": AngleError("yaw_max", "theta_gt"),
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "uv_max", "uv_gt"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max", "uv_gt"),
            "yaw_recall_2°": AngleRecall(2.0, "yaw_max", "theta_gt"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max", "theta_gt"),
        }


