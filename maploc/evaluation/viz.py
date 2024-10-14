# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json

from ..utils.io import write_torch_image
from ..utils.viz_2d import plot_images, features_to_RGB, save_plot, save_fig
from ..utils.viz_localization import (
    likelihood_overlay,
    plot_pose,
    plot_dense_rotations,
    add_circle_inset,
)
from ..osm.viz import Colormap, plot_nodes

from maploc.data.carla.utils import( 
    load_point_cloud, 
    overlay_points_on_image,
    overlay_arrow_on_image,
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
from PIL import Image

# def plot_points_on_image():

def plot_example_single_rgb(
    idx,
    model,
    pred,
    data,
    root = "/home/rayguan/scratch_ssd/projects/carla/transfuser/results/",
    out_dir=None,
):
    # scene, name, rasters, uv_gt, uv_offset = (data[k] for k in ("route_dir", "aerial_idx", "ground_idx", "uv_gt", "uv_offset"))

    route_dir = data["route_dir"]
    aerial_idx = data["aerial_idx"]
    ground_idx = data["ground_idx"]

    map = data["map"]

    uv_offset = data["uv_offset"]
    pts_in_uv = data["pts_in_uv"]

    uv_init = data["uv_init"]
    theta_init = data["theta_init"]

    uv_gt = data["uv_gt"]
    theta_gt = data["theta_gt"]

    

    sub_folder = route_dir


    # Main execution
    point_cloud_path = os.path.join(root, sub_folder, "lidar/" + ground_idx + ".npy")
    ego_measurement_path = os.path.join(root, sub_folder, "measurements/" + ground_idx + ".json")
    ego_measurement = json.load(open(ego_measurement_path))


    satellite_image_path = os.path.join(root, sub_folder, "topdown_rgb/" + aerial_idx + ".png")
    cam_cur_measurement_path = os.path.join(root, sub_folder, "measurements/" + aerial_idx + ".json")
    cam_cur_measurement = json.load(open(cam_cur_measurement_path))

    ego_to_lidar = get_transformation_from_ego(*lidar_param)

    ego_to_cur1 = get_rot_matrix(ego_measurement["theta"])
    ego_to_cur2 = get_trans_matrix(ego_measurement["y"], -1 * ego_measurement["x"])
    ego_to_cur3 = get_trans_matrix(-1 * cam_cur_measurement["y"], cam_cur_measurement["x"])
    ego_to_cur4 = get_rot_matrix(-1 * cam_cur_measurement["theta"])
    ego_to_cur = np.matmul(ego_to_cur4, 
                            np.matmul(ego_to_cur3,
                                        np.matmul(ego_to_cur2,
                                                ego_to_cur1)))
    
    cur_to_cam = get_transformation_from_ego(*cam_extrinsic_param)
    cam_intrinsic = get_cam_intrinsic(*cam_intrinsic_param)


    point_cloud = load_point_cloud(point_cloud_path)
    # https://github.com/carla-simulator/carla/issues/392
    # carla_trans = np.eye(4)
    # Transform(Rotation(yaw=90), Scale(z=-1)) 
    carla_trans = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Transform(Rotation(pitch=180)) 
    to_normal_pc_trans = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # carla_rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # carla_scale = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # carla_trans = np.matmul(carla_scale, carla_rotation)

    # from IPython import embed;embed()
    # o3d.visualization.draw_geometries([point_cloud])
    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window()
    # viewer.add_geometry(point_cloud)
    # opt = viewer.get_render_option()
    # opt.show_coordinate_frame = True
    # viewer.run()
    # viewer.destroy_window()

    point_cloud.transform(carla_trans)
    point_cloud.transform(to_normal_pc_trans)

    point_cloud.transform(np.linalg.inv(to_normal_pc_trans))
    point_cloud.transform(np.linalg.inv(ego_to_lidar))
    point_cloud.transform(ego_to_cur)
    point_cloud.transform(cur_to_cam)

    final_transformation = np.matmul(cur_to_cam, np.matmul(ego_to_cur, np.matmul(np.linalg.inv(ego_to_lidar), np.linalg.inv(to_normal_pc_trans))))
    # point_cloud.transform(final_transformation)

    # from IPython import embed;embed()
    # point_cloud = np.asarray(point_cloud.points)
    # lst = [int(a) for a in  point_cloud.T[2]]
    # np.unique( lst, return_counts=True)

    point_cloud = np.asarray(point_cloud.points)

    points_2d_homogeneous = cam_intrinsic @ point_cloud.T

    points_2d =  points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

    overlay_image = overlay_points_on_image(np.array(map), (points_2d.T - np.array(uv_offset)).astype(np.int_))
    overlay_image = overlay_points_on_image(overlay_image, [uv_gt.int()], (0, 255, 0), r=10)

    arrow_len = 50
    # from IPython import embed;embed()
    x_front = uv_gt[0] + arrow_len * np.cos(theta_gt.numpy())
    y_front = uv_gt[1] + arrow_len * np.sin(theta_gt.numpy())

    # theta_gt = math.atan2(
    #     uv_gt_front[1] - uv_gt[1],
    #     uv_gt_front[0] - uv_gt[0]
    # )

    out_arrow = torch.tensor([x_front, y_front])

    overlay_image = overlay_arrow_on_image(overlay_image, uv_gt.int(), out_arrow.int(), (0, 255, 0), r=2)
    

    uv_p, yaw_p = pred["uv_max"], pred.get("yaw_max")

    overlay_image = overlay_points_on_image(overlay_image, [uv_p.int()], (0, 0, 255), r=10)

    arrow_len = 50
    # from IPython import embed;embed()
    x_front = uv_p[0] + arrow_len * np.cos(yaw_p.numpy())
    y_front = uv_p[1] + arrow_len * np.sin(yaw_p.numpy())

    # theta_gt = math.atan2(
    #     uv_gt_front[1] - uv_gt[1],
    #     uv_gt_front[0] - uv_gt[0]
    # )

    out_arrow = torch.tensor([x_front, y_front])

    overlay_image = overlay_arrow_on_image(overlay_image, uv_p.int(), out_arrow.int(), (0, 0, 255), r=2)
        
    plt.imshow(overlay_image)
    plt.show()

    if out_dir is not None:
        p = str(out_dir / f"{route_dir}_{aerial_idx}_{ground_idx}_{{}}.pdf")
        save_plot(p.format("bev"))
        plt.close()


def plot_example_single_rgb_paper_fig(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=False,
    out_dir=None,
    fig_for_paper=False,
    show_dir_error=False,
    vis=True,
    draw_scale=1,
    vis_mask=False
):
    route_dir, aerial_idx, ground_idx, rasters, uv_gt = (data[k] for k in ("route_dir", "aerial_idx", "ground_idx", "map", "uv_gt"))

    yaw_gt = data["theta_gt"].numpy()

    lp_uvt = lp_uv = pred["log_probs"]

    has_rotation = lp_uvt.ndim == 3
    if has_rotation:
        lp_uv = lp_uvt.max(-1).values
    if lp_uv.min() > -np.inf:
        lp_uv = lp_uv.clip(min=np.percentile(lp_uv, 1))
    prob = lp_uv.exp()
    uv_p, yaw_p = pred["uv_max"], pred.get("yaw_max")

    feats_map = pred["map"]["map_features"][0]
    (feats_map_rgb,) = features_to_RGB(feats_map.numpy())

    if "f_map_skeleton_mask_pred" in pred and vis_mask:
        f_map_skeleton_mask_pred = pred["f_map_skeleton_mask_pred"]
        # from IPython import embed;embed()
        skeleton_map_rgb = f_map_skeleton_mask_pred[0, ...].unsqueeze(-1).repeat(1,1,3).numpy()

    text1 = rf'$\Delta xy$: {results["xy_max_error"]:.1f}m'
    if has_rotation:
        text1 += rf', $\Delta\theta$: {results["yaw_max_error"]:.1f}°'

    if show_dir_error and "directional_error" in results:
        err_lat, err_lon = results["directional_error"]
        text1 += rf",  $\Delta$lateral/longitundinal={err_lat:.1f}m/{err_lon:.1f}m"


    map_viz = np.array(rasters).copy()/255
    overlay = likelihood_overlay(prob.numpy(), map_viz.mean(-1, keepdims=True))
    lidar_viz = overlay_points_on_image(np.ones(rasters.size())*255, (np.array(data["pts_in_uv"])).astype(np.int_), r=(int(2*draw_scale)))
    mask_viz = overlay_points_on_image(np.ones(rasters.size())*255, (np.array(data["pts_in_uv"])).astype(np.int_), color=((0, 0, 0)), r=(int(2*draw_scale)))
    gt_viz = overlay_points_on_image(np.array(rasters).copy(), (np.array(data["pts_in_uv"])).astype(np.int_), r=(int(2*draw_scale)))
    
    
    data["ori_uv_pts"][:,0] = data["ori_uv_pts"][:,0] - min(data["ori_uv_pts"][:,0])
    data["ori_uv_pts"][:,1] = data["ori_uv_pts"][:,1] - min(data["ori_uv_pts"][:,1])
    
    ori_scale_viz = overlay_points_on_image(np.ones((max(data["ori_uv_pts"][:,0]), max(data["ori_uv_pts"][:,1]), 3))*255, (np.array(data["ori_uv_pts"])).astype(np.int_), 
                                            r=(int(2*draw_scale * max(max(data["ori_uv_pts"][:,0]), max(data["ori_uv_pts"][:,1])) / max(rasters.size()[0], rasters.size()[1]))))


    # pred_viz = overlay_points_on_image(np.array(rasters).copy(), (np.array(data["pts_in_uv"])).astype(np.int_), r=(int(2*draw_scale)))
    # pred_viz = overlay_points_on_image(pred_viz, [uv_gt.int()], (0, 255, 0), r=int(10*draw_scale))
    pred_viz = overlay_points_on_image(np.array(rasters).copy(), [uv_gt.int()], (0, 255, 0), r=int(10*draw_scale))

    arrow_len = int(50 * draw_scale)
    # from IPython import embed;embed()
    x_front = uv_gt[0] + arrow_len * np.cos(yaw_gt)
    y_front = uv_gt[1] + arrow_len * np.sin(yaw_gt)

    out_arrow = torch.tensor([x_front, y_front])

    pred_viz = overlay_arrow_on_image(pred_viz, uv_gt.int(), out_arrow.int(), (0, 255, 0), r=int(2*draw_scale))
    pred_viz = overlay_points_on_image(pred_viz, [uv_p.int()], (0, 0, 255), r=int(10*draw_scale))

    # from IPython import embed;embed()
    x_front = uv_p[0] + arrow_len * np.cos(yaw_p.numpy())
    y_front = uv_p[1] + arrow_len * np.sin(yaw_p.numpy())

    out_arrow = torch.tensor([x_front, y_front])

    pred_viz = overlay_arrow_on_image(pred_viz, uv_p.int(), out_arrow.int(), (0, 0, 255), r=int(2*draw_scale))

    # overlay = likelihood_overlay(prob.numpy(), pred_viz/255)
    if "f_map_skeleton_mask_pred" in pred and vis_mask:
        plot_images(
            [pred_viz, overlay, feats_map_rgb, skeleton_map_rgb],
            titles=[text1, "Likelihood", "Neural Map", "Skeleton Map"],
            dpi=100,
            cmaps="jet",
        )
    else:
        plot_images(
            [pred_viz, overlay, feats_map_rgb],
            titles=[text1, "Likelihood", "Neural Map"],
            dpi=100,
            cmaps="jet",
        )
    # plot_images(
    #     [pred_viz, map_viz, overlay, feats_map_rgb, skeleton_map_rgb],
    #     titles=[text1, "map", "likelihood", "neural map", "pred_skeleton map"],
    #     dpi=75,
    #     cmaps="jet",
    # )
    fig = plt.gcf()
    axes = fig.axes
    axes[1].images[0].set_interpolation("none")
    axes[2].images[0].set_interpolation("none")
    # Colormap.add_colorbar()
    # plot_nodes(1, rasters[2])

    # plot_pose([1], uv_gt, yaw_gt, c="g")
    # plot_pose([1], uv_p, yaw_p, c="b")
    # plot_dense_rotations(2, lp_uvt.exp())
    inset_center = pred["uv_max"] if results["xy_max_error"] < 5 else uv_gt
    axins = add_circle_inset(axes[1], inset_center)
    # axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50, zorder=15)
    axes[0].text(
        0.003,
        0.003,
        f"{route_dir}/{aerial_idx}/{ground_idx}",
        transform=axes[0].transAxes,
        fontsize=3,
        va="bottom",
        ha="left",
        color="w",
    )
    fig_save = plt.gcf()
    if vis:
        plt.show()
    if out_dir is not None:
        name_ = route_dir.replace("/", "_")
        p = str(out_dir / f"{name_}_{aerial_idx}_{ground_idx}_{{}}.{{}}")
        save_fig(p.format("pred", "pdf"), fig_save)
        save_fig(p.format("pred", "png"), fig_save)
        im_map = Image.fromarray((map_viz* 255).astype(np.uint8))
        im_map.save(p.format("map", "png"))
        im_lidar = Image.fromarray((lidar_viz).astype(np.uint8))
        im_lidar.save(p.format("lidar", "png"))
        im_mask = Image.fromarray((mask_viz).astype(np.uint8))
        im_mask.save(p.format("mask", "png"))
        im_sk_ori = Image.fromarray((ori_scale_viz).astype(np.uint8))
        im_sk_ori.save(p.format("lidar_ori", "png"))
        im_gt = Image.fromarray((gt_viz).astype(np.uint8))
        im_gt.save(p.format("gt", "png"))
        im_feats = Image.fromarray((feats_map_rgb* 255).astype(np.uint8))
        im_feats.save(p.format("feature", "png"))
        # save_fig(p.format("map", "png"), map_viz)
        # save_fig(p.format("lidar", "png"), sk_viz)
        # save_fig(p.format("gt", "png"), gt_viz)
        # save_fig(p.format("feature", "png"), feats_map_rgb)
        plt.close()

def plot_example_single_rgb_more(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=False,
    out_dir=None,
    fig_for_paper=False,
    show_dir_error=False,
    vis=True,
    draw_scale=1,
    vis_mask=True
):
    route_dir, aerial_idx, ground_idx, rasters, uv_gt = (data[k] for k in ("route_dir", "aerial_idx", "ground_idx", "map", "uv_gt"))

    yaw_gt = data["theta_gt"].numpy()

    lp_uvt = lp_uv = pred["log_probs"]

    has_rotation = lp_uvt.ndim == 3
    if has_rotation:
        lp_uv = lp_uvt.max(-1).values
    if lp_uv.min() > -np.inf:
        lp_uv = lp_uv.clip(min=np.percentile(lp_uv, 1))
    prob = lp_uv.exp()
    uv_p, yaw_p = pred["uv_max"], pred.get("yaw_max")

    feats_map = pred["map"]["map_features"][0]
    (feats_map_rgb,) = features_to_RGB(feats_map.numpy())

    if "f_map_skeleton_mask_pred" in pred and vis_mask:
        f_map_skeleton_mask_pred = pred["f_map_skeleton_mask_pred"]
        # from IPython import embed;embed()
        skeleton_map_rgb = f_map_skeleton_mask_pred[0, ...].unsqueeze(-1).repeat(1,1,3).numpy()

    text1 = rf'$\Delta xy$: {results["xy_max_error"]:.1f}m'
    if has_rotation:
        text1 += rf', $\Delta\theta$: {results["yaw_max_error"]:.1f}°'

    if show_dir_error and "directional_error" in results:
        err_lat, err_lon = results["directional_error"]
        text1 += rf",  $\Delta$lateral/longitundinal={err_lat:.1f}m/{err_lon:.1f}m"


    map_viz = np.array(rasters).copy()/255
    overlay = likelihood_overlay(prob.numpy(), map_viz.mean(-1, keepdims=True))

    pred_viz = overlay_points_on_image(np.array(rasters).copy(), (np.array(data["pts_in_uv"])).astype(np.int_), r=(int(2*draw_scale)))
    pred_viz = overlay_points_on_image(pred_viz, [uv_gt.int()], (0, 255, 0), r=int(10*draw_scale))

    arrow_len = int(50 * draw_scale)
    # from IPython import embed;embed()
    x_front = uv_gt[0] + arrow_len * np.cos(yaw_gt)
    y_front = uv_gt[1] + arrow_len * np.sin(yaw_gt)

    out_arrow = torch.tensor([x_front, y_front])

    pred_viz = overlay_arrow_on_image(pred_viz, uv_gt.int(), out_arrow.int(), (0, 255, 0), r=int(2*draw_scale))
    pred_viz = overlay_points_on_image(pred_viz, [uv_p.int()], (0, 0, 255), r=int(10*draw_scale))

    # from IPython import embed;embed()
    x_front = uv_p[0] + arrow_len * np.cos(yaw_p.numpy())
    y_front = uv_p[1] + arrow_len * np.sin(yaw_p.numpy())

    out_arrow = torch.tensor([x_front, y_front])

    pred_viz = overlay_arrow_on_image(pred_viz, uv_p.int(), out_arrow.int(), (0, 0, 255), r=int(2*draw_scale))

    # overlay = likelihood_overlay(prob.numpy(), pred_viz/255)
    if "f_map_skeleton_mask_pred" in pred and vis_mask:
        plot_images(
            [pred_viz, overlay, feats_map_rgb, skeleton_map_rgb],
            titles=[text1, "likelihood", "neural map", "pred_skeleton map"],
            dpi=75,
            cmaps="jet",
        )
    else:
        plot_images(
            [pred_viz, overlay, feats_map_rgb],
            titles=[text1, "likelihood", "neural map"],
            dpi=75,
            cmaps="jet",
        )
    # plot_images(
    #     [pred_viz, map_viz, overlay, feats_map_rgb, skeleton_map_rgb],
    #     titles=[text1, "map", "likelihood", "neural map", "pred_skeleton map"],
    #     dpi=75,
    #     cmaps="jet",
    # )
    fig = plt.gcf()
    axes = fig.axes
    axes[1].images[0].set_interpolation("none")
    axes[2].images[0].set_interpolation("none")
    # Colormap.add_colorbar()
    # plot_nodes(1, rasters[2])

    # plot_pose([1], uv_gt, yaw_gt, c="g")
    # plot_pose([1], uv_p, yaw_p, c="b")
    # plot_dense_rotations(2, lp_uvt.exp())
    inset_center = pred["uv_max"] if results["xy_max_error"] < 5 else uv_gt
    axins = add_circle_inset(axes[1], inset_center)
    # axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50, zorder=15)
    axes[0].text(
        0.003,
        0.003,
        f"{route_dir}/{aerial_idx}/{ground_idx}",
        transform=axes[0].transAxes,
        fontsize=3,
        va="bottom",
        ha="left",
        color="w",
    )
    fig_save = plt.gcf()
    if vis:
        plt.show()
    if out_dir is not None:
        name_ = route_dir.replace("/", "_")
        p = str(out_dir / f"{name_}_{aerial_idx}_{ground_idx}_{{}}.pdf")
        save_fig(p.format("pred"), fig_save)
        plt.close()

        if fig_for_paper:
            # !cp ../datasets/MGL/{scene}/images/{name}.jpg {out_dir}/{scene}_{name}.jpg
            plot_images([map_viz])
            plt.gca().images[0].set_interpolation("none")
            plot_nodes(0, rasters[2])
            plot_pose([0], uv_gt, yaw_gt, c="red")
            plot_pose([0], pred["uv_max"], pred["yaw_max"], c="k")
            save_plot(p.format("map"))
            plt.close()
            plot_images([lp_uv], cmaps="jet")
            plot_dense_rotations(0, lp_uvt.exp())
            save_plot(p.format("loglikelihood"), dpi=100)
            plt.close()
            plot_images([overlay])
            plt.gca().images[0].set_interpolation("none")
            axins = add_circle_inset(plt.gca(), inset_center)
            axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50)
            save_plot(p.format("likelihood"))
            plt.close()
            write_torch_image(
                p.format("neuralmap").replace("pdf", "jpg"), feats_map_rgb
            )

    if not plot_bev:
        return

    feats_q = pred["features_bev"]
    mask_bev = pred["valid_bev"]
    prior = None
    if "log_prior" in pred["map"]:
        prior = pred["map"]["log_prior"][0].sigmoid()
    if "bev" in pred and "confidence" in pred["bev"]:
        conf_q = pred["bev"]["confidence"]
    else:
        conf_q = torch.norm(feats_q, dim=0)
    conf_q = conf_q.masked_fill(~mask_bev, np.nan)
    (feats_q_rgb,) = features_to_RGB(feats_q.numpy(), masks=[mask_bev.numpy()])
    # feats_map_rgb, feats_q_rgb, = features_to_RGB(
    #     feats_map.numpy(), feats_q.numpy(), masks=[None, mask_bev])
    norm_map = torch.norm(feats_map, dim=0)

    plot_images(
        [conf_q, feats_q_rgb, norm_map] + ([] if prior is None else [prior]),
        titles=["BEV confidence", "BEV features", "map norm"]
        + ([] if prior is None else ["map prior"]),
        dpi=50,
        cmaps="jet",
    )
    if vis:
        plt.show()

    # if out_dir is not None:
    #     save_plot(p.format("bev"))
    #     plt.close()


def plot_example_single(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    scene, name, rasters, uv_gt = (data[k] for k in ("scene", "name", "map", "uv"))
    uv_gps = data.get("uv_gps")
    yaw_gt = data["roll_pitch_yaw"][-1].numpy()
    image = data["image"].permute(1, 2, 0)
    if "valid" in data:
        image = image.masked_fill(~data["valid"].unsqueeze(-1), 0.3)

    lp_uvt = lp_uv = pred["log_probs"]
    if show_fused and "log_probs_fused" in pred:
        lp_uvt = lp_uv = pred["log_probs_fused"]
    elif not show_masked_prob and "scores_unmasked" in pred:
        lp_uvt = lp_uv = pred["scores_unmasked"]
    has_rotation = lp_uvt.ndim == 3
    if has_rotation:
        lp_uv = lp_uvt.max(-1).values
    if lp_uv.min() > -np.inf:
        lp_uv = lp_uv.clip(min=np.percentile(lp_uv, 1))
    prob = lp_uv.exp()
    uv_p, yaw_p = pred["uv_max"], pred.get("yaw_max")
    if show_fused and "uv_fused" in pred:
        uv_p, yaw_p = pred["uv_fused"], pred.get("yaw_fused")
    feats_map = pred["map"]["map_features"][0]
    (feats_map_rgb,) = features_to_RGB(feats_map.numpy())

    text1 = rf'$\Delta xy$: {results["xy_max_error"]:.1f}m'
    if has_rotation:
        text1 += rf', $\Delta\theta$: {results["yaw_max_error"]:.1f}°'
    if show_fused and "xy_fused_error" in results:
        text1 += rf', $\Delta xy_{{fused}}$: {results["xy_fused_error"]:.1f}m'
        text1 += rf', $\Delta\theta_{{fused}}$: {results["yaw_fused_error"]:.1f}°'
    if show_dir_error and "directional_error" in results:
        err_lat, err_lon = results["directional_error"]
        text1 += rf",  $\Delta$lateral/longitundinal={err_lat:.1f}m/{err_lon:.1f}m"
    if "xy_gps_error" in results:
        text1 += rf',  $\Delta xy_{{GPS}}$: {results["xy_gps_error"]:.1f}m'

    map_viz = Colormap.apply(rasters)
    overlay = likelihood_overlay(prob.numpy(), map_viz.mean(-1, keepdims=True))
    plot_images(
        [image, map_viz, overlay, feats_map_rgb],
        titles=[text1, "map", "likelihood", "neural map"],
        dpi=75,
        cmaps="jet",
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[1].images[0].set_interpolation("none")
    axes[2].images[0].set_interpolation("none")
    Colormap.add_colorbar()
    plot_nodes(1, rasters[2])

    if show_gps and uv_gps is not None:
        plot_pose([1], uv_gps, c="blue")
    plot_pose([1], uv_gt, yaw_gt, c="red")
    plot_pose([1], uv_p, yaw_p, c="k")
    plot_dense_rotations(2, lp_uvt.exp())
    inset_center = pred["uv_max"] if results["xy_max_error"] < 5 else uv_gt
    axins = add_circle_inset(axes[2], inset_center)
    axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50, zorder=15)
    axes[0].text(
        0.003,
        0.003,
        f"{scene}/{name}",
        transform=axes[0].transAxes,
        fontsize=3,
        va="bottom",
        ha="left",
        color="w",
    )
    plt.show()
    if out_dir is not None:
        name_ = name.replace("/", "_")
        p = str(out_dir / f"{scene}_{name_}_{{}}.pdf")
        save_plot(p.format("pred"))
        plt.close()

        if fig_for_paper:
            # !cp ../datasets/MGL/{scene}/images/{name}.jpg {out_dir}/{scene}_{name}.jpg
            plot_images([map_viz])
            plt.gca().images[0].set_interpolation("none")
            plot_nodes(0, rasters[2])
            plot_pose([0], uv_gt, yaw_gt, c="red")
            plot_pose([0], pred["uv_max"], pred["yaw_max"], c="k")
            save_plot(p.format("map"))
            plt.close()
            plot_images([lp_uv], cmaps="jet")
            plot_dense_rotations(0, lp_uvt.exp())
            save_plot(p.format("loglikelihood"), dpi=100)
            plt.close()
            plot_images([overlay])
            plt.gca().images[0].set_interpolation("none")
            axins = add_circle_inset(plt.gca(), inset_center)
            axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50)
            save_plot(p.format("likelihood"))
            plt.close()
            write_torch_image(
                p.format("neuralmap").replace("pdf", "jpg"), feats_map_rgb
            )
            write_torch_image(p.format("image").replace("pdf", "jpg"), image.numpy())

    if not plot_bev:
        return

    feats_q = pred["features_bev"]
    mask_bev = pred["valid_bev"]
    prior = None
    if "log_prior" in pred["map"]:
        prior = pred["map"]["log_prior"][0].sigmoid()
    if "bev" in pred and "confidence" in pred["bev"]:
        conf_q = pred["bev"]["confidence"]
    else:
        conf_q = torch.norm(feats_q, dim=0)
    conf_q = conf_q.masked_fill(~mask_bev, np.nan)
    (feats_q_rgb,) = features_to_RGB(feats_q.numpy(), masks=[mask_bev.numpy()])
    # feats_map_rgb, feats_q_rgb, = features_to_RGB(
    #     feats_map.numpy(), feats_q.numpy(), masks=[None, mask_bev])
    norm_map = torch.norm(feats_map, dim=0)

    plot_images(
        [conf_q, feats_q_rgb, norm_map] + ([] if prior is None else [prior]),
        titles=["BEV confidence", "BEV features", "map norm"]
        + ([] if prior is None else ["map prior"]),
        dpi=50,
        cmaps="jet",
    )
    plt.show()

    if out_dir is not None:
        save_plot(p.format("bev"))
        plt.close()


def plot_example_sequential_rgb(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    return



def plot_example_sequential(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    return
