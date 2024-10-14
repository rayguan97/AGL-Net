# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat
import numpy as np
from .utils import deg2rad, rotmat2d


def location_error(uv, uv_gt, ppm=1):
    return torch.norm(uv - uv_gt.to(uv), dim=-1) / ppm


def angle_error(t, t_gt, radius:bool = False):
    if radius:
        t = t * 180 / np.pi
        t_gt = t_gt * 180 / np.pi
    error = torch.abs(t % 360 - t_gt.to(t) % 360)
    error = torch.minimum(error, 360 - error)
    return error


class Location2DRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, pixel_per_meter, key="uv_max", data_key="uv", *args, **kwargs):
        self.threshold = threshold
        self.ppm = pixel_per_meter
        self.key = key
        self.data_key = data_key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        if "adjust_pixels_per_meter" in data:
            error = location_error(pred[self.key], data[self.data_key], data["adjust_pixels_per_meter"])
        else:
            error = location_error(pred[self.key], data[self.data_key], self.ppm)
        super().update((error <= self.threshold).float())


class AngleRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="yaw_max", data_key="roll_pitch_yaw", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        self.data_key = data_key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        if self.data_key == "roll_pitch_yaw":
            error = angle_error(pred[self.key], data[self.data_key][..., -1])
        else:
            error = angle_error(pred[self.key], data[self.data_key], True)
        super().update((error <= self.threshold).float())


class MeanMetricWithRecall(torchmetrics.Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("value", default=[], dist_reduce_fx="cat")

    def compute(self):
        return dim_zero_cat(self.value).mean(0)

    def get_errors(self):
        return dim_zero_cat(self.value)

    def recall(self, thresholds):
        error = self.get_errors()
        thresholds = error.new_tensor(thresholds)
        return (error.unsqueeze(-1) < thresholds).float().mean(0) * 100


class AngleError(MeanMetricWithRecall):
    def __init__(self, key, data_key="roll_pitch_yaw"):
        super().__init__()
        self.key = key
        self.data_key = data_key

    def update(self, pred, data):
        if self.data_key == "roll_pitch_yaw":
            value = angle_error(pred[self.key], data[self.data_key][..., -1])
        else:
            value = angle_error(pred[self.key], data[self.data_key], True)
    
        if value.numel():
            self.value.append(value)


class Location2DError(MeanMetricWithRecall):
    def __init__(self, key, pixel_per_meter, data_key="uv"):
        super().__init__()
        self.key = key
        self.ppm = pixel_per_meter
        self.data_key = data_key

    def update(self, pred, data):

        if "adjust_pixels_per_meter" in data:
            value = location_error(pred[self.key], data[self.data_key], data["adjust_pixels_per_meter"])
        else:
            value = location_error(pred[self.key], data[self.data_key], self.ppm)

        if value.numel():
            self.value.append(value)


class LateralLongitudinalError(MeanMetricWithRecall):
    def __init__(self, pixel_per_meter, key="uv_max", uv_key="uv", theta_key="roll_pitch_yaw"):
        super().__init__()
        self.ppm = pixel_per_meter
        self.key = key
        self.uv_key = uv_key
        self.theta_key = theta_key

    def update(self, pred, data):

        if self.theta_key == "roll_pitch_yaw":
            yaw = deg2rad(data["roll_pitch_yaw"][..., -1])
        else:
            yaw = data[self.theta_key]

        shift = (pred[self.key] - data[self.uv_key]) * yaw.new_tensor([-1, 1])
        shift = (rotmat2d(yaw) @ shift.unsqueeze(-1)).squeeze(-1)

        if "adjust_pixels_per_meter" in data:
            error = torch.abs(shift) / data["adjust_pixels_per_meter"]
        else:
            error = torch.abs(shift) / self.ppm

        value = error.view(-1, 2)
        if value.numel():
            self.value.append(value)
