# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import HGFilter
from ..net_util import init_net
from ..networks import define_G


class PIFuCoarse(BasePIFuNet):
    """
    Coarse PIFu model uses stacked hourglass as an image encoder.
    """

    def __init__(
        self, opt, projection_mode="orthogonal", criteria={"occ": nn.MSELoss()}
    ):
        super(PIFuCoarse, self).__init__(
            projection_mode=projection_mode, criteria=criteria
        )

        self.name = "PIFuHD"

        in_ch = 3
        try:
            if opt.use_front_normal:
                in_ch += 3
            if opt.use_back_normal:
                in_ch += 3
        except:
            pass
        self.opt = opt
        self.image_filter = HGFilter(
            opt.num_stack, opt.hg_depth, in_ch, opt.hg_dim, opt.norm, opt.hg_down, False
        )

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            merge_layer=self.opt.merge_layer,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid(),
        )

        self.spatial_enc = DepthNormalizer(opt)

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.phi = None

        self.intermediate_preds_list = []

        init_net(self)

        self.netF = None
        self.netB = None
        try:
            if opt.use_front_normal:
                self.netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
            if opt.use_back_normal:
                self.netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
        except:
            pass
        self.normal_front = None
        self.normal_backward = None

    def filter(self, images):
        """
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        """
        normals = []
        # if you wish to train jointly, remove detach etc.
        with torch.no_grad():
            if self.netF is not None:
                self.normal_front = self.netF.forward(images).detach()
                normals.append(self.normal_front)
            if self.netB is not None:
                self.normal_backward = self.netB.forward(images).detach()
                normals.append(self.normal_backward)
        if len(normals) != 0:
            normals = torch.cat(normals, 1)
            if images.size()[2:] != normals.size()[2:]:
                normals = nn.Upsample(
                    size=images.size()[2:], mode="bilinear", align_corners=True
                )(normals)
            images = torch.cat([images, normals], 1)

        self.im_feat_list, self.normx = self.image_filter(images)

        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(
        self,
        points,
        calibs,
        transforms=None,
        labels=None,
        update_pred=True,
        update_phi=True,
    ):
        """
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            labels: [B, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        """
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]

        # if the point is outside bounding box, return outside.
        in_bb = (xyz >= -1) & (xyz <= 1)
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :]
        in_bb = in_bb[:, None, :].detach().float()

        if labels is not None:
            self.labels = in_bb * labels

        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        intermediate_preds_list = []

        phi = None
        for i, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [self.index(im_feat, xy), sp_feat]
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred, phi = self.mlp(point_local_feat)
            pred = in_bb * pred

            intermediate_preds_list.append(pred)

        if update_phi:
            self.phi = phi

        if update_pred:
            self.intermediate_preds_list = intermediate_preds_list
            self.preds = self.intermediate_preds_list[-1]

    def forward(self, images, points, calibs, labels):
        self.filter(images)
        self.query(points, calibs, labels=labels)
        res = self.get_preds()

        return None, res
