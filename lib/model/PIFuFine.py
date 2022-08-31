# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .HGFilters import HGFilter
from ..net_util import init_net


class PIFuFine(BasePIFuNet):
    """
    Fine PIFu model with stacked hourglass as an image encoder.
    """

    def __init__(
        self, opt, netG, projection_mode="orthogonal", criteria={"occ": nn.MSELoss()}
    ):
        super(PIFuFine, self).__init__(
            projection_mode=projection_mode, criteria=criteria
        )
        self.opt = opt

        ########## TODO: Define network ##########

        ########## [End] Define network ##########
        init_net(self)
        self.netG = netG

    def extract_feature_global(self, images):
        """
        Extract coarse image feature from given images
        """
        with torch.no_grad():
            self.netG.extract_feature(images)

    def extract_feature_local(self, images, rect=None):
        """
        Extract fine image feature from given images and coarse image features
        """
        normals = []
        normals.append(self.netG.normal_front)
        normals.append(self.netG.normal_backward)

        normals = nn.Upsample(
            size=(self.opt.loadSizeBig, self.opt.loadSizeBig),
            mode="bilinear",
            align_corners=True,
        )(torch.cat(normals, 1))

        ########## TODO: Extract image features ##########

        ########## [End] Extract image feature ##########
        if not self.training:
            self.im_feat = self.im_feat[-1]

    def query(self, points, calib_local):
        """
        Predict occupancy fields given 3D sample points and camera parameters (calibs)
        """
        calib_global = calib_local

        xyz = self.projection(points, calib_local)
        xy = xyz[:, :2, :]

        # if the point is outside bounding box, return outside.
        in_bb = (xyz >= -1) & (xyz <= 1)
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :]
        in_bb = in_bb[:, None, :].detach().float()

        self.netG.query(points=points, calibs=calib_global)

        z_feat = self.netG.phi
        if not self.opt.train_full_pifu:
            z_feat = z_feat.detach()

        ########## TODO: Extract PIFu features ##########

        ########## [End] Extract PIFu features ##########
        pred = in_bb * pred

        self.preds = pred
