# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import HGFilter
from ..net_util import init_net
from ..networks import define_G


class PIFuCoarse(BasePIFuNet):
    """
    Coarse PIFu model with stacked hourglass as an image encoder.
    """

    def __init__(
        self, opt, projection_mode="orthogonal", criteria={"occ": nn.MSELoss()}
    ):
        super(PIFuCoarse, self).__init__(
            projection_mode=projection_mode, criteria=criteria
        )
        self.opt = opt

        ########## TODO: Define network ##########
        in_ch = 3 + 3 + 3
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
        self.netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
        ########## [End] Define network ##########
        self.spatial_enc = DepthNormalizer(opt)
        self.phi = None
        self.normal_front = None
        self.normal_backward = None
        init_net(self)

    def extract_feature(self, images):
        """
        Extract image features from given images
        """

        normals = []
        ########## TODO: Translate images to normal maps ##########

        ########## [End] Translate images to normal maps ##########
        normals = torch.cat(normals, 1)

        if images.size()[2:] != normals.size()[2:]:
            normals = nn.Upsample(
                size=images.size()[2:], mode="bilinear", align_corners=True
            )(normals)

        ########## TODO: Extract Image Features ##########

        ########## [End] Extract Image Features ##########

        if not self.training:
            self.im_feat = self.im_feat[-1]

    def query(self, points, calibs):
        """
        Predict occupancy fields given 3D sample points and camera parameters (calibs)
        """
        xyz = self.projection(points, calibs)
        xy = xyz[:, :2, :]

        # if the point is outside bounding box, return outside.
        in_bb = (xyz >= -1) & (xyz <= 1)
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :]
        in_bb = in_bb[:, None, :].detach().float()

        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        ########## TODO: Extract PIFu features ##########

        ########## [End] Extract PIFu features ##########

        pred = in_bb * pred

        self.phi = phi
        self.preds = pred
