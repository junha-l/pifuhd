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
    HGPIFu uses stacked hourglass as an image encoder.
    """

    def __init__(
        self, opt, netG, projection_mode="orthogonal", criteria={"occ": nn.MSELoss()}
    ):
        super(PIFuFine, self).__init__(
            projection_mode=projection_mode, criteria=criteria
        )

        self.name = "PIFuHD"

        in_ch = 3
        try:
            if netG.opt.use_front_normal:
                in_ch += 3
            if netG.opt.use_back_normal:
                in_ch += 3
        except:
            pass

        self.opt = opt
        self.image_filter = HGFilter(
            opt.num_stack, opt.hg_depth, in_ch, opt.hg_dim, opt.norm, "no_down", False
        )

        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim,
            merge_layer=-1,
            res_layers=self.opt.mlp_res_layers,
            norm=self.opt.mlp_norm,
            last_op=nn.Sigmoid(),
        )

        self.im_feat_list = []

        init_net(self)

        self.netG = netG

    def train(self, mode=True):
        r"""Sets the module in training mode."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.opt.train_full_pifu:
            self.netG.eval()
        return self

    def filter_global(self, images):
        """
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B1, C, H, W]
        """
        if self.opt.train_full_pifu:
            self.netG.filter(images)
        else:
            with torch.no_grad():
                self.netG.filter(images)

    def filter_local(self, images, rect=None):
        """
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B1, B2, C, H, W]
        """
        normals = []
        try:
            if self.netG.opt.use_front_normal:
                normals.append(self.netG.normal_front)
            if self.netG.opt.use_back_normal:
                normals.append(self.netG.normal_backward)
        except:
            pass

        if len(normals):
            normals = nn.Upsample(
                size=(self.opt.loadSizeBig, self.opt.loadSizeBig),
                mode="bilinear",
                align_corners=True,
            )(torch.cat(normals, 1))

            if rect is None:
                images = torch.cat(
                    [images, normals[:, None].expand(-1, images.size(1), -1, -1, -1)], 2
                )
            else:
                nml = []
                for i in range(rect.size(0)):
                    for j in range(rect.size(1)):
                        x1, y1, x2, y2 = rect[i, j]
                        tmp = normals[i, :, y1:y2, x1:x2]
                        nml.append(normals[i, :, y1:y2, x1:x2])
                nml = torch.stack(nml, 0).view(*rect.shape[:2], *nml[0].size())
                images = torch.cat([images, nml], 2)

        self.im_feat_list, self.normx = self.image_filter(
            images.view(-1, *images.size()[2:])
        )
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(
        self, points, calib_local, calib_global=None, transforms=None, labels=None
    ):
        """
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B1, B2, 3, N] 3d points in world space
            calibs_local: [B1, B2, 4, 4] calibration matrices for each image
            calibs_global: [B1, 4, 4] calibration matrices for each image
            transforms: [B1, 2, 3] image space coordinate transforms
            labels: [B1, B2, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        """
        if calib_global is not None:
            B = calib_local.size(1)
        else:
            B = 1
            points = points[:, None]
            calib_global = calib_local
            calib_local = calib_local[:, None]

        preds = []
        for i in range(B):
            xyz = self.projection(points[:, i], calib_local[:, i], transforms)
            xy = xyz[:, :2, :]

            # if the point is outside bounding box, return outside.
            in_bb = (xyz >= -1) & (xyz <= 1)
            in_bb = in_bb[:, 0, :] & in_bb[:, 1, :]
            in_bb = in_bb[:, None, :].detach().float()

            self.netG.query(points=points[:, i], calibs=calib_global)

            z_feat = self.netG.phi
            if not self.opt.train_full_pifu:
                z_feat = z_feat.detach()

            intermediate_preds_list = []
            for j, im_feat in enumerate(self.im_feat_list):
                point_local_feat_list = [
                    self.index(im_feat.view(-1, B, *im_feat.size()[1:])[:, i], xy),
                    z_feat,
                ]
                point_local_feat = torch.cat(point_local_feat_list, 1)
                pred = self.mlp(point_local_feat)[0]
                pred = in_bb * pred
                intermediate_preds_list.append(pred)

            preds.append(intermediate_preds_list[-1])

        self.preds = torch.cat(preds, 0)

    def forward(
        self,
        images_local,
        images_global,
        points,
        calib_local,
        calib_global,
        labels,
        rect=None,
    ):
        self.filter_global(images_global)
        self.filter_local(images_local, rect)
        self.query(points, calib_local, calib_global, labels=labels)
        res = self.get_preds()

        return None, res
