import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import FeaturePyramidNetwork

from collections import defaultdict
from loguru import logger

from .convlstm import LSTMFusion
from .decoder import Decoder, Decoder2
from .losses import LossMeter, update_losses


class DepthRNN(nn.Module):
    def __init__(self, cfg):
        super(DepthRNN, self).__init__()
        self.cfg = cfg.MODEL

        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        self.load_img = cfg.DATA.LOAD_IMG
        if self.load_img:
            logger.info('Train with depth and image')
            self.feature_extractor = FeatureExtractor(alpha, 4)
        else:
            logger.info('Train with depth')
            self.feature_extractor = FeatureExtractor(alpha, 1)
        self.rnn = LSTMFusion()
        self.decoder = Decoder()

        # logger.info('no. of parameters in feature extractor: {}'.format(
        #     sum(p.numel() for p in self.feature_extractor.parameters())))
        # logger.info('no. of parameters in feature rnn: {}'.format(
        #     sum(p.numel() for p in self.rnn.parameters())))
        # logger.info('no. of parameters in feature decoder: {}'.format(
        #     sum(p.numel() for p in self.decoder.parameters())))

    def forward(self, inputs, is_training=True):
        outputs = defaultdict(list)

        # move batch dimension to 1
        if self.load_img:
            imgs = torch.permute(inputs['imgs'], (1, 0, 2, 3, 4))
        pred_depths = torch.permute(inputs['pred_depths'], (1, 0, 2, 3)).unsqueeze(2)
        gt_depths = torch.permute(inputs['gt_depths'], (1, 0, 2, 3)).unsqueeze(2)
        intrinsics = torch.permute(inputs['intrinsics'], (1, 0, 2, 3))
        poses = torch.permute(inputs['poses'], (1, 0, 2, 3))

        feature_halves, feature_quarters, pred_depths_quarters = [], [], []
        # feature_halves, feature_quarters, feature_one_eights, pred_depths_one_eights = [], [], [], []

        optimizer_loss = 0
        l1_meter = LossMeter()
        huber_meter = LossMeter()
        l1_inv_meter = LossMeter()
        l1_rel_meter = LossMeter()

        for i, depth in enumerate(pred_depths):
            if self.load_img:
                feature_half, feature_quarter = self.feature_extractor(torch.cat((imgs[i], depth), dim=1))
                # feature_half, feature_quarter, feature_one_eight = self.feature_extractor(torch.cat((imgs[i], depth), dim=1))
            else:
                feature_half, feature_quarter = self.feature_extractor(depth)
            feature_halves.append(feature_half)
            feature_quarters.append(feature_quarter)
            # feature_one_eights.append(feature_one_eight)
            pred_depths_quarters.append(F.interpolate(input=depth,
                                                      scale_factor=(1.0 / 4.0),
                                                      mode="nearest"))
            # pred_depths_one_eights.append(F.interpolate(input=depth,
            #                                           scale_factor=(1.0 / 8.0),
            #                                           mode="nearest"))

        rnn_intrinsics = intrinsics.clone()
        rnn_intrinsics[:, :, 0:2, :] = rnn_intrinsics[:, :, 0:2, :] / 4.0

        rnn_state = self.rnn(
            current_encoding=feature_quarters[0],
            # current_encoding=feature_one_eights[0],
            current_state=None,
            previous_pose=None,
            current_pose=poses[0],
            estimated_current_depth=pred_depths_quarters[0],
            camera_matrix=rnn_intrinsics[0],
        )
        weights = [1, 1]
        # weights = [1, 1, 1]
        for i in range(1, len(pred_depths)):
            rnn_state = self.rnn(
                current_encoding=feature_quarters[i],
                # current_encoding=feature_one_eights[i],
                current_state=rnn_state,
                previous_pose=poses[i - 1],
                current_pose=poses[i],
                estimated_current_depth=pred_depths_quarters[i],
                camera_matrix=rnn_intrinsics[i],
            )
            depth_full, depth_half = self.decoder(
                pred_depths[i], feature_halves[i],
                feature_quarters[i], rnn_state[0],
                imgs[i]
            )

            optimizer_loss = optimizer_loss + update_losses(
                predictions=[depth_half, depth_full],
                # predictions=[depth_quarter, depth_half, depth_full],
                weights=weights,
                groundtruth=gt_depths[i],
                is_training=is_training,
                l1_meter=l1_meter,
                l1_inv_meter=l1_inv_meter,
                l1_rel_meter=l1_rel_meter,
                huber_meter=huber_meter,
                loss_type="L1")

            # outputs['quarter_preds'].append(depth_quarter)
            outputs['half_preds'].append(depth_half)
            outputs['full_preds'].append(depth_full)

        return outputs, {'total_loss': optimizer_loss}


class FeatureExtractor(nn.Module):
    def __init__(self, alpha=1.0, in_channels=1):
        super(FeatureExtractor, self).__init__()
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(weights=None, progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
        )

        self.conv1 = nn.Sequential(
            MNASNet.layers._modules['8'],
            # nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # if one_eight_feat:
        # self.conv2 = MNASNet.layers._modules['9']

    def forward(self, x):
        # (B, 16, H/2, W/2)
        feature_half = self.conv0(x)
        # (B, 24, H/4, W/4)
        feature_quarter = self.conv1(feature_half)
        # (B, 40, H/8, W/8)
        # feature_one_eight = self.conv2(feature_quarter)
        return feature_half, feature_quarter
        # return feature_half, feature_quarter, feature_one_eight


class DepthRNN2(nn.Module):
    def __init__(self, cfg):
        super(DepthRNN2, self).__init__()
        self.cfg = cfg.MODEL

        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        self.load_img = cfg.DATA.LOAD_IMG
        if self.load_img:
            logger.info('Train with depth and image')
            self.feature_extractor = FeatureExtractor2(alpha, 4)
        else:
            logger.info('Train with depth')
            self.feature_extractor = FeatureExtractor2(alpha, 1)
        self.rnn = LSTMFusion(hidden_dim=40)
        self.decoder = Decoder2()

    def forward(self, inputs, is_training=True):
        outputs = defaultdict(list)

        # move batch dimension to 1
        if self.load_img:
            imgs = torch.permute(inputs['imgs'], (1, 0, 2, 3, 4))
        pred_depths = torch.permute(inputs['pred_depths'], (1, 0, 2, 3)).unsqueeze(2)
        gt_depths = torch.permute(inputs['gt_depths'], (1, 0, 2, 3)).unsqueeze(2)
        intrinsics = torch.permute(inputs['intrinsics'], (1, 0, 2, 3))
        poses = torch.permute(inputs['poses'], (1, 0, 2, 3))

        feature_halves, feature_quarters, feature_one_eights, pred_depths_one_eights = [], [], [], []

        optimizer_loss = 0
        l1_meter = LossMeter()
        huber_meter = LossMeter()
        l1_inv_meter = LossMeter()
        l1_rel_meter = LossMeter()

        for i, depth in enumerate(pred_depths):
            if self.load_img:
                feature_half, feature_quarter, feature_one_eight = self.feature_extractor(torch.cat((imgs[i], depth), dim=1))
            else:
                feature_half, feature_quarter, feature_one_eight = self.feature_extractor(depth)
            feature_halves.append(feature_half)
            feature_quarters.append(feature_quarter)
            feature_one_eights.append(feature_one_eight)
            pred_depths_one_eights.append(F.interpolate(input=depth,
                                                      scale_factor=(1.0 / 8.0),
                                                      mode="nearest"))

        rnn_intrinsics = intrinsics.clone()
        rnn_intrinsics[:, :, 0:2, :] = rnn_intrinsics[:, :, 0:2, :] / 4.0

        rnn_state = self.rnn(
            current_encoding=feature_one_eights[0],
            current_state=None,
            previous_pose=None,
            current_pose=poses[0],
            estimated_current_depth=pred_depths_one_eights[0],
            camera_matrix=rnn_intrinsics[0],
        )

        weights = [1, 1, 1]
        for i in range(1, len(pred_depths)):
            rnn_state = self.rnn(
                current_encoding=feature_one_eights[i],
                current_state=rnn_state,
                previous_pose=poses[i - 1],
                current_pose=poses[i],
                estimated_current_depth=pred_depths_one_eights[i],
                camera_matrix=rnn_intrinsics[i],
            )

            depth_full, depth_half, depth_quarter = self.decoder(
                pred_depths[i], feature_halves[i],
                feature_quarters[i], rnn_state[0]
            )

            optimizer_loss = optimizer_loss + update_losses(
                predictions=[depth_quarter, depth_half, depth_full],
                weights=weights,
                groundtruth=gt_depths[i],
                is_training=is_training,
                l1_meter=l1_meter,
                l1_inv_meter=l1_inv_meter,
                l1_rel_meter=l1_rel_meter,
                huber_meter=huber_meter,
                loss_type="L1")

            outputs['quarter_preds'].append(depth_quarter)
            outputs['half_preds'].append(depth_half)
            outputs['full_preds'].append(depth_full)

        return outputs, {'total_loss': optimizer_loss}


class FeatureExtractor2(nn.Module):
    def __init__(self, alpha=1.0, in_channels=1):
        super(FeatureExtractor2, self).__init__()
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(weights=None, progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
        )

        self.conv1 = MNASNet.layers._modules['8']
        self.conv2 = MNASNet.layers._modules['9']

    def forward(self, x):
        # (B, 16, H/2, W/2)
        feature_half = self.conv0(x)
        # (B, 24, H/4, W/4)
        feature_quarter = self.conv1(feature_half)
        # (B, 40, H/8, W/8)
        feature_one_eight = self.conv2(feature_quarter)
        return feature_half, feature_quarter, feature_one_eight