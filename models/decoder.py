import torch
import torch.nn.functional as F

def conv_layer(input_channels, output_channels, kernel_size, stride, apply_bn_relu):
    if apply_bn_relu:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                bias=False),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True))
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                bias=False))


def depth_layer_3x3(input_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(input_channels, 1, 3, padding=1),
        # torch.nn.Sigmoid()
        torch.nn.Tanh()
    )


class UpconvolutionLayer(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(UpconvolutionLayer, self).__init__()
        self.conv = conv_layer(input_channels=input_channels,
                               output_channels=output_channels,
                               stride=1,
                               kernel_size=kernel_size,
                               apply_bn_relu=True)

    def forward(self, x):
        x = torch.nn.functional.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x


class DecoderBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, apply_bn_relu, plus_one):
        super(DecoderBlock, self).__init__()
        # Upsample the input coming from previous layer
        self.up_convolution = UpconvolutionLayer(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size
        )

        if plus_one:
            next_input_channels = input_channels + 1
        else:
            next_input_channels = input_channels

        # Aggregate skip and upsampled input
        self.convolution1 = conv_layer(input_channels=next_input_channels,
                                       output_channels=output_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       apply_bn_relu=True)

        # Learn from aggregation
        self.convolution2 = conv_layer(input_channels=output_channels,
                                       output_channels=output_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       apply_bn_relu=apply_bn_relu)

    def forward(self, x, skip, depth):
        x = self.up_convolution(x)

        if depth is None:
            x = torch.cat([x, skip], dim=1)
        else:
            depth = torch.nn.functional.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, skip, depth], dim=1)

        x = self.convolution1(x)
        x = self.convolution2(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, min_depth=0.25, max_depth=6.0, base_channels=8):
        super(Decoder, self).__init__()
        self.inverse_depth_base = 1 / max_depth
        self.inverse_depth_multiplier = 1 / min_depth - 1 / max_depth

        # self.decoder_block1 = DecoderBlock(input_channels=base_channels * 16,
        #                                    output_channels=base_channels * 8,
        #                                    kernel_size=3,
        #                                    apply_bn_relu=True,
        #                                    plus_one=False)
        #
        # self.decoder_block2 = DecoderBlock(input_channels=base_channels * 8,
        #                                    output_channels=base_channels * 4,
        #                                    kernel_size=3,
        #                                    apply_bn_relu=True,
        #                                    plus_one=True)

        # self.decoder_block3 = DecoderBlock(input_channels=base_channels * 4,
        #                                    output_channels=base_channels * 2,
        #                                    kernel_size=3,
        #                                    apply_bn_relu=True,
        #                                    plus_one=True)

        self.decoder_block4 = DecoderBlock(input_channels=base_channels * 3,
                                           output_channels=base_channels * 1,
                                           kernel_size=5,
                                           apply_bn_relu=True,
                                           plus_one=False)

        self.refine = torch.nn.Sequential(
            conv_layer(
                input_channels=base_channels + 5,
                # input_channels=base_channels + 2,
                output_channels=base_channels,
                kernel_size=5, stride=1, apply_bn_relu=True),
            conv_layer(
                input_channels=base_channels,
                output_channels=base_channels,
                kernel_size=5, stride=1, apply_bn_relu=True))

        # self.depth_layer_one_sixteen = depth_layer_3x3(hyper_channels * 8)
        # self.depth_layer_one_eight = depth_layer_3x3(hyper_channels * 4)
        # self.depth_layer_quarter = depth_layer_3x3(base_channels * 2)
        self.depth_layer_half = depth_layer_3x3(base_channels)
        self.depth_layer_full = depth_layer_3x3(base_channels)

    def forward(self, pred_depth, feature_half, feature_quarter, rnn_state, image=None):
        # work on cost volume
        # decoder_block1 = self.decoder_block1(bottom, skip3, None)
        # sigmoid_depth_one_sixteen = self.depth_layer_one_sixteen(decoder_block1)
        # inverse_depth_one_sixteen = self.inverse_depth_multiplier * sigmoid_depth_one_sixteen + self.inverse_depth_base
        #
        # decoder_block2 = self.decoder_block2(decoder_block1, skip2, sigmoid_depth_one_sixteen)
        # sigmoid_depth_one_eight = self.depth_layer_one_eight(decoder_block2)
        # inverse_depth_one_eight = self.inverse_depth_multiplier * sigmoid_depth_one_eight + self.inverse_depth_base

        # pred_depth_quarter = torch.nn.functional.interpolate(pred_depth, scale_factor=(1.0 / 4.0), mode='nearest')
        #
        # decoder_block3 = self.decoder_block3(rnn_state, feature_quarter, None)
        # residual_depth_quarter = self.depth_layer_quarter(decoder_block3)
        # depth_quarter = pred_depth_quarter + residual_depth_quarter

        pred_depth_half = torch.nn.functional.interpolate(pred_depth, scale_factor=(1.0 / 2.0), mode='nearest')

        decoder_block4 = self.decoder_block4(rnn_state, feature_half, None)
        # decoder_block4 = self.decoder_block4(rnn_state, feature_half, residual_depth_quarter)
        residual_depth_half = self.depth_layer_half(decoder_block4)
        depth_half = pred_depth_half + residual_depth_half

        scaled_depth = F.interpolate(
            residual_depth_half, scale_factor=2, mode='bilinear', align_corners=True
        )
        scaled_decoder = F.interpolate(
            decoder_block4, scale_factor=2, mode='bilinear', align_corners=True
        )
        # scaled_combined = torch.cat([scaled_decoder, scaled_depth, pred_depth], dim=1)
        scaled_combined = torch.cat([scaled_decoder, scaled_depth, pred_depth, image], dim=1)
        scaled_combined = self.refine(scaled_combined)
        residual_depth_full = self.depth_layer_full(scaled_combined)
        depth_full = pred_depth + residual_depth_full

        return depth_full, depth_half
        # return depth_full, depth_half, depth_quarter


class DecoderBlock2(torch.nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels, kernel_size, apply_bn_relu, plus_one):
        super(DecoderBlock2, self).__init__()
        self.up_convolution = UpconvolutionLayer(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size
        )

        if plus_one:
            next_input_channels = skip_channels + output_channels + 1
        else:
            next_input_channels = skip_channels + output_channels

        # Aggregate skip and upsampled input
        self.convolution1 = conv_layer(input_channels=next_input_channels,
                                       output_channels=output_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       apply_bn_relu=True)

        # Learn from aggregation
        self.convolution2 = conv_layer(input_channels=output_channels,
                                       output_channels=output_channels,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       apply_bn_relu=apply_bn_relu)

    def forward(self, x, skip, depth):
        x = self.up_convolution(x)

        if depth is None:
            x = torch.cat([x, skip], dim=1)
        else:
            depth = torch.nn.functional.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, skip, depth], dim=1)

        x = self.convolution1(x)
        x = self.convolution2(x)
        return x

class Decoder2(torch.nn.Module):
    def __init__(self, base_channels=8):
        super(Decoder2, self).__init__()

        self.decoder_block3 = DecoderBlock2(input_channels=base_channels * 5,
                                            skip_channels=base_channels * 3,
                                           output_channels=base_channels * 3,
                                           kernel_size=3,
                                           apply_bn_relu=True,
                                           plus_one=False)

        self.decoder_block4 = DecoderBlock2(input_channels=base_channels * 3,
                                            skip_channels=base_channels * 2,
                                           output_channels=base_channels * 2,
                                           kernel_size=5,
                                           apply_bn_relu=True,
                                           plus_one=True)

        self.refine = torch.nn.Sequential(
            conv_layer(
                input_channels=base_channels * 2 + 2,
                output_channels=base_channels * 2,
                kernel_size=5,
                stride=1,
                apply_bn_relu=True
            ),
            conv_layer(
                input_channels=base_channels * 2,
                output_channels=base_channels * 2,
                kernel_size=5,
                stride=1,
                apply_bn_relu=True
            ))

        self.depth_layer_quarter = depth_layer_3x3(base_channels * 3)
        self.depth_layer_half = depth_layer_3x3(base_channels * 2)
        self.depth_layer_full = depth_layer_3x3(base_channels * 2)

    def forward(self, pred_depth, feature_half, feature_quarter, rnn_state):
        pred_depth_quarter = F.interpolate(pred_depth, scale_factor=0.25, mode='nearest')
        pred_depth_half = F.interpolate(pred_depth, scale_factor=0.5, mode='nearest')

        decoder_block3 = self.decoder_block3(rnn_state, feature_quarter, None)
        residual_depth_quarter = self.depth_layer_quarter(decoder_block3)
        depth_quarter = pred_depth_quarter + residual_depth_quarter

        decoder_block4 = self.decoder_block4(decoder_block3, feature_half, residual_depth_quarter)
        residual_depth_half = self.depth_layer_half(decoder_block4)
        depth_half = pred_depth_half + residual_depth_half

        scaled_depth = F.interpolate(
            residual_depth_half, scale_factor=2, mode='bilinear', align_corners=True
        )
        scaled_decoder = F.interpolate(
            decoder_block4, scale_factor=2, mode='bilinear', align_corners=True
        )
        scaled_combined = torch.cat([scaled_decoder, scaled_depth, pred_depth], dim=1)

        scaled_combined = self.refine(scaled_combined)
        residual_depth_full = self.depth_layer_full(scaled_combined)
        depth_full = pred_depth + residual_depth_full

        return depth_full, depth_half, depth_quarter