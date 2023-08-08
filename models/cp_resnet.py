# coding: utf-8
import math

import torch.nn as nn

import logging
logger = logging.getLogger("model")

layer_index_total = 0
first_RUN = True

def initialize_weights(module):
    """
    @param module: Resnet module entry
    initializes the weights of the network
    """

    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

def initialize_weights_fixup(module):
    if isinstance(module, BasicBlock):
        # He init, rescaled by Fixup multiplier
        b = module
        n = b.conv1.kernel_size[0] * b.conv1.kernel_size[1] * b.conv1.out_channels
        print(b.layer_index, math.sqrt(2. / n), layer_index_total ** (-0.5))
        b.conv1.weight.data.normal_(0, (layer_index_total ** (-0.5)) * math.sqrt(2. / n))
        b.conv2.weight.data.zero_()
        if b.shortcut._modules.get('conv') is not None:
            convShortcut = b.shortcut._modules.get('conv')
            n = convShortcut.kernel_size[0] * convShortcut.kernel_size[1] * convShortcut.out_channels
            convShortcut.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        if module.bias:
            module.bias.data.zero_()

def calc_padding(kernal):
    """
    @param kernal: kernel input
    @return: calculates padding required to get the same shape before entering the convolution
    """
    try:
        return kernal // 3
    except TypeError:
        return [k // 3 for k in kernal]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, k1=3, k2=3, groups=1):
        """
        @param in_channels: input channels to the block
        @param out_channels: output channels of the block
        @param k1: kernel size of the first convolution in the block
        @param k2:kernel size of the second convolution in the block
        @param groups: grouping applied to the block default 1 : No Grouping
        """
        super(BasicBlock, self).__init__()
        global layer_index_total
        self.layer_index = layer_index_total
        layer_index_total = layer_index_total + 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            groups=groups,
            kernel_size=k1,
            stride=stride,  # downsample with first conv
            padding=calc_padding(k1),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            groups=groups,
            kernel_size=k2,
            stride=1,
            padding=calc_padding(k2),
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_add = nn.quantized.FloatFunctional()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False,
                    groups=groups))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        # residual connection with addition compatible with quantization
        y = self.skip_add.add(y, self.shortcut(x))
        y = self.relu2(y)  # apply ReLU after addition
        return y

def safe_list_get (l, idx, default):
  try:
      return l[idx]
  except IndexError:
      return default

class Network(nn.Module):
    def __init__(self, config):
        """

        @type config: dict object contains the details about the network :
        input_shape: input shape of the network
        num_classes : number of the classes for the predictions
        channel_multiplier: multiply of the channels after each block ( changes the width of the network)
        base_channels: starting number of the channels at the beginning of the network
        cut_channels_s2: how much of the channels should be cut from the specific stage 2
        cut_channels_s3: how much of the channels should be cut from the specific stage 3
        n_blocks_per_stage: number of blocks in each stage

        """
        super(Network, self).__init__()
        input_shape = config['input_shape']
        n_classes = config['n_classes']
        channels_multiplier = config['channels_multiplier']
        base_channels = config['base_channels']
        cut_channels_s3 = config['cut_channels_s3']
        cut_channels_s2 = config['cut_channels_s2']

        n_blocks_per_stage = config.get("n_blocks_per_stage")

        self.pooling_padding = config.get("pooling_padding", 0) or 0
        self.maxpool_kernel = config.get("maxpool_kernel") or (2, 2)
        self.maxpool_stride = config.get("maxpool_stride") or (2, 2)

        n_channels = [
            base_channels,
            base_channels * channels_multiplier - cut_channels_s2,
            base_channels * channels_multiplier * channels_multiplier - cut_channels_s3
        ]

        self.in_c = nn.Sequential(nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=5,
            stride=2,
            padding=1,
            bias=False),
            nn.BatchNorm2d(n_channels[0]),
            nn.ReLU(True)
        )
        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage[0], BasicBlock,
            maxpool_pos=config['stage1']['maxpool_positions'], k1s=config['stage1']['k1s'], k2s=config['stage1']['k2s'],
            groups=config['stage1']['groups'], maxpool_pos0_kernel=config['stage1']['maxpool_pos0_kernel'],
            maxpool_pos0_stride=config['stage1']['maxpool_pos0_kernel'])
        self.stage2 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage[1], BasicBlock,
            maxpool_pos=config['stage2']['maxpool_positions'], k1s=config['stage2']['k1s'], k2s=config['stage2']['k2s'],
            groups=config['stage2']['groups'],)
        if n_blocks_per_stage[2] == 0:
            self.stage3 = nn.Sequential()
            n_channels[2] = n_channels[1]
            print("WARNING: stage3 removed")
        else:
            self.stage3 = self._make_stage(
                n_channels[1], n_channels[2], n_blocks_per_stage[2], BasicBlock,
                maxpool_pos=config['stage3']['maxpool_positions'], k1s=config['stage3']['k1s'], k2s=config['stage3']['k2s'],
                groups=config['stage3']['groups'],)

        # sets width_per_block
        self.width_per_block = n_channels

        ff_list = []
        ff_list += [nn.Conv2d(
            n_channels[2],
            n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False),
            nn.BatchNorm2d(n_classes),
        ]

        ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )

        # initialize weights
        self.apply(initialize_weights)
        if isinstance(self.feed_forward[0], nn.Conv2d):
            self.feed_forward[0].weight.data.zero_()
        self.apply(initialize_weights_fixup)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, maxpool_pos=set(), k1s=[3, 3, 3, 3, 3, 3],
                    k2s=[3, 3, 3, 3, 3, 3], groups=1, maxpool_pos0_kernel=None, maxpool_pos0_stride=None):
        """

        @param in_channels: in channels to the stage
        @param out_channels: out channels of the stage
        @param n_blocks: number of blocks in the stage
        @param block: block type in the stage ( basic block )
        @param maxpool_pos: location of the max pooling in the stage
        @param k1s: list of first convolution kernels in the stage for each block one value should exist
        @param k2s: list of second convolution kernels in the stage for each block one value should exist
        @param groups: groups of the block , if group of a stage more than one , all convolutions of the stage are grouped
        @param maxpool_pos0_kernel: only for first stage, if we want a different kernel for the position 0
        @param maxpool_pos0_stride: only for first stage, if we want a different stride for the position 0
        @return: an object containing several resnet blocks with arbitrary config

        """

        if maxpool_pos0_kernel is None:
            maxpool_pos0_kernel = self.maxpool_kernel

        if maxpool_pos0_stride is None:
            maxpool_pos0_stride = self.maxpool_stride

        stride = 1
        stage = nn.Sequential()
        if 0 in maxpool_pos:
            stage.add_module("maxpool{}_{}".format(0, 0)
                             , nn.MaxPool2d(kernel_size=maxpool_pos0_kernel,
                                            stride=maxpool_pos0_stride,
                                            padding=self.pooling_padding))
        for index in range(n_blocks):
            stage.add_module('block{}'.format(index + 1),
                             block(in_channels,
                                   out_channels,
                                   stride=stride,
                                   k1=safe_list_get(k1s,index,1),
                                   k2=safe_list_get(k2s,index,1),
                                   groups=groups,
                                   ))

            in_channels = out_channels
            # if index + 1 in maxpool:
            for m_i, mp_pos in enumerate(maxpool_pos):
                if index + 1 == mp_pos:
                    stage.add_module("maxpool{}_{}".format(index + 1, m_i)
                                     , nn.MaxPool2d(kernel_size=self.maxpool_kernel,
                                                    stride=self.maxpool_stride,
                                                    padding=self.pooling_padding))
        return stage

    def half_damper(self):
        global cach_damp
        for k in cach_damp.keys():
            cach_damp[k] = cach_damp[k].half()

    def _forward_conv(self, x):
        global first_RUN
        if first_RUN: print("x:", x.size())
        x = self.in_c(x)
        if first_RUN: print("in_c:", x.size())
        x = self.stage1(x)
        if first_RUN: print("stage1:", x.size())
        x = self.stage2(x)
        if first_RUN: print("stage2:", x.size())
        x = self.stage3(x)
        if first_RUN: print("stage3:", x.size())

        return x

    def forward(self, x):
        global first_RUN

        x = self._forward_conv(x)
        x = self.feed_forward(x)
        if first_RUN: print("feed_forward:", x.size())
        logit = x.squeeze(2).squeeze(2)
        if first_RUN: print("logit:", logit.size())
        first_RUN = False
        return logit


def get_model(rho=8, in_channels=1, arch="cp_resnet", n_classes=10,
              base_channels=32, cut_channels_s2=0, cut_channels_s3=36, channels_multiplier=2, n_blocks=(2, 1, 1),
              s1_group=1, s2_group=2, s3_group=1,
              maxpool_pos_stage1=None, maxpool_pos_stage2=None, maxpool_pos_stage3=None,
              maxpool_stage1_pos0_kernel=(2, 2), maxpool_stage1_pos0_stride=(2, 2),
              maxpool_kernel=(2, 1), maxpool_stride=(2, 1)):
    """

    @param rho: controls the receptive field of the network ,4 is default , rho>4 increase rf and rho<4 decrease it
    @param in_channels: input channels to the network for the audio its by default 1
    @param arch: name of the architecture for saving
    @param n_classes: number of the classes to create the network
    @param base_channels: starting channels of the network
    @param cut_channels_s2: controls how many channels should be cut from stage 2 channels
    @param cut_channels_s3: controls how many channels should be cut from stage 3 channels
    @param channels_multiplier: controls the increase in the width of the network after each stage
    @param n_blocks: number of blocks that should exist in each stage
    @param s1_group: amount of grouping that should be applied to stage 1
    @param s2_group: amount of grouping that should be applied to stage 2
    @param s3_group: amount of grouping that should be applied to stage 3
    @param maxpool_pos_stage1: on which position a maxpool-layer should be implemented in stage 1
    @param maxpool_pos_stage2: on which position a maxpool-layer should be implemented in stage 2
    @param maxpool_pos_stage3: on which position a maxpool-layer should be implemented in stage 3
    @param maxpool_kernel: the kernel the maxpool-layer should have
    @param maxpool_stride: the stride the maxpool-layer should have
    @return: full neural network model based on the specified configs.
    """

    if maxpool_pos_stage1 is None:
        maxpool_pos_stage1 = [0, 1, 2, 4]

    if maxpool_pos_stage2 is None:
        maxpool_pos_stage2 = []

    if maxpool_pos_stage3 is None:
        maxpool_pos_stage3 = []

    extra_kernal_rf = rho - 4

    model_config = {
        "arch": arch,
        "base_channels": base_channels,
        "cut_channels_s2": cut_channels_s2,
        "cut_channels_s3": cut_channels_s3,
        "channels_multiplier": channels_multiplier,
        "input_shape": [
            10,
            in_channels,
            -1,
            -1
        ],
        "n_blocks_per_stage": n_blocks,
        "n_classes": n_classes,
        "maxpool_kernel": maxpool_kernel,
        "maxpool_stride": maxpool_stride,
        "stage1": {"maxpool_positions": maxpool_pos_stage1,
                   "maxpool_pos0_kernel": maxpool_stage1_pos0_kernel,
                   "maxpool_pos0_stride": maxpool_stage1_pos0_stride,
                   "k1s": [3,
                           3 - (-extra_kernal_rf > 2) * 2],
                   "k2s": [1,
                           3 - (-extra_kernal_rf > 1) * 2],
                   "groups": s1_group},

        "stage2": {"maxpool_positions": maxpool_pos_stage2, "k1s": [3 - (-extra_kernal_rf > 0) * 2,
                                          1 + (extra_kernal_rf > 1) * 2],
                   "k2s": [1 + (extra_kernal_rf > 0) * 2,
                           1 + (extra_kernal_rf > 2) * 2],
                   "groups": s2_group},

        "stage3": {"maxpool_positions": maxpool_pos_stage3,
                   "k1s": [1 + (extra_kernal_rf > 3) * 2,
                           1 + (extra_kernal_rf > 5) * 2],
                   "k2s": [1 + (extra_kernal_rf > 4) * 2,
                           1 + (extra_kernal_rf > 6) * 2],
                   "groups": s3_group},
    }

    m = Network(model_config)
    return m