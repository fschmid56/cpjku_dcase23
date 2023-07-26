# coding: utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from librosa.filters import mel as librosa_mel_fn
from functools import partial

from helpers.utils import mixstyle

import logging

logger = logging.getLogger("model")


RELU_INPLACE_ACT= partial(F.relu, inplace=True)
ACT_FUNC= RELU_INPLACE_ACT


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

layer_index_total = 0


def initialize_weights_fixup(module):
    if isinstance(module, AttentionAvg):
        print("AttentionAvg init..")
        module.forw_conv[0].weight.data.zero_()
        module.atten[0].bias.data.zero_()
        nn.init.kaiming_normal_(module.atten[0].weight.data, mode='fan_in', nonlinearity="sigmoid")
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
    if isinstance(module, nn.Conv2d):
        pass
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


first_RUN = True


def calc_padding(kernal):
    try:
        return kernal // 3
    except TypeError:
        return [k // 3 for k in kernal]


class AttentionAvg(nn.Module):

    def __init__(self, in_channels, out_channels, sum_all=True):
        super(AttentionAvg, self).__init__()
        self.sum_dims = [2, 3]
        if sum_all:
            self.sum_dims = [1, 2, 3]
        self.forw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.atten = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        a1 = self.forw_conv(x)
        atten = self.atten(x)
        num = atten.size(2) * atten.size(3)
        asum = atten.sum(dim=self.sum_dims, keepdim=True) + 1e-8
        return a1 * atten * num / asum

groups_num=1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, k1=3, k2=3):
        super(BasicBlock, self).__init__()
        global layer_index_total
        self.layer_index = layer_index_total
        layer_index_total = layer_index_total + 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            groups=groups_num,
            kernel_size=k1,
            stride=stride,  # downsample with first conv
            padding=calc_padding(k1),
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            groups=groups_num,
            kernel_size=k2,
            stride=1,
            padding=calc_padding(k2),
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = ACT_FUNC(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = ACT_FUNC(y)  # apply ReLU after addition
        return y

def safe_list_get (l, idx, default):
  try:
      return l[idx]
  except IndexError:
      return default

class Network(nn.Module):
    def __init__(self, config,ff_weight_anticolapse_limit=0.5):
        super(Network, self).__init__()
        global groups_num
        groups_num = config.get("groups_num", 1) or 1
        if groups_num>1:
            print(f"Groups = {groups_num}")
        input_shape = config['input_shape']
        n_classes = config['n_classes']

        channels_multiplier = config['channels_multiplier']
        if channels_multiplier != 2:
            logger.warning(f"channels_multiplier={channels_multiplier}")
        base_channels = config['base_channels']
        block_type = config['block_type']
        depth = config['depth']
        self.pooling_padding = config.get("pooling_padding", 0) or 0
        self.use_raw_spectograms = config.get("use_raw_spectograms") or False
        self.apply_softmax = config.get("apply_softmax") or False
        self.return_embed = config.get("return_embed") or False
        self.maxpool_kernel = config.get("maxpool_kernel") or (2, 2)
        self.maxpool_stride = config.get("maxpool_stride") or (2, 2)

        self.mixstyle_p = config.get("mixstyle_p", 0) or 0
        self.mixstyle_alpha = config.get("mixstyle_alpha", 0) or 0
        self.mixstyle_where = config.get("mixstyle_where", []) or []

        self.rfn_lambda = config.get("rfn_lambda", 0) or 0
        self.rfn_where = config.get("rfn_where", []) or []

        assert block_type in ['basic', 'bottleneck']
        if self.use_raw_spectograms:
            mel_basis = librosa_mel_fn(
                22050, 2048, 256)
            mel_basis = torch.from_numpy(mel_basis).float()
            self.register_buffer('mel_basis', mel_basis)
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth
        n_blocks_per_stage = [n_blocks_per_stage, n_blocks_per_stage, n_blocks_per_stage]

        if config.get("n_blocks_per_stage") is not None:
            logger.warning(
                "n_blocks_per_stage is specified ignoring the depth param, nc=" + str(config.get("n_channels")))
            n_blocks_per_stage = config.get("n_blocks_per_stage")

        n_channels = config.get("n_channels")
        if n_channels is None:
            n_channels = [
                base_channels,
                base_channels * channels_multiplier * block.expansion,
                base_channels * channels_multiplier * channels_multiplier * block.expansion
            ]
        if config.get("grow_a_lot"):
            n_channels[2] = base_channels * 8 * block.expansion

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
            n_channels[0], n_channels[0], n_blocks_per_stage[0], block, stride=1, maxpool=config['stage1']['maxpool'],
            k1s=config['stage1']['k1s'], k2s=config['stage1']['k2s'])
        if n_blocks_per_stage[1] == 0:
            self.stage2 = nn.Sequential()
            n_channels[1] = n_channels[0]
            print("WARNING: stage2 removed")
        else:
            self.stage2 = self._make_stage(
                n_channels[0], n_channels[1], n_blocks_per_stage[1], block, stride=1, maxpool=config['stage2']['maxpool'],
                k1s=config['stage2']['k1s'], k2s=config['stage2']['k2s'])
        if n_blocks_per_stage[2] == 0:
            self.stage3 = nn.Sequential()
            n_channels[2] = n_channels[1]
            print("WARNING: stage3 removed")
        else:
            self.stage3 = self._make_stage(
                n_channels[1], n_channels[2], n_blocks_per_stage[2], block, stride=1, maxpool=config['stage3']['maxpool'],
                k1s=config['stage3']['k1s'], k2s=config['stage3']['k2s'])
        # sets width_per_block
        self.width_per_block = n_channels

        ff_list = []
        if config.get("attention_avg"):
            if config.get("attention_avg") == "sum_all":
                ff_list.append(AttentionAvg(n_channels[2], n_classes, sum_all=True))
            else:
                ff_list.append(AttentionAvg(n_channels[2], n_classes, sum_all=False))
        else:
            no_bn_out = config.get("no_bn_out") or False
            ff_list += [nn.Conv2d(
                n_channels[2],
                n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=no_bn_out),
                nn.BatchNorm2d(n_classes),
            ]
            if no_bn_out:
                # remove the bn
                ff_list[-1] = nn.Sequential() # no op
            ff_list[-2].weight_anticolapse_limit = ff_weight_anticolapse_limit

        self.stop_before_global_avg_pooling = False
        if config.get("stop_before_global_avg_pooling"):
            self.stop_before_global_avg_pooling = True
        else:
            ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )
        # # compute conv feature size
        # with torch.no_grad():
        #     self.feature_size = self._forward_conv(
        #         torch.zeros(*input_shape)).view(-1).shape[0]
        #
        # self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        if config.get("weight_init") == "fixup":
            self.apply(initialize_weights)
            if isinstance(self.feed_forward[0], nn.Conv2d):
                self.feed_forward[0].weight.data.zero_()
            self.apply(initialize_weights_fixup)
        else:
            self.apply(initialize_weights)
        self.use_check_point = config.get("use_check_point") or False

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride, maxpool=set(), k1s=[3, 3, 3, 3, 3, 3],
                    k2s=[3, 3, 3, 3, 3, 3]):
        stage = nn.Sequential()
        if 0 in maxpool:
            stage.add_module("maxpool{}_{}".format(0, 0)
                             , nn.MaxPool2d(kernel_size=self.maxpool_kernel,
                                            stride=self.maxpool_stride,
                                            padding=self.pooling_padding))
        for index in range(n_blocks):
            stage.add_module('block{}'.format(index + 1),
                             block(in_channels,
                                   out_channels,
                                   stride=stride, k1= safe_list_get(k1s,index,1), k2=safe_list_get(k2s,index,1) ))

            in_channels = out_channels
            stride = 1
            # if index + 1 in maxpool:
            for m_i, mp_pos in enumerate(maxpool):
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

        if self.use_check_point:
            if first_RUN: print("use_check_point:", x.size())
            return checkpoint_sequential([self.stage1, self.stage2, self.stage3], 3,
                                         (x))

        # stage 1
        x = self.stage1(x)

        if self.training:
            if 1 in self.rfn_where:
                x = rfn(x, self.rfn_lambda)

            if 1 in self.mixstyle_where and self.mixstyle_p > 0:
                x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)

        if first_RUN: print("stage1:", x.size())

        # stage 2
        x = self.stage2(x)

        if self.training:
            if 2 in self.rfn_where:
                x = rfn(x, self.rfn_lambda)

            if 2 in self.mixstyle_where and self.mixstyle_p > 0:
                x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)

        if first_RUN: print("stage2:", x.size())

        # stage 3
        x = self.stage3(x)

        if self.training:
            if 3 in self.rfn_where:
                x = rfn(x, self.rfn_lambda)

            if 3 in self.mixstyle_where and self.mixstyle_p > 0:
                x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)

        if first_RUN: print("stage3:", x.size())
        return x

    def forward(self, x):
        global first_RUN
        if self.use_raw_spectograms:
            if first_RUN: print("raw_x:", x.size())
            x = torch.log10(torch.sqrt((x * x).sum(dim=3)))
            if first_RUN: print("log10_x:", x.size())
            x = torch.matmul(self.mel_basis, x)
            if first_RUN: print("mel_basis_x:", x.size())
            x = x.unsqueeze(1)
        x = self._forward_conv(x)
        features = x
        x = self.feed_forward(x)
        if first_RUN: print("feed_forward:", x.size())
        if self.stop_before_global_avg_pooling:
            first_RUN = False
            return x
        logit = x.squeeze(2).squeeze(2)

        if first_RUN: print("logit:", logit.size())
        if self.apply_softmax:
            logit = torch.softmax(logit, 1)
        first_RUN = False
        if self.return_embed:
            return logit, features
        return logit

    def reset_weights_of_last_block(self, block=3):
        if block == 3:
            b = self.stage3.block3
        else:
            b = self.stage3.block4

        n = b.conv1.kernel_size[0] * b.conv1.kernel_size[1] * b.conv1.out_channels
        print(b.layer_index, math.sqrt(2. / n), layer_index_total ** (-0.5))
        b.conv1.weight.data.normal_(0, (layer_index_total ** (-0.5)) * math.sqrt(2. / n))
        b.conv2.weight.data.zero_()
        if b.shortcut._modules.get('conv') is not None:
            convShortcut = b.shortcut._modules.get('conv')
            n = convShortcut.kernel_size[0] * convShortcut.kernel_size[1] * convShortcut.out_channels
            convShortcut.weight.data.normal_(0, math.sqrt(2. / n))

        b.bn1.weight.data.fill_(1)
        b.bn1.bias.data.zero_()

        b.bn2.weight.data.fill_(1)
        b.bn2.bias.data.zero_()

    def reset_weights_of_classifier(self):

        ff_list = []

        ff_list += [nn.Conv2d(
            512,
            10,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False),
            nn.BatchNorm2d(10),
        ]

        ff_list[-2].weight_anticolapse_limit = 0.5

        ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )

        if isinstance(self.feed_forward[0], nn.Conv2d):
            self.feed_forward[0].weight.data.zero_()


def get_model(rho=7, in_channels=1, arch="cp_resnet_eusipco", cut_stage2=0, cut_stage3=0, config_only=False, n_classes=10,
              base_channels=128, channels_multiplier=2, spread="default", weight_init="fixup",
              ff_weight_anticolapse_limit=0.5, groups_num=1, n_channels=None, no_bn_out=False,
              maxpool_stage1=None, maxpool_stage2=None, maxpool_stage3=None,
              maxpool_kernel=(2, 2), maxpool_stride=(2, 2), model_config_overrides={}):
    # extra receptive checking
    if maxpool_stage1 is None:
        maxpool_stage1 = [1, 2, 4]

    if maxpool_stage2 is None:
        maxpool_stage2 = []

    if maxpool_stage3 is None:
        maxpool_stage3 = []

    extra_kernal_rf = rho - 7

    model_config = {
        "arch": arch,
        "base_channels": base_channels,
        "channels_multiplier":channels_multiplier,
        "block_type": "basic",
        "groups_num": groups_num,
        "depth": 26,
        "input_shape": [
            10,
            in_channels,
            -1,
            -1
        ],
        "n_channels": n_channels,
        "n_blocks_per_stage": [4, 4 - cut_stage2, 4 - cut_stage3],
        "multi_label": False,
        "n_classes": n_classes,
        "prediction_threshold": 0.4,
        "maxpool_kernel": maxpool_kernel,
        "maxpool_stride": maxpool_stride,
        "stage1": {"maxpool": maxpool_stage1,
                   "k1s": [3,
                           3 - (-extra_kernal_rf > 6) * 2,
                           3 - (-extra_kernal_rf > 4) * 2,
                           3 - (-extra_kernal_rf > 2) * 2],
                   "k2s": [1,
                           3 - (-extra_kernal_rf > 5) * 2,
                           3 - (-extra_kernal_rf > 3) * 2,
                           3 - (-extra_kernal_rf > 1) * 2]},

        "stage2": {"maxpool": maxpool_stage2, "k1s": [3 - (-extra_kernal_rf > 0) * 2,
                                          1 + (extra_kernal_rf > 1) * 2,
                                          1 + (extra_kernal_rf > 3) * 2,
                                          1 + (extra_kernal_rf > 5) * 2],
                   "k2s": [1 + (extra_kernal_rf > 0) * 2,
                           1 + (extra_kernal_rf > 2) * 2,
                           1 + (extra_kernal_rf > 4) * 2,
                           1 + (extra_kernal_rf > 6) * 2]},
        "stage3": {"maxpool": maxpool_stage3,
                   "k1s": [1 + (extra_kernal_rf > 7) * 2,
                           1 + (extra_kernal_rf > 9) * 2,
                           1 + (extra_kernal_rf > 11) * 2,
                           1 + (extra_kernal_rf > 13) * 2],
                   "k2s": [1 + (extra_kernal_rf > 8) * 2,
                           1 + (extra_kernal_rf > 10) * 2,
                           1 + (extra_kernal_rf > 12) * 2,
                           1 + (extra_kernal_rf > 14) * 2]},
        "use_bn": True,
        "no_bn_out": no_bn_out,
        "weight_init": weight_init
    }
    if spread=="o1":
        update_spread(model_config)
    elif spread=="default":
        # check for removed extra filters
        if ((extra_kernal_rf > -1 and cut_stage2 > 3)
                or (extra_kernal_rf > 1 and cut_stage2 > 2)
                or (extra_kernal_rf > 3 and cut_stage2 > 1)
                or (extra_kernal_rf > 5 and cut_stage2 > 0)):
            raise RuntimeError(f"incompatible rho value: rho={rho}, cut_stage2={cut_stage2}!")
        if ((extra_kernal_rf > 7 and cut_stage3 > 3)
                or (extra_kernal_rf > 9 and cut_stage3 > 2)
                or (extra_kernal_rf > 11 and cut_stage3 > 1)
                or (extra_kernal_rf > 13 and cut_stage3 > 0)):
            raise RuntimeError(f"incompatible rho value: rho={rho}, cut_stage3={cut_stage3}!")
    else:
        raise ValueError(f"Unknown value for spread={spread}!")
    # override model_config
    if config_only:
        return model_config
    m = Network(model_config, ff_weight_anticolapse_limit)
    print(m)
    print("")
    print("rho = ", rho)
    print(model_config['stage1'])
    print(model_config['stage2'])
    print(model_config['stage3'])
    return m


def update_spread(orig_config,rho):
    extra_kernal_rf = rho - 7
    orig_config.update({
        "stage1": {"maxpool": [1, 3],
                   "k1s": [3,
                           3 - (-extra_kernal_rf > 6) * 2,
                           3 - (-extra_kernal_rf > 5) * 2,
                           3 - (-extra_kernal_rf > 4) * 2],
                   "k2s": [1,
                          1,
                           1,
                           1]},

        "stage2": {"maxpool": [3], "k1s": [3 - (-extra_kernal_rf > 3) * 2,
                                          3 - (-extra_kernal_rf > 2) * 2,
                                          3 - (-extra_kernal_rf > 1) * 2,
                                          3 - (-extra_kernal_rf > 0) * 2],
                   "k2s": [1 ,
                           1 ,
                           1 ,
                           1 ]},
        "stage3": {"maxpool": [],
                   "k1s": [  1 + (extra_kernal_rf > 0) * 2,
                             1 + (extra_kernal_rf > 1) * 2,
                             1 + (extra_kernal_rf > 2) * 2,
                             1 + (extra_kernal_rf > 3) * 2],
                   "k2s": [1,
                           1,
                           1,
                           1]},})