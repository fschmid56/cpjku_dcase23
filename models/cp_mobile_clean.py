import torch
import torch.nn as nn
from models.helpers.utils import make_divisible
from torch.ao.quantization import QuantStub, DeQuantStub
from torchvision.ops.misc import ConvNormActivation


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GRN(nn.Module):
    """
    global response normalization as introduced in https://arxiv.org/pdf/2301.00808.pdf
    """

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # dequantize and quantize since torch.norm not implemented for quantized tensors
        x = self.dequant(x)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)

        x = self.gamma * (x * nx) + self.beta + x
        return self.quant(x)


class CPMobileBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expansion_rate,
            stride
    ):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)

        # create the three factorized convs that make up our block
        exp_conv = ConvNormActivation(in_channels,
                                      exp_channels,
                                      kernel_size=1,
                                      stride=1,
                                      norm_layer=nn.BatchNorm2d,
                                      activation_layer=nn.ReLU,
                                      inplace=False
                                      )

        # depthwise convolution with possible stride
        depth_conv = ConvNormActivation(exp_channels,
                                        exp_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        groups=exp_channels,
                                        norm_layer=nn.BatchNorm2d,
                                        activation_layer=nn.ReLU,
                                        inplace=False
                                        )

        proj_conv = ConvNormActivation(exp_channels,
                                       out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       norm_layer=nn.BatchNorm2d,
                                       activation_layer=None,
                                       inplace=False
                                       )

        self.after_block_norm = GRN()
        self.after_block_activation = nn.ReLU()

        if in_channels == out_channels:
            self.use_shortcut = True
            if stride == 1 or stride == (1, 1):
                self.shortcut = nn.Sequential()
            else:
                # average pooling required for shortcut
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Sequential()
                )
        else:
            self.use_shortcut = False

        self.block = nn.Sequential(
            exp_conv,
            depth_conv,
            proj_conv
        )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_shortcut:
            x = self.skip_add.add(self.block(x), self.shortcut(x))
        else:
            x = self.block(x)
        x = self.after_block_norm(x)
        x = self.after_block_activation(x)
        return x


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        n_blocks = config['n_blocks']
        strides = config['strides']
        n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8)
                                                for stage_id in range(n_stages)]
        self.total_block_count = 0

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.in_c = nn.Sequential(
            ConvNormActivation(in_channels,
                               channels_per_stage[0] // 4,
                               kernel_size=3,
                               stride=2,
                               inplace=False
                               ),
            ConvNormActivation(channels_per_stage[0] // 4,
                               channels_per_stage[0],
                               activation_layer=torch.nn.ReLU,
                               kernel_size=3,
                               stride=2,
                               inplace=False
                               ),
        )

        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(channels_per_stage[stage_id],
                                     channels_per_stage[stage_id + 1],
                                     n_blocks[stage_id],
                                     strides=strides,
                                     expansion_rate=expansion_rate
                                     )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        ff_list = []
        ff_list += [nn.Conv2d(
            channels_per_stage[-1],
            n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False),
            nn.BatchNorm2d(n_classes),
        ]

        ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )

        self.apply(initialize_weights)

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    n_blocks,
                    strides,
                    expansion_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f'b{block_id}'
            self.total_block_count = self.total_block_count + 1
            if bname in strides:
                stride = strides[bname]
            else:
                stride = (1, 1)

            block = self._make_block(
                in_channels,
                out_channels,
                stride=stride,
                expansion_rate=expansion_rate
            )
            stage.add_module(bname, block)

            in_channels = out_channels
        return stage

    def _make_block(self,
                    in_channels,
                    out_channels,
                    stride,
                    expansion_rate,
                    ):

        block = CPMobileBlock(in_channels,
                              out_channels,
                              expansion_rate,
                              stride
                              )
        return block

    def _forward_conv(self, x):
        x = self.in_c(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        global first_RUN
        x = self.quant(x)
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        logits = x.squeeze(2).squeeze(2)
        logits = self.dequant(logits)
        return logits

    def fuse_model(self):
        for m in self.named_modules():
            module_name = m[0]
            module_instance = m[1]
            if module_name == 'in_c':
                in_conv_1 = module_instance[0]
                in_conv_2 = module_instance[1]
                torch.quantization.fuse_modules(in_conv_1, ['0', '1', '2'], inplace=True)
                torch.quantization.fuse_modules(in_conv_2, ['0', '1', '2'], inplace=True)
            elif isinstance(module_instance, CPMobileBlock):
                exp_conv = module_instance.block[0]
                depth_conv = module_instance.block[1]
                proj_conv = module_instance.block[2]
                torch.quantization.fuse_modules(exp_conv, ['0', '1', '2'], inplace=True)
                torch.quantization.fuse_modules(depth_conv, ['0', '1', '2'], inplace=True)
                torch.quantization.fuse_modules(proj_conv, ['0', '1'], inplace=True)
            elif module_name == "feed_forward":
                torch.quantization.fuse_modules(module_instance, ['0', '1'], inplace=True)


def get_model(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=2.3, expansion_rate=3.0,
              n_blocks=(3, 2, 1), strides=None):
    """
    @param n_classes: number of the classes to predict
    @param in_channels: input channels to the network, for audio it is by default 1
    @param base_channels: number of channels after in_conv
    @param channels_multiplier: controls the increase in the width of the network after each stage
    @param expansion_rate: determines the expansion rate in inverted bottleneck blocks
    @param n_blocks: number of blocks that should exist in each stage
    @param strides: default value set below
    @return: full neural network model based on the specified configs
    """

    if strides is None:
        strides = dict(
            b2=(2, 2),
            b4=(2, 1)
        )

    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "n_blocks": n_blocks,
        "strides": strides
    }

    m = Network(model_config)
    print(m)
    return m
