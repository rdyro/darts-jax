from typing import List, Callable, Optional, Union
from copy import copy

# import torch

from jfi import jaxm, make_random_key as mrk

import equinox as eqx
from equinox import nn
from equinox.experimental import BatchNorm

from jax import Array

USE_REGULAR_BATCH_VERSION = True

if USE_REGULAR_BATCH_VERSION:
    batch_wrapper = lambda x: x
else:
    batch_wrapper = lambda x: jaxm.vmap(x, axis_name="batch")


class ReLU(eqx.Module):
    def __call__(self, x, *args, **kw):
        return jaxm.maximum(x, 0)

####################################################################################################


class BatchNorm2d(eqx.Module):
    bn: BatchNorm

    def __init__(self, channels, affine=True):
        self.bn = BatchNorm(channels, channelwise_affine=affine, axis_name="batch")

    def __call__(self, x, *args, **kw):
        return self.bn(x)

####################################################################################################

class Conv2d(eqx.Module):
    op: nn.Conv2d

    def __init__(self, *args, **kw):
        self.op = nn.Conv2d(*args, **kw)

    def __call__(self, x, *args, **kw):
        return batch_wrapper(self.op)(x)


class AdaptiveAvgPool2d(eqx.Module):
    op: nn.AdaptiveAvgPool2d

    def __init__(self, *args, **kw):
        self.op = nn.AdaptiveAvgPool2d(*args, **kw)

    def __call__(self, x, *args, **kw):
        return batch_wrapper(self.op)(x)


class Linear(eqx.Module):
    op: nn.Linear

    def __init__(self, *args, **kw):
        self.op = nn.Linear(*args, **kw)

    def __call__(self, x, *args, **kw):
        return batch_wrapper(self.op)(x)


class MaxPool2d(eqx.Module):
    op: nn.MaxPool2d

    def __init__(self, *args, **kw):
        self.op = nn.MaxPool2d(*args, **kw)

    def __call__(self, x, *args, **kw):
        return batch_wrapper(self.op)(x)


class AvgPool2d(eqx.Module):
    op: nn.AvgPool2d

    def __init__(self, *args, **kw):
        self.op = nn.AvgPool2d(*args, **kw)

    def __call__(self, x, *args, **kw):
        return batch_wrapper(self.op)(x)


####################################################################################################


class ReLUConvBN(eqx.Module):
    op: nn.Sequential

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        self.op = nn.Sequential(
            [
                ReLU(),
                Conv2d(
                    C_in,
                    C_out,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    use_bias=False,
                    key=mrk(),
                ),
                BatchNorm2d(C_out, affine=affine),
            ]
        )

    def __call__(self, x, *args, **kw):
        return self.op(x, *args, **kw)


class DilConv(eqx.Module):
    op: nn.Sequential

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        self.op = nn.Sequential(
            [
                ReLU(),
                Conv2d(
                    C_in,
                    C_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=C_in,
                    use_bias=False,
                    key=mrk(),
                ),
                Conv2d(C_in, C_out, kernel_size=1, padding=0, use_bias=False, key=mrk()),
                BatchNorm2d(C_out, affine=affine),
            ]
        )

    def __call__(self, x, *args, **kw):
        return self.op(x, *args, **kw)


class SepConv(eqx.Module):
    op: nn.Sequential

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        self.op = nn.Sequential(
            [
                ReLU(),
                Conv2d(
                    C_in,
                    C_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=C_in,
                    use_bias=False,
                    key=mrk(),
                ),
                Conv2d(C_in, C_in, kernel_size=1, padding=0, use_bias=False, key=mrk()),
                BatchNorm2d(C_in, affine=affine),
                ReLU(),
                Conv2d(
                    C_in,
                    C_in,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    groups=C_in,
                    use_bias=False,
                    key=mrk(),
                ),
                Conv2d(C_in, C_out, kernel_size=1, padding=0, use_bias=False, key=mrk()),
                BatchNorm2d(C_out, affine=affine),
            ]
        )

    def __call__(self, x, *args, **kw):
        return self.op(x, *args, **kw)


class Identity(eqx.Module):
    def __call__(self, x, *args, **kw):
        return x


class Zero(eqx.Module):
    stride: int

    def __init__(self, stride):
        self.stride = stride

    def __call__(self, x, *args, **kw):
        if self.stride == 1:
            return 0 * x
        return 0 * x[..., :: self.stride, :: self.stride]


class FactorizedReduce(eqx.Module):
    relu: ReLU
    conv_1: Conv2d
    conv_2: Conv2d
    bn: BatchNorm2d

    def __init__(self, C_in, C_out, affine=True):
        assert C_out % 2 == 0
        self.relu = ReLU()
        self.conv_1 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, use_bias=False, key=mrk())
        self.conv_2 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, use_bias=False, key=mrk())
        self.bn = BatchNorm2d(C_out, affine=affine)

    def __call__(self, x, *args, **kw):
        x = self.relu(x)
        out = jaxm.cat([self.conv_1(x), self.conv_2(x[..., :, 1:, 1:])], axis=-3)
        out = self.bn(out)
        return out


OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: AvgPool2d(3, stride=stride, padding=1),
    "max_pool_3x3": lambda C, stride, affine: MaxPool2d(3, stride=stride, padding=1),
    "skip_connect": lambda C, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    "conv_7x1_1x7": lambda C, stride, affine: nn.Sequential(
        [
            ReLU(),
            Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), use_bias=False, key=mrk()),
            Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), use_bias=False, key=mrk()),
            BatchNorm(C, affine=affine),
        ]
    ),
}
