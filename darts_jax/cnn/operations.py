from typing import List, Callable, Optional, Union
from copy import copy

# import torch

from jfi import jaxm, make_random_key as mrk

import equinox as eqx
from equinox import nn
from equinox.experimental import BatchNorm

from jax import Array

batch_wrapper = lambda x: jaxm.vmap(x)

class ReLU(eqx.Module):
    def __call__(self, x, *args, **kw):
        return jaxm.maximum(x, 0)

####################################################################################################


class BatchNorm2d(eqx.Module):
    momentum: float = 0.1
    gamma: Array
    beta: Array
    affine: bool
    in_channels: int
    hash: int
    eps: float

    def __init__(self, in_channels, affine=False, momentum=0.1, eps=1e-5):
        self.in_channels = in_channels
        self.momentum = momentum
        self.affine = affine
        self.gamma, self.beta = jaxm.ones(1), jaxm.zeros(1)
        self.hash = id(self)
        self.eps = eps

    def init_state(self):
        running_mean = jaxm.zeros(self.in_channels)
        running_var = jaxm.ones(self.in_channels)
        return {
            f"{self.hash}_running_mean": running_mean,
            f"{self.hash}_running_var": running_var,
        }

    def __call__(self, x, state=None):
        if state is None:
            state = self.init_state()
        if f"{self.hash}_running_mean" not in state or f"{self.hash}_running_var" not in state:
            state = dict(state, **self.init_state())
        running_mean = state[f"{self.hash}_running_mean"]
        running_var = state[f"{self.hash}_running_var"]

        axis = tuple(-i for i in range(1, x.ndim + 1) if i != -3)
        mean = jaxm.mean(x, axis=axis)
        var = jaxm.var(x, axis=axis)
        running_mean = (1 - self.momentum) * running_mean + self.momentum * mean
        running_var = (1 - self.momentum) * running_var + self.momentum * var
        state[f"{self.hash}_running_mean"] = running_mean
        state[f"{self.hash}_running_var"] = running_var

        mean, var = running_mean[:, None, None], running_var[:, None, None]
        if self.affine:
            x = (x - mean) / jaxm.sqrt(var + self.eps) * self.gamma + self.beta
        else:
            x = (x - mean) / jaxm.sqrt(var + self.eps)
        return x, state


####################################################################################################


class Sequential(eqx.Module):
    mod_list: List[eqx.Module]

    def __init__(self, mod_list: List[eqx.Module]):
        self.mod_list = mod_list

    def init_state(self):
        state = dict()
        for mod in self.mod_list:
            if hasattr(mod, "init_state"):
                state = dict(state, **mod.init_state())
        return state

    def __call__(self, x, state=None):
        total_state = dict() if state is None else copy(state)
        for mod in self.mod_list:
            out = mod(x, state=total_state)
            if isinstance(out, tuple):
                x, state = out
                total_state = dict(total_state, **state)
            else:
                x = out
        return x, total_state


class Conv2d(eqx.Module):
    op: nn.Conv2d

    def __init__(self, *args, **kw):
        self.op = nn.Conv2d(*args, **kw)

    def __call__(self, x, state=None):
        return batch_wrapper(self.op)(x)


class AdaptiveAvgPool2d(eqx.Module):
    op: nn.AdaptiveAvgPool2d

    def __init__(self, *args, **kw):
        self.op = nn.AdaptiveAvgPool2d(*args, **kw)

    def __call__(self, x, state=None):
        return batch_wrapper(self.op)(x)


class Linear(eqx.Module):
    op: nn.Linear

    def __init__(self, *args, **kw):
        self.op = nn.Linear(*args, **kw)

    def __call__(self, x, state=None):
        return batch_wrapper(self.op)(x)


class MaxPool2d(eqx.Module):
    op: nn.MaxPool2d

    def __init__(self, *args, **kw):
        self.op = nn.MaxPool2d(*args, **kw)

    def __call__(self, x, state=None):
        return batch_wrapper(self.op)(x)


class AvgPool2d(eqx.Module):
    op: nn.AvgPool2d

    def __init__(self, *args, **kw):
        self.op = nn.AvgPool2d(*args, **kw)

    def __call__(self, x, state=None):
        return batch_wrapper(self.op)(x)


####################################################################################################


class ReLUConvBN(eqx.Module):
    op: Sequential

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        self.op = Sequential(
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
    op: Sequential

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        self.op = Sequential(
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
    op: Sequential

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        self.op = Sequential(
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

    def __call__(self, x, state=None):
        x = self.relu(x)
        out = jaxm.cat([self.conv_1(x), self.conv_2(x[..., :, 1:, 1:])], axis=-3)
        out, state = self.bn(out, state=state)
        return out, state


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
    "conv_7x1_1x7": lambda C, stride, affine: Sequential(
        [
            ReLU(),
            Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), use_bias=False, key=mrk()),
            Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), use_bias=False, key=mrk()),
            BatchNorm(C, affine=affine),
        ]
    ),
}
