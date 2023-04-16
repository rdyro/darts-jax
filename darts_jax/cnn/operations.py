from typing import List, Callable, Optional, Union, Tuple
from copy import copy

# import torch

from jfi import jaxm, make_random_key as mrk
from flax import linen as nn

from jax import Array

batch_wrapper = lambda x: jaxm.vmap(x)


class ReLU(nn.Module):
    def __call__(self, x):
        return jaxm.maximum(x, 0)


####################################################################################################


class BatchNorm2d(nn.Module):
    in_channels: int
    affine: bool = False

    def setup(self):
        self.op = nn.BatchNorm(use_running_average=True, axis=-3)

    def __call__(self, x):
        return self.op(x)


####################################################################################################


class Conv2d(nn.Module):
    C_in: int
    C_out: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]] = 1
    padding: Union[int, Tuple[int, int], str] = "SAME"
    dilation: int = 1
    groups: int = 1
    use_bias: bool = True

    def setup(self):
        kernel_size = (
            (self.kernel_size,) * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        )
        padding = (self.padding,) * 2 if isinstance(self.padding, int) else self.padding
        stride = (self.stride,) * 2 if isinstance(self.stride, int) else self.stride

        self.op = nn.Conv(
            self.C_out,
            kernel_size,
            stride,
            padding,
            kernel_dilation=self.dilation,
            feature_group_count=self.groups,
            use_bias=self.use_bias,
        )

    def __call__(self, x):
        return self.op(jaxm.swapaxes(x, -1, -3)).swapaxes(-1, -3)


class AdaptiveAvgPool2d(nn.Module):
    output_size: Union[int, Tuple[int, int]]

    def setup(self):
        assert self.output_size == 1 or self.output_size == (1, 1)

    def __call__(self, x):
        return jaxm.mean(x, (-1, -2), keepdims=True)


class Linear(nn.Module):
    out_features: int

    def setup(self):
        self.op = nn.Dense(self.out_features)

    def __call__(self, x):
        return self.op(x)


class MaxPool2d(nn.Module):
    window: int
    stride: int
    padding: str

    def __call__(self, x):
        x = jaxm.swapaxes(x, -3, -1)
        x = nn.max_pool(x, (self.window, self.window), (self.stride, self.stride), self.padding)
        return jaxm.swapaxes(x, -3, -1)


class AvgPool2d(nn.Module):
    window: int
    stride: int
    padding: str

    def __call__(self, x):
        x = jaxm.swapaxes(x, -3, -1)
        x = nn.avg_pool(x, (self.window, self.window), (self.stride, self.stride), self.padding)
        return jaxm.swapaxes(x, -3, -1)


####################################################################################################


class ReLUConvBN(nn.Module):
    C_in: int
    C_out: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]]
    padding: Union[int, Tuple[int, int]]
    affine: bool = True

    def setup(self):
        self.op = nn.Sequential(
            [
                ReLU(),
                Conv2d(
                    self.C_in,
                    self.C_out,
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    use_bias=False,
                ),
                BatchNorm2d(self.C_out, affine=self.affine),
            ]
        )

    def __call__(self, x):
        return self.op(x)


class DilConv(nn.Module):
    C_in: int
    C_out: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]]
    padding: Union[int, Tuple[int, int]]
    dilation: Union[int, Tuple[int, int]]
    affine: bool = True

    def setup(self):
        self.op = nn.Sequential(
            [
                ReLU(),
                Conv2d(
                    self.C_in,
                    self.C_in,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.C_in,
                    use_bias=False,
                ),
                Conv2d(self.C_in, self.C_out, kernel_size=1, padding=0, use_bias=False),
                BatchNorm2d(self.C_out, affine=self.affine),
            ]
        )

    def __call__(self, x):
        return self.op(x)


class SepConv(nn.Module):
    C_in: int
    C_out: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]]
    padding: Union[int, Tuple[int, int]]
    affine: bool = True

    def setup(self):
        self.op = nn.Sequential(
            [
                ReLU(),
                Conv2d(
                    self.C_in,
                    self.C_in,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    groups=self.C_in,
                    use_bias=False,
                ),
                Conv2d(self.C_in, self.C_in, kernel_size=1, padding=0, use_bias=False),
                BatchNorm2d(self.C_in, affine=self.affine),
                ReLU(),
                Conv2d(
                    self.C_in,
                    self.C_in,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.padding,
                    groups=self.C_in,
                    use_bias=False,
                ),
                Conv2d(self.C_in, self.C_out, kernel_size=1, padding=0, use_bias=False),
                BatchNorm2d(self.C_out, affine=self.affine),
            ]
        )

    def __call__(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __call__(self, x):
        return x


class Zero(nn.Module):
    stride: int

    def __call__(self, x):
        if self.stride == 1:
            return 0 * x
        return 0 * x[..., :: self.stride, :: self.stride]


class FactorizedReduce(nn.Module):
    C_in: int
    C_out: int
    affine: bool = True

    def setup(self):
        assert self.C_out % 2 == 0
        self.relu = ReLU()
        self.conv_1 = Conv2d(self.C_in, self.C_out // 2, 1, stride=2, padding=0, use_bias=False)
        self.conv_2 = Conv2d(self.C_in, self.C_out // 2, 1, stride=2, padding=0, use_bias=False)
        self.bn = BatchNorm2d(self.C_out, affine=self.affine)

    def __call__(self, x):
        x = self.relu(x)
        x = jaxm.cat([self.conv_1(x), self.conv_2(x[..., :, 1:, 1:])], axis=-3)
        x = self.bn(x)
        return x


OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: AvgPool2d(3, stride=stride, padding="SAME"),
    "max_pool_3x3": lambda C, stride, affine: MaxPool2d(3, stride=stride, padding="SAME"),
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
            Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), use_bias=False),
            Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), use_bias=False),
            BatchNorm2d(C, affine=affine),
        ]
    ),
}
