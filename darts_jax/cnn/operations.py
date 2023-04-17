from typing import List, Callable, Optional, Union
from functools import partial
from copy import copy

from jfi import jaxm
import haiku as hk
from jax import Array, lax


# class BatchNorm2d(hk.Module):
#    def __init__(self, in_channels, affine=False, name=None):
#        super().__init__(name=name)
#        axis = tuple(-i for i in range(1, 4 + 1) if -i != -3)
#        self.op = hk.BatchNorm(
#            create_offset=affine, create_scale=affine, decay_rate=0.1, axis=axis, data_format="NCHW"
#        )
#        self.training = hk.get_state("training", (), bool, init=jaxm.ones)
#        # hk.set_state("training", jaxm.ones((), dtype=bool))
#
#        #@partial(jaxm.jit, static_argnums=1)
#        #def call(x, training):
#        #    return self.op(x, is_training=training)
#
#        #self.call = call
#
#    def __call__(self, x):
#        training = hk.get_state("training")
#        try:
#            return hk.cond(training, lambda: self.op(x, is_training=True), lambda: self.op(x, is_training=False))
#        except ValueError:
#            return self.op(x, is_training=True)
#        #return self.call(x, training)
#        #return hk.cond(training, lambda: self.op(x, True), lambda: self.op(x, False))


class BatchNorm2d(hk.Module):
    def __init__(self, in_channels, affine=False, momentum=0.1, eps=1e-5, name=None):
        super().__init__(name=name)
        self.in_channels = in_channels
        self.momentum = momentum
        self.affine = affine
        self.gamma = hk.get_parameter("gamma", (), init=jaxm.numpy.ones)
        self.beta = hk.get_parameter("beta", (), init=jaxm.numpy.zeros)
        self.eps = eps
        self.running_mean = hk.get_state(
            "running_mean", (in_channels,), jaxm.float32, init=jaxm.numpy.zeros
        )
        self.running_var = hk.get_state(
            "running_var", (in_channels,), jaxm.float32, init=jaxm.numpy.ones
        )
        self.training = hk.get_state("training", (), jaxm.float32, init=jaxm.numpy.ones)

    def __call__(self, x):
        running_mean = hk.get_state("running_mean")
        running_var = hk.get_state("running_var")
        training = hk.get_state("training")

        axis = tuple(-i for i in range(1, x.ndim + 1) if i != -3)
        mean = jaxm.mean(x, axis=axis)
        var = jaxm.var(x, axis=axis)

        momentum = training * self.momentum
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var

        hk.set_state("running_mean", running_mean)
        hk.set_state("running_var", running_var)

        gamma = hk.get_parameter("gamma", (), init=jaxm.numpy.ones)
        beta = hk.get_parameter("beta", (), init=jaxm.numpy.zeros)
        mean, var = running_mean[:, None, None], running_var[:, None, None]
        if self.affine:
            x = (x - mean) / jaxm.sqrt(var + self.eps) * gamma + beta
        else:
            x = (x - mean) / jaxm.sqrt(var + self.eps)
        return x


####################################################################################################


class ReLU(hk.Module):
    def __init__(self, name=None):
        assert name is not None
        super().__init__(name=name)

    def __call__(self, x):
        return jaxm.maximum(x, 0)


class AdaptiveAvgPool2d(hk.Module):
    def __init__(self, size: int, device=None, name=None):
        assert name is not None
        super().__init__(name=name)
        assert size == 1 or size == (1, 1)

    def __call__(self, x):
        return jaxm.mean(x, axis=(-1, -2), keepdims=True)


class Conv2D(hk.Module):
    def __init__(self, *args, name=None, **kw):
        assert name is not None
        super().__init__(name=name)
        if "device" in kw:
            del kw["device"]
        self.op = hk.Conv2D(*args, **kw, data_format="NCHW")

    def __call__(self, x):
        return self.op(x)


class Linear(hk.Module):
    def __init__(self, output_size, name=None):
        assert name is not None
        super().__init__(name=name)
        self.op = hk.Linear(output_size)

    def __call__(self, x):
        return self.op(x)


class MaxPool2d(hk.Module):
    def __init__(self, *args, name=None, **kw):
        assert name is not None
        super().__init__(name=name)
        if "stride" in kw:
            kw["strides"] = kw.pop("stride")
        if "device" in kw:
            del kw["device"]
        self.op = hk.MaxPool(*args, **kw, channel_axis=-3)

    def __call__(self, x):
        return self.op(x)


# class AvgPool2d(hk.Module):
#    def __init__(self, *args, name=None, **kw):
#        assert name is not None
#        super().__init__(name=name)
#        self.device = kw.get("device")
#        if "stride" in kw:
#            kw["strides"] = kw.pop("stride")
#        # self.op = hk.AvgPool(*args, **kw, channel_axis=-3)
#        kernel_shape = kw.get("kernel_shape", args[0])
#        self.kernel_shape = (
#            (kernel_shape, kernel_shape) if isinstance(kernel_shape, int) else kernel_shape
#        )
#        stride = kw.get("strides")
#        if stride is None:
#            stride = args[1]
#        self.stride = (stride, stride) if isinstance(stride, int) else stride
#        padding = kw.get("padding")
#        if padding is None:
#            padding = args[2]
#        self.padding = padding
#        self.kernel = jaxm.ones(self.kernel_shape, device=self.device) / (
#            self.kernel_shape[0] * self.kernel_shape[1]
#        )
#
#    def __call__(self, x):
#        kernel = jaxm.tile(self.kernel, (x.shape[-3], x.shape[-3], 1, 1))
#        kernel = jaxm.to(kernel, dtype=x.dtype, device=self.device)
#        return lax.conv(x, kernel, self.stride, self.padding)


class AvgPool2d(hk.Module):
    def __init__(self, *args, name=None, **kw):
        assert name is not None
        super().__init__(name=name)
        self.device = kw.get("device")
        if "stride" in kw:
            kw["strides"] = kw.pop("stride")
        kernel_shape = kw.get("kernel_shape", args[0])
        self.kernel_shape = (
            (kernel_shape, kernel_shape) if isinstance(kernel_shape, int) else kernel_shape
        )
        stride = kw.get("strides")
        if stride is None:
            stride = args[1]
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        padding = kw.get("padding")
        if padding is None:
            padding = args[2]
        self.padding = padding

    def __call__(self, x):
        return hk.avg_pool(x, self.kernel_shape, self.stride, self.padding, channel_axis=-3)


####################################################################################################


class ReLUConvBN(hk.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, affine=True, device=None, name=None
    ):
        assert name is not None
        super().__init__(name=name)
        padding = (padding, padding) if isinstance(padding, int) else padding
        self.op = hk.Sequential(
            [
                ReLU(name=f"{self.name}__relu"),
                Conv2D(
                    C_out,
                    kernel_shape=kernel_size,
                    stride=stride,
                    padding=padding,
                    with_bias=False,
                    device=device,
                    name=f"{self.name}__conv",
                ),
                BatchNorm2d(C_out, affine=affine, name=f"{self.name}__bn"),
            ]
        )

    def __call__(self, x, *args, **kw):
        return self.op(x, *args, **kw)


class DilConv(hk.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine=True,
        device=None,
        name=None,
    ):
        assert name is not None
        super().__init__(name=name)
        padding = (padding, padding) if isinstance(padding, int) else padding
        self.op = hk.Sequential(
            [
                ReLU(name=f"{self.name}__relu"),
                Conv2D(
                    C_out,
                    kernel_shape=kernel_size,
                    stride=stride,
                    padding=padding,
                    rate=dilation,
                    feature_group_count=C_in,
                    with_bias=False,
                    device=device,
                    name=f"{self.name}__conv1",
                ),
                Conv2D(
                    C_out,
                    kernel_shape=1,
                    padding=(0, 0),
                    with_bias=False,
                    device=device,
                    name=f"{self.name}__conv2",
                ),
                BatchNorm2d(C_out, affine=affine, name=f"{self.name}__bn"),
            ]
        )

    def __call__(self, x, *args, **kw):
        return self.op(x, *args, **kw)


class SepConv(hk.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, affine=True, device=None, name=None
    ):
        assert name is not None
        super().__init__(name=name)
        padding = (padding, padding) if isinstance(padding, int) else padding
        self.op = hk.Sequential(
            [
                ReLU(name=f"{self.name}__relu1"),
                Conv2D(
                    C_in,
                    kernel_shape=kernel_size,
                    stride=stride,
                    padding=padding,
                    feature_group_count=C_in,
                    with_bias=False,
                    device=device,
                    name=f"{self.name}__conv1",
                ),
                Conv2D(
                    C_in,
                    kernel_shape=1,
                    padding=(0, 0),
                    with_bias=False,
                    device=device,
                    name=f"{self.name}__conv2",
                ),
                BatchNorm2d(C_in, affine=affine, name=f"{self.name}__bn1"),
                ReLU(name=f"{self.name}__relu2"),
                Conv2D(
                    C_in,
                    kernel_shape=kernel_size,
                    stride=1,
                    padding=padding,
                    feature_group_count=C_in,
                    with_bias=False,
                    device=device,
                    name=f"{self.name}__conv3",
                ),
                Conv2D(
                    C_out,
                    kernel_shape=1,
                    padding=(0, 0),
                    with_bias=False,
                    device=device,
                    name=f"{self.name}__conv4",
                ),
                BatchNorm2d(C_out, affine=affine, name=f"{self.name}__bn2"),
            ]
        )

    def __call__(self, x, *args, **kw):
        return self.op(x, *args, **kw)


class Identity(hk.Module):
    def __init__(self, *args, name=None, **kw):
        assert name is not None
        super().__init__(name=name)

    def __call__(self, x, *args, **kw):
        return x


class Zero(hk.Module):
    def __init__(self, stride, device=None, name=None):
        assert name is not None
        super().__init__(name=name)
        self.stride = stride

    def __call__(self, x):
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]
        if stride == 1:
            return 0 * x
        return 0 * x[..., ::stride, ::stride]


class FactorizedReduce(hk.Module):
    def __init__(self, C_in, C_out, affine=True, device=None, name=None):
        assert name is not None
        super().__init__(name=name)
        assert C_out % 2 == 0
        self.relu = ReLU(name=f"{self.name}__relu")
        self.conv_1 = Conv2D(
            C_out // 2,
            1,
            stride=2,
            padding=(0, 0),
            with_bias=False,
            device=device,
            name=f"{self.name}__conv1",
        )
        self.conv_2 = Conv2D(
            C_out // 2,
            1,
            stride=2,
            padding=(0, 0),
            with_bias=False,
            device=device,
            name=f"{self.name}__conv2",
        )
        self.bn = BatchNorm2d(C_out, affine=affine, name=f"{self.name}__bn")

    def __call__(self, x):
        x = self.relu(x)
        out = jaxm.cat([self.conv_1(x), self.conv_2(x[..., :, 1:, 1:])], axis=-3)
        out = self.bn(out)
        return out


OPS = {
    "none": lambda C, stride, affine, device=None, name=None: Zero(stride, name=name),
    "avg_pool_3x3": (
        lambda C, stride, affine, device=None, name=None: AvgPool2d(
            3, stride=stride, padding="SAME", device=device, name=name
        )
    ),
    "max_pool_3x3": (
        lambda C, stride, affine, device=None, name=None: MaxPool2d(
            3, stride=stride, padding="SAME", device=device, name=name
        )
    ),
    "skip_connect": (
        lambda C, stride, affine, device=None, name=None: (
            Identity(name=name)
            if stride == 1
            else FactorizedReduce(C, C, affine=affine, device=device, name=name)
        )
    ),
    "sep_conv_3x3": (
        lambda C, stride, affine, device=None, name=None: SepConv(
            C, C, 3, stride, 1, affine=affine, device=device, name=name
        )
    ),
    "sep_conv_5x5": (
        lambda C, stride, affine, device=None, name=None: SepConv(
            C, C, 5, stride, 2, affine=affine, device=device, name=name
        )
    ),
    "sep_conv_7x7": (
        lambda C, stride, affine, device=None, name=None: SepConv(
            C, C, 7, stride, 3, affine=affine, device=device, name=name
        )
    ),
    "dil_conv_3x3": (
        lambda C, stride, affine, device=None, name=None: DilConv(
            C, C, 3, stride, 2, 2, affine=affine, device=device, name=name
        )
    ),
    "dil_conv_5x5": (
        lambda C, stride, affine, device=None, name=None: DilConv(
            C, C, 5, stride, 4, 2, affine=affine, device=device, name=name
        )
    ),
    "conv_7x1_1x7": (
        lambda C, stride, affine, device=None, name=None: hk.Sequential(
            [
                ReLU(name=f"{name}__relu1"),
                Conv2D(
                    C,
                    (1, 7),
                    stride=(1, stride),
                    padding=(0, 3),
                    with_bias=False,
                    device=device,
                    name=f"{name}__conv1",
                ),
                Conv2D(
                    C,
                    (7, 1),
                    stride=(stride, 1),
                    padding=(3, 0),
                    with_bias=False,
                    device=device,
                    name=f"{name}__conv2",
                ),
                BatchNorm2d(C, affine=affine, name=f"{name}__bn1"),
            ]
        )
    ),
}
