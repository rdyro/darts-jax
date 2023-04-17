import time
from typing import List, Callable, Optional, Union, Dict
import re
from copy import copy

# import torch
# from torch.autograd import Variable
from .operations import *
from .genotypes import PRIMITIVES
from .genotypes import Genotype

import numpy as np
from jfi import jaxm, make_random_key

import haiku as hk

Array = jaxm.jax.Array


class MixedOp(hk.Module):
    def __init__(self, C, stride, device=None, name=None):
        super().__init__(name=name)
        self._ops = []
        for i, primitive in enumerate(PRIMITIVES):
            op = OPS[primitive](C, stride, False, device=device, name=f"{self.name}__op_{i}")
            if "pool" in primitive:
                op = hk.Sequential(
                    [
                        op,
                        BatchNorm2d(C, affine=False, name=f"{self.name}__op_{i}__bn"),
                    ]
                )
            self._ops.append(op)

    def __call__(self, x, weights):
        # return sum(w * op(x) for w, op in zip(weights, self._ops))
        assert weights is not None
        xs = [op(x) for op in self._ops]
        x = sum(w * x for w, x in zip(weights, xs))
        return x


class Cell(hk.Module):
    def __init__(
        self,
        steps,
        multiplier,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        device=None,
        name=None,
    ):
        assert name is not None
        super().__init__(name=name)
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(
                C_prev_prev, C, affine=False, device=device, name=f"{self.name}__preprocess0"
            )
        else:
            self.preprocess0 = ReLUConvBN(
                C_prev_prev,
                C,
                1,
                1,
                0,
                affine=False,
                device=device,
                name=f"{self.name}__preprocess0",
            )
        self.preprocess1 = ReLUConvBN(
            C_prev, C, 1, 1, 0, affine=False, device=device, name=f"{self.name}__preprocess1"
        )
        self._steps = steps
        self._multiplier = multiplier
        self.C = C
        self.C_prev = C_prev
        self.C_prev_prev = C_prev_prev

        self._ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, device=device, name=f"{self.name}__op_{i}_{j}")
                self._ops.append(op)

    def __call__(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        outputs = [s0, s1]
        offset = 0
        for i in range(self._steps):
            ss = [
                self._ops[offset + j](h, weights=weights[offset + j])
                for j, h in enumerate(outputs)
            ]
            s = sum(ss)
            offset += len(outputs)
            outputs.append(s)

        x = jaxm.cat(outputs[-self._multiplier :], axis=-3)
        assert x.shape[-3] == self.C * self._multiplier
        return x


class Network(hk.Module):
    def __init__(
        self,
        C,
        num_classes,
        layers,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
        device=None,
        name=None,
    ):
        super().__init__(name=name)
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = hk.Sequential(
            [
                Conv2D(
                    C_curr,
                    3,
                    padding=(1, 1),
                    with_bias=False,
                    device=device,
                    name=f"{self.name}__stem__conv",
                ),
                BatchNorm2d(C_curr, name=f"{self.name}__stem__bn"),
            ],
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = []
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                device=device,
                name=f"{self.name}__cell{i}",
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = AdaptiveAvgPool2d(
            1, device=device, name=f"{self.name}__global_pooling"
        )
        self.classifier = Linear(num_classes, name=f"{self.name}__classifier")

        self._initialize_alphas(device)

    def __call__(self, input):
        s0 = self.stem(input)
        s1 = s0
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = jaxm.softmax(self.alphas_reduce, axis=-1)
            else:
                weights = jaxm.softmax(self.alphas_normal, axis=-1)
            (s0, s1) = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.reshape(out.shape[:-3] + (-1,)))
        return logits

    def _initialize_alphas(self, device=None):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        def random_init(shape, dtype):
            return 1e-3 * jaxm.randn(shape, dtype=dtype, device=device)

        self.alphas_normal = hk.get_parameter("alphas_normal", (k, num_ops), init=random_init)
        self.alphas_reduce = hk.get_parameter("alphas_reduce", (k, num_ops), init=random_init)
        # self.alphas_normal = 1e-3 * jaxm.randn((k, num_ops), device=device)
        # self.alphas_reduce = 1e-3 * jaxm.randn((k, num_ops), device=device)
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index("none"):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(np.array(jaxm.softmax(self.alphas_normal, axis=-1)))
        gene_reduce = _parse(np.array(jaxm.softmax(self.alphas_reduce, axis=-1)))

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


####################################################################################################


def make_haiku_network(
    C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3, device=None
):
    example_input = jaxm.randn((2, C, 32, 32), device=device)

    def _fwd(x):
        net = Network(C, num_classes, layers, steps, multiplier, stem_multiplier, device=device)
        return net(x)

    net = hk.transform_with_state(_fwd)
    net_params, state = net.init(make_random_key(), example_input)
    _, state = net.apply(net_params, state, make_random_key(), example_input)

    flat_param_dict = flatten_dict(net_params)
    z_params = {k: v for k, v in flat_param_dict.items() if re.search("alphas_", k) is None}
    p_params = {k: v for k, v in flat_param_dict.items() if re.search("alphas_", k) is not None}
    z_shapes = {k: v.shape for k, v in z_params.items()}
    p_shapes = {k: v.shape for k, v in p_params.items()}
    z_sizes = [v.size for v in z_params.values()]
    p_sizes = [v.size for v in p_params.values()]
    z_splits = np.cumsum(z_sizes)[:-1].tolist()
    p_splits = np.cumsum(p_sizes)[:-1].tolist()

    def params2zp(params: Dict):
        flat_params = flatten_dict(params)
        z = jaxm.cat(
            [v.reshape(-1) for k, v in flat_params.items() if re.search("alphas_", k) is None]
        )
        p = jaxm.cat(
            [v.reshape(-1) for k, v in flat_params.items() if re.search("alphas_", k) is not None]
        )
        return z, p

    def zp2params(z: Array, p: Array):
        ps = jaxm.split(p, p_splits)
        zs = jaxm.split(z, z_splits)
        z_dict = {k: v.reshape(s) for (k, s), v in zip(z_shapes.items(), zs)}
        p_dict = {k: v.reshape(s) for (k, s), v in zip(p_shapes.items(), ps)}
        params = z_dict
        params.update(p_dict)
        params = unflatten_dict(params)
        return params

    random_key = make_random_key()

    def fwd_fn(z, p, x, state):
        params = zp2params(z, p)
        y, state = net.apply(params, state, random_key, x)
        return y, state

    return fwd_fn, net, net_params, state, params2zp, zp2params


def flatten_dict(d: Dict, separator: str = ":") -> Dict:
    flat_d = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            flat_d.update({f"{k}{separator}{k2}": v2 for k2, v2 in flatten_dict(v).items()})
        else:
            flat_d[k] = v
    return flat_d


def unflatten_dict(flat_d: Dict, separator: str = ":") -> Dict:
    d = dict()
    for k, v in flat_d.items():
        if separator in k:
            k1, k2 = k.split(separator, maxsplit=1)
            if k1 not in d:
                d[k1] = dict()
            d[k1].update(unflatten_dict({k2: v}))
        else:
            d[k] = v
    return d
