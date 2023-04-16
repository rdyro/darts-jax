import time
from typing import List, Callable, Optional, Union
from copy import copy

# import torch
# from torch.autograd import Variable
from .operations import *
from .genotypes import PRIMITIVES
from .genotypes import Genotype

import numpy as np

from jfi import jaxm, make_random_key as mrk
from flax import linen as nn
import flax
from jax.tree_util import tree_map

Array = jaxm.jax.Array


class MixedOp(nn.Module):
    C: int
    stride: int

    def setup(self):
        _ops = []
        for primitive in PRIMITIVES:
            op = OPS[primitive](self.C, self.stride, False)
            if "pool" in primitive:
                op = nn.Sequential([op, BatchNorm2d(self.C, affine=False)])
            _ops.append(op)
        self._ops = _ops

    def __call__(self, x, weights):
        # return sum(w * op(x) for w, op in zip(weights, self._ops))
        assert weights is not None
        xs = [op(x) for op in self._ops]
        x = sum(w * x for (w, x) in zip(weights, xs))
        return x


class Cell(nn.Module):
    steps: int
    multiplier: int
    C_prev_prev: int
    C_prev: int
    C: int
    reduction: bool
    reduction_prev: bool

    def setup(self):
        if self.reduction_prev:
            self.preprocess0 = FactorizedReduce(self.C_prev_prev, self.C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(self.C_prev_prev, self.C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(self.C_prev, self.C, 1, 1, 0, affine=False)

        _ops = []
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if self.reduction and j < 2 else 1
                op = MixedOp(self.C, stride)
                _ops.append(op)
        self._ops = _ops

    def __call__(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        outputs = [s0, s1]
        offset = 0
        for i in range(self.steps):
            ss = [
                self._ops[offset + j](h, weights=weights[offset + j]) for j, h in enumerate(outputs)
            ]
            s = sum(ss)
            offset += len(outputs)
            outputs.append(s)

        x = jaxm.cat(outputs[-self.multiplier :], axis=-3)
        assert x.shape[-3] == self.C * self.multiplier
        return x


class Network(nn.Module):
    C: int
    num_classes: int
    layers: int
    steps: int = 4
    multiplier: int = 4
    stem_multiplier: int = 3

    def setup(self):
        C_curr = self.stem_multiplier * self.C
        self.stem = nn.Sequential(
            [Conv2d(3, C_curr, 3, padding=1, use_bias=False), BatchNorm2d(C_curr)]
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        cells = []
        reduction_prev = False
        for i in range(self.layers):
            if i in [self.layers // 3, 2 * self.layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                self.steps, self.multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev
            )
            reduction_prev = reduction
            cells.append(cell)
            C_prev_prev, C_prev = C_prev, self.multiplier * C_curr
        self.cells = cells

        self.global_pooling = AdaptiveAvgPool2d(1)
        self.classifier = Linear(self.num_classes)

        self._initialize_alphas()

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

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self.steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        # self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        # self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_normal = 1e-3 * jaxm.randn((k, num_ops))
        self.alphas_reduce = 1e-3 * jaxm.randn((k, num_ops))
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
