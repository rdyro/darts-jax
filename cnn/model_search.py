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
import equinox as eqx
from equinox import nn

Array = jaxm.jax.Array


class MixedOp(eqx.Module):
    _ops: List[eqx.Module]

    def __init__(self, C, stride):
        self._ops = []
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op = Sequential([op, BatchNorm2d(C, affine=False)])
            self._ops.append(op)

    def __call__(self, x, state=None, weights=None):
        #return sum(w * op(x) for w, op in zip(weights, self._ops))
        assert weights is not None
        outputs = [op(x, state=state) for op in self._ops]
        xs, states = zip(
            *[(out[0], out[1]) if isinstance(out, tuple) else (out[0], dict()) for out in outputs]
        )
        x = sum(w * x for w, x in zip(weights, xs)) 
        state = dict()
        for state_ in states:
            state.update(state_)
        return x, state


class Cell(eqx.Module):
    preprocess0: Union[ReLUConvBN, FactorizedReduce]
    preprocess1: ReLUConvBN
    _steps: int
    _multiplier: int
    _ops: List[eqx.Module]
    reduction: bool
    C: int
    C_prev: int
    C_prev_prev: int

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self.C = C
        self.C_prev = C_prev
        self.C_prev_prev = C_prev_prev

        self._ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def __call__(self, s0, s1, weights, state=None):
        state = dict() if state is None else copy(state)
        s0, state = self.preprocess0(s0, state=state)
        s1, state = self.preprocess1(s1, state=state)

        outputs = [s0, s1]
        offset = 0
        for i in range(self._steps):
            #ss, states = zip(*[self._ops[offset + j](h, weights=weights[offset + j], state=state) for j, h in enumerate(states)])
            rets = [self._ops[offset + j](h, weights=weights[offset + j], state=state) for j, h in enumerate(outputs)]
            ss, states = zip(*[(ret[0], ret[1]) if isinstance(ret, tuple) else (ret[0], dict()) for ret in rets])
            s = sum(ss)
            assert not isinstance(s, tuple)
            for state_ in states:
                state.update(state_)
            offset += len(outputs)
            outputs.append(s)

        x = jaxm.cat(outputs[-self._multiplier :], axis=-3)
        assert x.shape[-3] == self.C * self._multiplier
        return x, state


class Network(eqx.Module):
    _C: int
    _num_classes: int
    _layers: int
    _criterion: Callable
    _steps: int
    _multiplier: int
    cells: List[eqx.Module]
    alphas_normal: Array
    alphas_reduce: Array
    _arch_parameters: List[Array]
    stem: Sequential
    global_pooling: AdaptiveAvgPool2d
    classifier: Linear

    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion: Optional[Callable] = None,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
    ):
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = Sequential(
            [Conv2d(3, C_curr, 3, padding=1, use_bias=False, key=mrk()), BatchNorm2d(C_curr)]
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
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = AdaptiveAvgPool2d(1)
        self.classifier = Linear(C_prev, num_classes, key=mrk())

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def __call__(self, input, state=None):
        state = dict() if state is None else copy(state)
        s0, state = self.stem(input, state=state)
        s1 = s0
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = jaxm.softmax(self.alphas_reduce, axis=-1)
            else:
                weights = jaxm.softmax(self.alphas_normal, axis=-1)
            (s0, (s1, state)) = s1, cell(s0, s1, weights, state=state)
        out = self.global_pooling(s1, state=state)
        logits = self.classifier(out.reshape(out.shape[:-3] + (-1,)), state=state)
        return logits, state

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
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
