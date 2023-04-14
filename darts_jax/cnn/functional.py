from jfi import jaxm

import numpy as np

import equinox as eqx
from .model_search import Network
from jax import tree_util as tu


def make_functional(net: Network):
    params, others = eqx.partition(net, eqx.is_array)
    p_ids = [id(net.alphas_normal), id(net.alphas_reduce)]

    z_params, p_params = eqx.partition(params, lambda x: id(x) not in p_ids)

    z_flat, z_struct = tu.tree_flatten(z_params)
    p_flat, p_struct = tu.tree_flatten(p_params)
    z_shapes = [z.shape for z in z_flat]
    z_sizes = [z.size for z in z_flat]
    p_shapes = [p.shape for p in p_flat]
    p_sizes = [p.size for p in p_flat]

    def forward(z, p, *args, **kw):
        zs = jaxm.split(z, np.cumsum(z_sizes)[:-1])
        ps = jaxm.split(p, np.cumsum(p_sizes)[:-1])
        zs = [z.reshape(z_shape) for z, z_shape in zip(zs, z_shapes)]
        ps = [p.reshape(p_shape) for p, p_shape in zip(ps, p_shapes)]
        params = eqx.combine(tu.tree_unflatten(z_struct, zs), tu.tree_unflatten(p_struct, ps))
        net = eqx.combine(params, others)
        return net(*args, **kw)

    z = jaxm.cat([z.reshape(-1) for z in z_flat])
    p = jaxm.cat([p.reshape(-1) for p in p_flat])

    return forward, (z, p)
