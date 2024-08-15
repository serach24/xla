from timeit import timeit, default_timer
import argparse
import os
import json
import sys
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--det", action="store_true")
parser.add_argument("--jax-scan", action="store_true")
args = parser.parse_args()

if args.det:
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["XLA_FLAGS"] += f' --xla_dump_to={cur_dir}/tmp{"_jax-scan" if args.jax_scan else "_det" if args.det else ""} --xla_dump_hlo_as_text'

import jax
import jax.numpy as jnp


@jax.jit
def scatter_add(
    operand,  # [operand_size]
    indices,  # [updates_size, 1]
    updates,  # [updates_size]
):
    res = jax.lax.scatter_add(
        operand,
        indices,
        updates,
        dimension_numbers=jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)
        ),
        mode="drop",
    )
    return res

@jax.jit
def scatter_add_jax_associative_scan(
    operand,  # [operand_size]
    indices,  # [updates_size, 1]
    updates,  # [updates_size]
):
    def add_segment(iv, jt):
        i, v = iv
        j, t = jt
        return j, v * jnp.equal(i, j) + t

    indices = jnp.reshape(indices, updates.shape)
    indices, sorted = jax.lax.sort_key_val(indices, updates, dimension=-1)
    
    _, sums = jax.lax.associative_scan(add_segment, (indices, sorted))
    end_of_run = jnp.concatenate([jnp.not_equal(indices[1:], indices[:-1]), jnp.array([True])])
    indices = jnp.where(end_of_run, indices, operand.shape[-1])
    return operand.at[indices].add(sums, mode='drop', unique_indices=True)

def main(args):
    if args.det:
        print("Running with deterministic ops")
        exp_name = "det"
    else:
        print("Running with non-deterministic (reference) ops")
        exp_name = "non_det"
    if args.jax_scan:
        print("Running with JAX scan")
    
    op = scatter_add_jax_associative_scan if args.jax_scan else scatter_add
    input_size = 2
    index_size = 3
    rng_key = jax.random.PRNGKey(0)
    data = jnp.zeros(input_size, dtype=jnp.float32)
    indices = jax.random.randint(rng_key, (index_size, 1), minval=0, maxval=input_size)
    print(f"{indices=}")
    updates = jax.random.uniform(rng_key, (index_size,), dtype=jnp.float32)
    print(f"{updates=}")
    output = op(data, indices, updates)
    print(f"{output}")



if __name__ == "__main__":
    main(args)
