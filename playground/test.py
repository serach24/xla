from timeit import timeit, default_timer
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--det", action="store_true")
args = parser.parse_args()

if args.det:
    print("Running with deterministic ops")
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
from jax import jit, lax, vmap
import jax.numpy as jnp


def scatter_add(
    operand,  # [operand_size]
    updates,  # [updates_size]
    indices,  # [updates_size, 1]
):
    # Define dimension numbers
    update_window_dims = tuple()
    inserted_window_dims = (0,)
    scatter_dims_to_operand_dims = (0,)
    res = jax.lax.scatter_add(
        operand,
        indices,
        updates,
        dimension_numbers=jax.lax.ScatterDimensionNumbers(update_window_dims, inserted_window_dims, scatter_dims_to_operand_dims),
        mode="drop",
    )
    return res

scatter_add_jit = jax.jit(scatter_add)


operand_size = 1000
indices_size = 1_000_000
# indices_size = operand_size * 4 
# operand_size = 64 * 64  # e.g. a 64x64 image
rng = jax.random.PRNGKey(0)

operand = jnp.zeros((operand_size,))
updates = jnp.ones((indices_size))
indices = jax.random.randint(rng, shape=(indices_size, 1), minval=0, maxval=operand_size)

# scatter_add_jit(operand, updates, indices).block_until_ready()

starting_time = default_timer()
scatter_add_jit(operand, updates, indices).block_until_ready()
# time = timeit(scatter_add_jit(operand, updates, indices).block_until_ready())
time = default_timer() - starting_time
print(f"{time} s")

# # For f which outputs a single array, this simulates vmap using Python map
# pymap = lambda f: lambda *args: jnp.stack(list(map(f, *args)))

# operands = jnp.ones((100, 32))
# updates = jnp.ones((100, 2))
# starts = jnp.ones((100, 1), dtype='int32')

# f = lax.dynamic_update_slice

# f_vmapped = jit(vmap(f))
# f_pymapped = jit(pymap(f))

# # Ensure compiled
# f_vmapped(operands, updates, starts)
# f_pymapped(operands, updates, starts)

# t_vmapped = timeit(
#     lambda: f_vmapped(operands, updates, starts).block_until_ready(), number=100
# ) / 100

# t_pymapped = timeit(
#     lambda: f_pymapped(operands, updates, starts).block_until_ready(), number=100
# ) / 100

# print(f"Time vmap(f): {t_vmapped:.2}s")
# print(f"Time pymap(f): {t_pymapped:.2}s")