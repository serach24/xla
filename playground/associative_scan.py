import jax
import jax.numpy as jp

from timeit import timeit, default_timer


def add_segment(iv, jt):
  i, v = iv
  j, t = jt
  return j, v * jp.equal(i, j) + t
  
@jax.jit
def ass_scan(fn, indices, sorted):
  return jax.lax.associative_scan(add_segment, (indices, sorted))

# @jax.jit
def scatter_add_det(operand, updates, indices):
  indices = jp.reshape(indices, updates.shape)
  # Sort the indices and the values by the indices.
  indices, sorted = jax.lax.sort_key_val(indices, updates, dimension=-1)
  print(f"{indices=}")
  print(f"{sorted=}")
  # Sum up runs of the same index - the sum for each index will be at the end of each run.
  _, sums = ass_scan(add_segment, indices, sorted)
#   _, sums = jax.lax.associative_scan(add_segment, (indices, sorted))
  # Produce an array of bools - if an element is set then the position
  # is the end of run of the same index.
  end_of_run = jp.concatenate([jp.not_equal(indices[1:], indices[:-1]), jp.array([True])])
  # Set all position that are not at end of run to an out-of-bound index.
  indices = jp.where(end_of_run, indices, operand.shape[-1])
  # Now do scatter-add where we know the (in-bounds) indices are unique.
  # That is still fast on GPUs (no non-determinism from atomics).
  return operand.at[indices].add(sums, mode='drop', unique_indices=True)


# operand_size = 1000
# indices_size = 1_000_000


operand_size = 4
indices_size = 4
# indices_size = operand_size * 4 
# operand_size = 64 * 64  # e.g. a 64x64 image
rng = jax.random.PRNGKey(0)

operand = jp.zeros((operand_size,))
updates = jp.ones((indices_size))
indices = jax.random.randint(rng, shape=(indices_size, 1), minval=0, maxval=operand_size)

# Obtain the HLO representation
indices = jp.reshape(indices, updates.shape)
indices, sorted = jax.lax.sort_key_val(indices, updates, dimension=-1)

hlo = jax.xla_computation(ass_scan)(jax.tree_util.Partial(add_segment), indices, sorted)
hlo_str = hlo.as_hlo_text()
print(hlo_str)
exit()


# Warm up 
scatter_add_det(operand, updates, indices).block_until_ready()

starting_time = default_timer()
scatter_add_det(operand, updates, indices).block_until_ready()
# time = timeit(scatter_add_jit(operand, updates, indices).block_until_ready())
time = default_timer() - starting_time
print(f"{time} s")