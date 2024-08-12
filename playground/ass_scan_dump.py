import jax
import jax.numpy as jp

from timeit import timeit, default_timer
XLA_FLAGS="--xla_gpu_dump_llvmir --xla_dump_to=/tmp/jax_output --xla_gpu_dump_ptx --xla_gpu_dump_optimized_hlo"

def add_segment(iv, jt):
  i, v = iv
  j, t = jt
  return j, v * jp.equal(i, j) + t
  
@jax.jit
def ass_scan(fn, indices, sorted):
  return jax.lax.associative_scan(add_segment, (indices, sorted))

# operand_size = 1000
# indices_size = 1_000_000


operand_size = 4
indices_size = 4
# indices_size = operand_size * 4 
# operand_size = 64 * 64  # e.g. a 64x64 image
# rng = jax.random.PRNGKey(0)

# operand = jp.zeros((operand_size,))
# updates = jp.ones((indices_size))
# indices = jax.random.randint(rng, shape=(indices_size, 1), minval=0, maxval=operand_size)

# # Obtain the HLO representation
# indices = jp.reshape(indices, updates.shape)
# indices, sorted = jax.lax.sort_key_val(indices, updates, dimension=-1)
# print(f"{indices=}")
# print(f"{sorted=}")
indices = jp.array([0,1,1,2])
sorted = jp.array([1.,1.,1.,1.], dtype=jp.float32)

res = ass_scan(jax.tree_util.Partial(add_segment), indices, sorted)

print(res)
# hlo = jax.xla_computation(ass_scan)(jax.tree_util.Partial(add_segment), indices, sorted)
# hlo_str = hlo.as_hlo_text()
# print(hlo_str)
exit()


# Warm up 
scatter_add_det(operand, updates, indices).block_until_ready()

starting_time = default_timer()
scatter_add_det(operand, updates, indices).block_until_ready()
# time = timeit(scatter_add_jit(operand, updates, indices).block_until_ready())
time = default_timer() - starting_time
print(f"{time} s")