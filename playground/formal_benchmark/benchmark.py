from timeit import timeit, default_timer
import argparse
import os
import json
import sys
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--det", action="store_true")
args = parser.parse_args()

if args.det:
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

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

def benchmark_scatter(data, indices, updates, repeat=5):
    # Perform a warm-up run to ensure the operation is JIT-compiled
    execution_time = 0
  
    _ = scatter_add(data, indices, updates).block_until_ready()
    
    # Start the timer after the warm-up
    for _ in range(repeat):
        start_time = time.perf_counter()
        output = scatter_add(data, indices, updates).block_until_ready()
        end_time = time.perf_counter()

        execution_time += end_time - start_time
    
    return output, execution_time / repeat


def main(args):
    if args.det:
        print("Running with deterministic ops")
        exp_name = "det"
    else:
        print("Running with non-deterministic (reference) ops")
        exp_name = "non_det"

    sizes = [10, 100, 1000, 10000, 100000, 1_000_000]
    input_sizes = sizes
    index_sizes = sizes

    results = {}
    compare_file = f'reference_output.json'
    timing_results = {}

    seed = 1234
    rng_key = jax.random.PRNGKey(seed)

    try:
        with open(compare_file, 'r') as f:
            reference_data = json.load(f)
    except FileNotFoundError:
        reference_data = {}

    for input_size in input_sizes:
        for index_size in index_sizes:
            key = (input_size, index_size)
            data = jnp.zeros(input_size, dtype=jnp.float32)
            indices = jax.random.randint(rng_key, (index_size, 1), minval=0, maxval=input_size)
            updates = jax.random.uniform(rng_key, (index_size,), dtype=jnp.float32)

            assert data.shape == (input_size,), f"Data shape mismatch: expected ({input_size},)"
            assert indices.shape == (index_size, 1), f"Indices shape mismatch: expected ({index_size}, 1)"
            assert updates.shape == (index_size,), f"Updates shape mismatch: expected ({index_size},)"


            output, execution_time = benchmark_scatter(data, indices, updates)
            timing_results[str(key)] = execution_time
            if args.det and str(key) in reference_data:
                reference = jnp.array(reference_data[str(key)])
                if not jnp.allclose(output, reference, atol=1e-5):
                        print(f"Mismatch between two results.")
                        differences = jnp.where(jnp.abs(output - reference) > 1e-4)
                        diff_output = output[differences]
                        diff_reference = reference[differences]
                        print(f"Mismatch detected for input size {len(data)} and index size {len(indices)}.")
                        print(f"Differences at indices {differences[0]}:")
                        print(f"Output values: {diff_output}")
                        print(f"Reference values: {diff_reference}")
                        sys.exit(1)  # Aborts the process if the output does not match the reference
            results[str(key)] = output.tolist()

    if not args.det and results:
        with open(compare_file, 'w') as f:
            json.dump(results, f)
    else:
        with open("det_output.json", 'w') as f:
            json.dump(results, f)

    csv_file_path = f'benchmark_times_{exp_name}.csv'
    write_header = not os.path.exists(csv_file_path)
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Input Size', 'Index Size', 'Execution Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for key, execution_time in timing_results.items():
            input_size, index_size = key.strip('()').split(', ')
            writer.writerow({'Input Size': input_size, 'Index Size': index_size, 'Execution Time': execution_time})

if __name__ == "__main__":
    main(args)
