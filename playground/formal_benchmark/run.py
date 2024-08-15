import subprocess
import os

def run_benchmark(det, jax_scan):
    command = ["python", "benchmark.py"]
    if det:
        command.append("--det")
    if jax_scan:
        command.append("--jax-scan")
    result = subprocess.run(command)
    if result.returncode != 0:
        print("Benchmark aborted due to errors.")

if __name__ == "__main__":
    # Run non-deterministic version to generate reference outputs
    run_benchmark(det=False)
    
    # Run deterministic version to compare with reference
    run_benchmark(det=True)

