import subprocess
import os

def run_benchmark(det):
    command = ["python", "benchmark.py"]
    if det:
        command.append("--det")
    result = subprocess.run(command)
    if result.returncode != 0:
        print("Benchmark aborted due to errors.")

if __name__ == "__main__":
    # Run non-deterministic version to generate reference outputs
    run_benchmark(False)
    
    # Run deterministic version to compare with reference
    run_benchmark(True)

