import sys
import time
from pathlib import Path
import numpy as np

# Set project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# TFLite Import Fallback
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ImportError:
    import tensorflow.lite as tflite

from config import MODEL_DIR


def evaluate_tflite_model(model_path, num_runs=100):
    """Measures TFLite model size, latency, and estimated RAM usage."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None, None, None

    # 1. Model Size in KB
    file_size_kb = model_path.stat().st_size / 1024.0

    # 2. Initialize Interpreter
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare dummy input based on tensor shape and dtype
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.int8:
        dummy_input = np.zeros(input_shape, dtype=np.int8)
    else:
        dummy_input = np.zeros(input_shape, dtype=np.float32)

    # Warm-up run
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

    # 3. Benchmark Latency
    start_time = time.perf_counter()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0

    # Estimated RAM Footprint
    ram_mb = (file_size_kb / 1024.0) * 1.5 + 8.0

    return file_size_kb, avg_latency_ms, ram_mb


def main():
    variants = [
        ("M1: Baseline", MODEL_DIR / "model_fp32.tflite", ".tflite (FP32)"),
        ("M2: PTQ INT8", MODEL_DIR / "model_ptq.tflite", ".tflite (INT8)"),
        ("M3: Pruned + PTQ", MODEL_DIR / "model_pruned_ptq.tflite", ".tflite (INT8)"),
    ]

    print("\n" + "=" * 90)
    print("                      BENCHMARKING RESULTS TABLE                       ")
    print("=" * 90 + "\n")

    # Table Header (5 columns)
    print(
        f"| {'Model Variant':<20} | {'File Format':<15} | {'Model Size (KB)':<15} | {'Mean Latency (ms)':<17} | {'Memory Footprint (MB)':<21} |"
    )
    print(f"|{'-'*22}|{'-'*17}|{'-'*17}|{'-'*19}|{'-'*23}|")

    # Table Rows (3 rows x 5 columns = 15 cells)
    for name, path, fmt in variants:
        size_kb, latency, ram = evaluate_tflite_model(path)
        if size_kb is not None:
            print(
                f"| {name:<20} | {fmt:<15} | {size_kb:<15.2f} | {latency:<17.3f} | {ram:<21.2f} |"
            )
        else:
            print(
                f"| {name:<20} | {fmt:<15} | {'N/A':<15} | {'N/A':<17} | {'N/A':<21} |"
            )

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()