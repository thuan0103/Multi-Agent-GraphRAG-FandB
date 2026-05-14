import subprocess
import sys

def main():
    cmd = [
        sys.executable, "training/quantize_edge.py",
        "--benchmark-only",
        "--out-dir", "models/gguf",
        "--quant-levels", "Q4_0", "Q4_K_M", "Q4_K_S", "IQ4_XS",
        "--report-json", "reports/edge_benchmark.json",
    ]
    print("Chạy Edge Benchmark (CPU ARM)...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()