# SVD Benchmark

This project evaluates the performance and accuracy of different Singular Value Decomposition (SVD) implementations in Python, including a naive raw Python approach, NumPy, and PyTorch.

## Overview

* **Goal:** To compare the speed and accuracy of SVD implementations across various matrix sizes and thread counts.
* **Implementations Tested:**

  * Raw Python (custom, unoptimized)
  * NumPy (CPU, multi-threaded)
  * PyTorch (CPU, multi-threaded)

## Collecting Data

To generate benchmarking data for your experiments, run the following command:

```bash
python3 benchmark.py --iterations 100 --processes 1 --size 100
```

* Adjust `--size`, `--processes`, and other parameters as needed to produce data for different scenarios.
* Output data will be saved in the `data` directory (or as configured in the script).

## Experiments

* **Performance:** Measured average execution time as matrix size and thread count increased.
* **Accuracy:** Assessed reconstruction error using Frobenius norm and relative error.
* **Setup:** All experiments were run on an Apple M2 Pro CPU, using dense random matrices.

## Key Findings

* **NumPy and PyTorch** are orders of magnitude faster and more accurate than the naive Python implementation.
* Both libraries benefit from multi-threading, especially on large matrices.
* The raw Python implementation is not practical for real-world applications due to high errors and poor scalability.

## Limitations

- **Hardware Constraints**: All experiments were conducted on a single hardware configuration (Apple M2 Pro, CPU only). Results may differ on systems with different CPUs, more cores, or on dedicated GPUs, especially for PyTorch.
- **Fixed Data Size Range**: Matrix sizes tested were between 100 and 2000. Larger or smaller matrices, or non-square matrices, may result in different trends.

## Conclusion

For practical SVD computation, always use optimized libraries such as NumPy or PyTorch for both speed and accuracy.
