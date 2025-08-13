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
python3 benchmark.py --size 500
```