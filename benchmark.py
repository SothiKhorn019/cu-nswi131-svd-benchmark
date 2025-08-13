import os
import numpy as np
import torch
import time
import argparse
import csv
import threadpoolctl


def generate_random_matrix(n, m, seed=42):
    np.random.seed(seed)
    return np.random.rand(n, m).astype(np.float64)


def normalize(vec):
    norm = sum(x**2 for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def dot(u, v):
    total = 0
    for ui, vi in zip(u, v):
        total += ui * vi

    return total


def matvec(A, v):
    result = []
    for i in range(len(A)):
        total = 0
        for j in range(len(v)):
            total += A[i][j] * v[j]
        result.append(total)

    return result


def calculate_frobenius_norm(matrix):
    return np.linalg.norm(matrix, "fro")


def raw_python_svd(A, iterations=10):
    """Computes the top singular component (k=1) using Power Iteration."""
    m = len(A[0])
    v = normalize([1.0] * m)
    for _ in range(iterations):
        u = normalize(matvec(A, v))
        v = normalize(matvec(list(zip(*A)), u))

    sigma = dot(u, matvec(A, v))

    U_approx = np.array(u).reshape(-1, 1).astype(np.float64)
    S_approx = np.array([sigma]).astype(np.float64)
    Vt_approx = np.array(v).reshape(1, -1).astype(np.float64)
    return S_approx, U_approx, Vt_approx


def numpy_svd(A):
    """Computes SVD via Numpy and returns only the top component (k=1)."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return (
        S[:1].astype(np.float64),
        U[:, :1].astype(np.float64),
        Vt[:1, :].astype(np.float64),
    )


def pytorch_svd(A):
    """Computes SVD via PyTorch and returns only the top component (k=1)."""
    A_torch = torch.from_numpy(A).double()
    U, S, Vh = torch.linalg.svd(A_torch, full_matrices=False)
    return (
        S[:1].cpu().numpy().astype(np.float64),
        U[:, :1].cpu().numpy().astype(np.float64),
        Vh[:1, :].cpu().numpy().astype(np.float64),
    )


def time_single_iteration(func, A):
    """Times a single SVD function call."""
    start = time.perf_counter()
    s, u, vt = func(A)
    end = time.perf_counter()
    return (end - start, (s, u, vt))


def benchmark_svd_methods(n=100, repeat=50, output_file=None, num_threads=1):
    if output_file is None:
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, f"benchmark_s{n}_i{repeat}_t{num_threads}.csv"
        )

    A_np = generate_random_matrix(n, n)
    frobenius_norm_A = calculate_frobenius_norm(A_np)

    implementations = {
        "raw_python": (raw_python_svd, A_np),
        "numpy": (numpy_svd, A_np),
        "torch": (pytorch_svd, A_np),
    }

    all_results = []
    print(f"\nRunning benchmarks sequentially for {repeat} repetitions...")

    for impl_name, (func, A_data) in implementations.items():
        print(f"Benchmarking '{impl_name}' with {num_threads} thread(s)...")
        with threadpoolctl.threadpool_limits(limits=num_threads):
            for i in range(repeat):
                duration, svd_components = time_single_iteration(func, A_data)
                all_results.append(
                    (
                        impl_name,
                        i + 1,
                        n,
                        duration,
                        svd_components,
                        A_np,
                        frobenius_norm_A,
                        num_threads,
                    )
                )

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "no",
                "implementation",
                "size",
                "threads",
                "time",
                "frobenius_error",
                "relative_error",
            ]
        )

        for result_tuple in all_results:
            (
                impl,
                i,
                n_val,
                t,
                svd_components,
                A_np_original,
                frobenius_norm_A_original,
                current_num_threads,
            ) = result_tuple

            s_k, u_k, vt_k = svd_components

            A_reconstructed = u_k @ np.diag(s_k) @ vt_k
            error_matrix = A_np_original - A_reconstructed
            frobenius_error = calculate_frobenius_norm(error_matrix)
            relative_error = (
                (frobenius_error / frobenius_norm_A_original)
                if frobenius_norm_A_original != 0
                else np.nan
            )

            writer.writerow(
                [
                    i,
                    impl,
                    n_val,
                    current_num_threads,
                    f"{t:.8f}",
                    f"{frobenius_error:.8f}",
                    f"{relative_error:.8f}",
                ]
            )

    print(f"\nBenchmark complete. Results saved to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark SVD implementations (k=1) with thread control."
    )
    parser.add_argument(
        "--size", "-n", type=int, default=100, help="Matrix size (n x n)"
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=50, help="Number of repeats for timing"
    )
    args = parser.parse_args()

    thread_counts = [1, 4, 8, 12]
    max_cpu_cores = os.cpu_count()

    for num_threads in thread_counts:
        if num_threads > max_cpu_cores:
            print(
                f"\n--- Skipping benchmark for {num_threads} threads (more than available cores: {max_cpu_cores}) ---"
            )
            continue

        print(f"\n--- Running Full Benchmark for {num_threads} thread(s) ---")

        benchmark_svd_methods(
            n=args.size,
            repeat=args.iterations,
            num_threads=num_threads,
        )
