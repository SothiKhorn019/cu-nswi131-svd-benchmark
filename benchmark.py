import os
import numpy as np
import torch
import time
import argparse
import csv
from multiprocessing import Pool, cpu_count


def generate_random_matrix(n, m, seed=42):
    np.random.seed(seed)
    return np.random.rand(n, m).astype(np.float64)


def normalize(vec):
    norm = sum(x**2 for x in vec) ** 0.5
    return [x / norm for x in vec]


def dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))


def matvec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def raw_python_svd(A, k=1, iterations=10):
    n = len(A)
    v = [1.0] * n
    for _ in range(iterations):
        v = normalize(matvec(A, v))
    sigma = dot(v, matvec(A, v)) ** 0.5

    # Always return components suitable for k=1 reconstruction, regardless of input k
    U_approx = (
        np.array(v).reshape(-1, 1).astype(np.float64)
        if sigma > 0
        else np.zeros((n, 1), dtype=np.float64)
    )
    S_approx = np.array([sigma]).astype(np.float64)  # This will be 1D array of size 1
    Vt_approx = (
        np.array(v).reshape(1, -1).astype(np.float64)
        if sigma > 0
        else np.zeros((1, n), dtype=np.float64)
    )

    # Return the k=1 approximated SVD components
    return S_approx, U_approx, Vt_approx


def numpy_svd(A, k=1):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # Ensure output types are float64 for consistency
    return (
        S[:k].astype(np.float64),
        U[:, :k].astype(np.float64),
        Vt[:k, :].astype(np.float64),
    )


def pytorch_svd(A, k=1):
    A_torch = torch.from_numpy(A).double()  # Change to .double() for float64
    U, S, V = torch.svd(A_torch)
    # Convert PyTorch tensors to NumPy arrays with float64
    return (
        S[:k].cpu().numpy().astype(np.float64),
        U[:, :k].cpu().numpy().astype(np.float64),
        V[:, :k].T.cpu().numpy().astype(np.float64),
    )


def calculate_frobenius_norm(matrix):
    return np.linalg.norm(matrix, "fro")


# Time function to run a single iteration
def time_single_iteration(func, A, k):
    start = time.perf_counter()
    s, u, vt = func(A, k=k)
    end = time.perf_counter()
    return (end - start, (s, u, vt))


def benchmark_svd_methods(n=100, k=1, repeat=100, output_file=None, num_processes=1):
    # Prepare output path
    if output_file is None:
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, f"benchmark_s{n}_k{k}_i{repeat}_t{num_processes}.csv"
        )

    # Generate matrix
    A_np = generate_random_matrix(n, n)
    A_list = A_np.tolist()
    frobenius_norm_A = calculate_frobenius_norm(A_np)

    # Dictionary to map implementation names to their respective functions and argument formats
    implementations = {
        "raw_python": (raw_python_svd, A_list),
        "numpy": (numpy_svd, A_np),
        "torch": (pytorch_svd, A_np),
    }

    all_results = []

    # Use multiprocessing Pool for parallelizing repetitions
    if num_processes == 1:
        print(f"Running benchmarks sequentially for {repeat} repetitions...")
        for impl_name, (func, A_data) in implementations.items():
            for i in range(repeat):
                duration, svd_components = time_single_iteration(func, A_data, k)
                all_results.append(
                    (
                        impl_name,
                        i + 1,
                        n,
                        duration,
                        svd_components,
                        A_np,
                        frobenius_norm_A,
                        k,
                        num_processes,
                    )
                )
    else:
        print(
            f"Running benchmarks with {num_processes} processes for {repeat} repetitions..."
        )
        with Pool(processes=num_processes) as pool:
            for impl_name, (func, A_data) in implementations.items():
                args_list = [(func, A_data, k) for _ in range(repeat)]
                results_for_impl = pool.starmap(time_single_iteration, args_list)
                for i, (duration, svd_components) in enumerate(results_for_impl):
                    all_results.append(
                        (
                            impl_name,
                            i + 1,
                            n,
                            duration,
                            svd_components,
                            A_np,
                            frobenius_norm_A,
                            k,
                            num_processes,
                        )
                    )

    # Write results to CSV
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "no",
                "implementation",
                "size",
                "processes",
                "time",
                "frobenius_error",
                "relative_error",
            ]
        )

        for result_tuple in all_results:
            (
                impl,
                i,
                n,
                t,
                svd_components,
                A_np_original,
                frobenius_norm_A_original,
                k_val,
                current_num_processes,
            ) = result_tuple
            s_k, u_k, vt_k = svd_components

            frobenius_error = np.nan
            relative_error = np.nan

            if impl == "raw_python" and k_val > 1:
                pass
            else:
                effective_k_for_reconstruction = k_val
                if impl == "raw_python":
                    effective_k_for_reconstruction = 1

                S_k_diag = np.zeros(
                    (effective_k_for_reconstruction, effective_k_for_reconstruction),
                    dtype=np.float64,
                )
                np.fill_diagonal(S_k_diag, s_k[:effective_k_for_reconstruction])

                A_reconstructed = u_k @ S_k_diag @ vt_k

                error_matrix = A_np_original - A_reconstructed
                frobenius_error = calculate_frobenius_norm(error_matrix)
                relative_error = (
                    frobenius_error / frobenius_norm_A_original
                    if frobenius_norm_A_original != 0
                    else np.nan
                )

            writer.writerow(
                [
                    i,
                    impl,
                    n,
                    current_num_processes,
                    f"{t:.8f}",
                    f"{frobenius_error:.8f}",
                    f"{relative_error:.8f}",
                ]
            )

    print(f"Benchmark complete. Results saved to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SVD implementations.")
    parser.add_argument(
        "--size",
        "-n",
        type=int,
        default=100,
        help="Matrix size for n x n matrix (default: 100)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of singular values to compute (default: 1)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Number of repeats for timing each method (default: 10)",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=1,
        help=f"Number of processes to use for repetitions (default: 1, max: {cpu_count()})",
    )
    args = parser.parse_args()

    num_processes_to_use = min(args.processes, cpu_count())
    if args.processes > cpu_count():
        print(
            f"Warning: Requested {args.processes} processes, but only {cpu_count()} CPU cores are available. Using {cpu_count()} processes."
        )
    elif args.processes <= 0:
        print("Warning: Number of processes must be positive. Defaulting to 1.")
        num_processes_to_use = 1

    benchmark_svd_methods(
        n=args.size, k=args.k, repeat=args.iterations, num_processes=num_processes_to_use
    )
