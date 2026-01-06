import argparse
import os
import sys
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
sys.path.insert(0, project_root)

from src.logistic_regression import load_precomputed_gradients_b
from src.kernel_estimation import estimate_affinity_scores


def main():
    parser = argparse.ArgumentParser(
        description="Compute affinity matrix T for UltraFeedback dataset"
    )
    parser.add_argument(
        "--precompute_file",
        type=str,
        required=True,
        help="Precomputed gradients file (e.g., Llama-3.2-3B-Instruct_200_0.01.pt)",
    )
    parser.add_argument(
        "--num_subsets",
        type=int,
        default=1000,
        help="Number of random subsets to sample (default: 1000)",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10,
        help="Size of each subset (default: 10)",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=1000,
        help="Max iterations for logistic regression (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for logistic regression (default: 0.1)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for convergence (default: 1e-6)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file name for T matrix (default: T_{precompute_file_basename}.npy)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: pre_compute/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress information",
    )

    args = parser.parse_args()

    # Load precomputed gradients
    print("=" * 60)
    print("Loading precomputed gradients...")
    print("=" * 60)
    gradients, b_values, z_values, projection_matrix, projection_dim = (
        load_precomputed_gradients_b(args.precompute_file)
    )

    n = len(gradients)
    print(f"\nLoaded {n} gradients")
    if len(gradients) > 0:
        print(f"Gradient dimension: {len(gradients[0])}")
    if projection_matrix is not None:
        print(f"Projection matrix shape: {projection_matrix.shape}")
    if projection_dim is not None:
        print(f"Projection dimension: {projection_dim}")

    # Compute affinity matrix T
    print("\n" + "=" * 60)
    print("Computing affinity matrix T...")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Number of subsets: {args.num_subsets}")
    print(f"  Subset size: {args.subset_size}")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Tolerance: {args.tol}")
    print()

    T = estimate_affinity_scores(
        gradients=gradients,
        z=z_values,
        projection_matrix=projection_matrix,
        num_subsets=args.num_subsets,
        subset_size=args.subset_size,
        max_iters=args.max_iters,
        lr=args.lr,
        tol=args.tol,
        verbose=args.verbose,
    )

    # Print statistics
    print("\n" + "=" * 60)
    print("Affinity Matrix T Statistics:")
    print("=" * 60)
    print(f"T shape: {T.shape}")
    print(f"  Mean: {np.mean(T):.6f}")
    print(f"  Std: {np.std(T):.6f}")
    print(f"  Min: {np.min(T):.6f}")
    print(f"  Max: {np.max(T):.6f}")
    print(f"  Diagonal mean: {np.mean(np.diag(T)):.6f}")
    print(f"  Non-diagonal mean: {np.mean(T[~np.eye(T.shape[0], dtype=bool)]):.6f}")

    # Determine output file path
    if args.output_file is None:
        # Generate output filename from precompute file
        precompute_basename = os.path.splitext(args.precompute_file)[0]
        output_file = f"T_{precompute_basename}.npy"
    else:
        output_file = args.output_file

    if args.output_dir is None:
        output_dir = os.path.join(project_root, "pre_compute")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # Save T matrix
    np.save(output_path, T)
    print(f"\nSaved affinity matrix to: {output_path}")
    print("=" * 60)

    return T, output_path


if __name__ == "__main__":
    main()
