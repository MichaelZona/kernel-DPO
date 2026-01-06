"""
Spectral clustering for preference grouping based on affinity matrix T.

According to paper Appendix A.2:
1. Build symmetric matrix A_1 = (1/2) * (T + T^T)
2. Build block matrix A = [[A_1, T], [T^T, 0]]
3. Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
4. Compute k eigenvectors of L with smallest eigenvalues
5. Run k-means on the row embeddings
6. Merge source and target copies to get final clusters
"""

import numpy as np
from typing import List, Tuple
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
import scipy.linalg


def cluster_preferences(
    T: np.ndarray,
    k: int = 4,
    random_state: int = 42,
    verbose: bool = True,
    normalize: bool = True,
    use_exp: bool = False,
    temperature: float = 1.0,
    smoothing_factor: float = 1.0,
) -> List[np.ndarray]:
    """
    Cluster preferences into k groups using spectral clustering on affinity matrix T.
    
    According to paper Appendix A.2:
    - T is the affinity matrix where T[i, j] estimates how task j affects task i
    - We build a block matrix A to capture both symmetric and directional relationships
    - Apply spectral clustering on the normalized Laplacian of A
    
    Args:
        T: Affinity matrix of shape [n, n] where T[i, j] estimates how task j affects task i
        k: Number of clusters (default: 4 for UltraFeedback criteria)
        random_state: Random seed for k-means
        verbose: Whether to print progress information
        normalize: Whether to normalize T matrix (row-wise normalization) to smooth the distribution
        use_exp: Whether to apply exp transformation to enhance differences
        temperature: Temperature parameter for exp transformation (only used if use_exp=True)
        smoothing_factor: Smoothing strength. Higher values = more smoothing (more uniform distribution).
                         When > 0, adds a uniform component to T matrix before normalization.
                         Formula: T_smooth = T + smoothing_factor * mean(T)
                         Higher smoothing_factor adds more uniform background, making distribution smoother.
                         Recommended values: 0.0 (no smoothing), 0.1-0.5 (light), 1.0-2.0 (moderate), 5.0+ (strong).
                         Default: 0.0 (simple normalization, no extra smoothing).
        
    Returns:
        clusters: List of k arrays, each containing the indices of preferences in that cluster
    """
    n = T.shape[0]
    
    if T.shape[1] != n:
        raise ValueError(f"T must be square, got shape {T.shape}")
    
    if verbose:
        print(f"Clustering {n} preferences into {k} clusters...")
        print(f"T matrix statistics (before smoothing):")
        print(f"  Mean: {np.mean(T):.6f}")
        print(f"  Std: {np.std(T):.6f}")
        print(f"  Min: {np.min(T):.6f}")
        print(f"  Max: {np.max(T):.6f}")
    
    # Smoothing: Normalize T matrix to prevent all samples clustering into one group
    T_smooth = T.copy()
    
    if normalize:
        # Apply smoothing: add a constant to make distribution more uniform
        # Higher smoothing_factor = add more constant = more uniform distribution
        if smoothing_factor > 0:
            # Add a constant term to smooth the distribution
            # T_smooth = T + smoothing_factor * mean(T) * uniform_matrix
            # This makes all entries more similar, leading to more uniform distribution
            
            T_mean = np.mean(T_smooth)
            T_max = np.max(T_smooth)
            
            # Add uniform background: smoothing_factor controls how much uniform component to add
            # Higher smoothing_factor -> more uniform -> smoother distribution
            uniform_component = T_mean * smoothing_factor
            
            # Add uniform component to all entries
            T_smooth = T_smooth + uniform_component
            
            if verbose:
                print(f"  Added uniform component: {uniform_component:.6f} (smoothing_factor={smoothing_factor:.2f})")
        
        # Always apply row-wise normalization after smoothing
        row_sums = np.sum(T_smooth, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)  # Avoid division by zero
        T_smooth = T_smooth / row_sums
        
        if verbose:
            smoothing_type = "with smoothing" if smoothing_factor > 0 else "simple"
            print(f"\nApplied {smoothing_type} normalization (smoothing_factor={smoothing_factor:.2f})")
            print(f"T matrix statistics (after normalization):")
            print(f"  Mean: {np.mean(T_smooth):.6f}")
            print(f"  Std: {np.std(T_smooth):.6f}")
            print(f"  Min: {np.min(T_smooth):.6f}")
            print(f"  Max: {np.max(T_smooth):.6f}")
    
    if use_exp:
        # Apply exp transformation to enhance differences
        # T_exp = exp(T / temperature)
        T_smooth = np.exp(T_smooth / temperature)
        
        if verbose:
            print(f"\nApplied exp transformation (temperature={temperature})")
            print(f"T matrix statistics (after exp):")
            print(f"  Mean: {np.mean(T_smooth):.6f}")
            print(f"  Std: {np.std(T_smooth):.6f}")
            print(f"  Min: {np.min(T_smooth):.6f}")
            print(f"  Max: {np.max(T_smooth):.6f}")
    
    # Step 1: Build symmetric matrix A_1 = (1/2) * (T + T^T)
    A_1 = 0.5 * (T_smooth + T_smooth.T)
    
    if verbose:
        print(f"\nStep 1: Built symmetric matrix A_1 (shape: {A_1.shape})")
    
    # Step 2: Build block matrix A = [[A_1, T], [T^T, 0]]
    # This creates a 2n x 2n matrix where:
    # - Upper-left block A_1: symmetric similarities
    # - Upper-right block T: directional transfer (source -> target)
    # - Lower-left block T^T: directional transfer (target -> source)
    # - Lower-right block 0: no extra similarity structure
    A = np.zeros((2 * n, 2 * n), dtype=T_smooth.dtype)
    A[:n, :n] = A_1  # Upper-left: symmetric similarities
    A[:n, n:] = T_smooth    # Upper-right: source -> target
    A[n:, :n] = T_smooth.T  # Lower-left: target -> source
    # Lower-right remains zeros
    
    if verbose:
        print(f"Step 2: Built block matrix A (shape: {A.shape})")
    
    # Step 3: Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    # where D is the diagonal degree matrix: D_{uu} = sum_v A_{uv}
    D = np.sum(A, axis=1)  # Degree of each node
    
    # Handle isolated nodes (degree = 0) to avoid division by zero
    D_safe = np.where(D > 0, D, 1.0)
    D_inv_sqrt = 1.0 / np.sqrt(D_safe)
    D_inv_sqrt[D == 0] = 0.0  # Isolated nodes get 0
    
    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt_diag = np.diag(D_inv_sqrt)
    L = np.eye(2 * n) - D_inv_sqrt_diag @ A @ D_inv_sqrt_diag
    
    if verbose:
        print(f"Step 3: Computed normalized Laplacian L (shape: {L.shape})")
        print(f"  Number of isolated nodes: {np.sum(D == 0)}")
    
    # Step 4: Compute k eigenvectors of L with smallest eigenvalues
    # Note: Laplacian has smallest eigenvalues for best-connected components
    try:
        eigenvalues, eigenvectors = scipy.linalg.eigh(L)
    except Exception as e:
        # Fallback to scipy.sparse if matrix is too large
        if verbose:
            print(f"  Using sparse eigensolver due to: {e}")
        from scipy.sparse.linalg import eigsh
        eigenvalues, eigenvectors = eigsh(L, k=min(k + 1, 2 * n - 1), which='SM')
    
    # Sort by eigenvalues (ascending) and take k smallest
    idx = np.argsort(eigenvalues)[:k]
    U = eigenvectors[:, idx]  # [2n, k]
    
    if verbose:
        print(f"Step 4: Computed {k} eigenvectors")
        print(f"  Eigenvalue range: [{np.min(eigenvalues[idx]):.6f}, {np.max(eigenvalues[idx]):.6f}]")
    
    # Step 5: Run k-means on row embeddings U
    # Each row of U is an embedding of a node (either source or target copy)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(U)
    
    if verbose:
        print(f"Step 5: Ran k-means clustering")
        print(f"  Cluster sizes: {np.bincount(cluster_labels)}")
    
    # Step 6: Merge source and target copies
    # Each original task appears twice in A (once as source, once as target)
    # We merge them by taking the union of assignments
    source_labels = cluster_labels[:n]  # First n nodes are sources
    target_labels = cluster_labels[n:]  # Last n nodes are targets
    
    # Build clusters: for each cluster, collect tasks that appear in either source or target
    clusters = [[] for _ in range(k)]
    for task_idx in range(n):
        source_cluster = source_labels[task_idx]
        target_cluster = target_labels[task_idx]
        # Add to both clusters if different, or to the single cluster if same
        clusters[source_cluster].append(task_idx)
        if target_cluster != source_cluster:
            clusters[target_cluster].append(task_idx)
    
    # Remove duplicates and convert to numpy arrays
    clusters = [np.unique(np.array(cluster)) for cluster in clusters]
    
    # Remove empty clusters and reorder
    clusters = [c for c in clusters if len(c) > 0]
    
    if verbose:
        print(f"\nStep 6: Merged source and target copies")
        print(f"Final clusters:")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i}: {len(cluster)} preferences - {cluster[:10]}{'...' if len(cluster) > 10 else ''}")
    
    return clusters


def save_clusters(
    clusters: List[np.ndarray],
    output_path: str,
    verbose: bool = True,
):
    """
    Save clusters to disk as a numpy file.
    
    Args:
        clusters: List of arrays, each containing preference indices in that cluster
        output_path: Path to save the clusters
        verbose: Whether to print confirmation
    """
    # Save as a dictionary for easy loading
    cluster_dict = {
        'clusters': clusters,
        'num_clusters': len(clusters),
        'total_preferences': sum(len(c) for c in clusters),
    }
    
    np.save(output_path, cluster_dict, allow_pickle=True)
    
    if verbose:
        print(f"\nSaved clusters to: {output_path}")
        print(f"  Number of clusters: {len(clusters)}")
        print(f"  Total preferences: {sum(len(c) for c in clusters)}")


def load_clusters(cluster_path: str) -> List[np.ndarray]:
    """
    Load clusters from disk.
    
    Args:
        cluster_path: Path to the saved clusters file
        
    Returns:
        clusters: List of arrays, each containing preference indices in that cluster
    """
    cluster_dict = np.load(cluster_path, allow_pickle=True).item()
    return cluster_dict['clusters']


if __name__ == "__main__":
    """
    Example usage:
    
    python -m src.cluster_preferences \
        --T_matrix pre_compute/T_ultrafeedback_llama1b_1000samples.npy \
        --k 4 \
        --output pre_compute/clusters_ultrafeedback_llama1b_1000samples.npy
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cluster preferences using spectral clustering on affinity matrix T"
    )
    parser.add_argument(
        "--T_matrix",
        type=str,
        required=True,
        help="Path to affinity matrix T (.npy file)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of clusters (default: 4 for UltraFeedback criteria)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for clusters (default: clusters_{T_matrix_basename}.npy)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for k-means (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress information",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Apply row-wise normalization to smooth T matrix (default: True)",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_false",
        dest="normalize",
        help="Disable normalization",
    )
    parser.add_argument(
        "--use_exp",
        action="store_true",
        help="Apply exp transformation to enhance differences",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for exp transformation (default: 1.0)",
    )
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.0,
        help="Smoothing strength. Higher = more smoothing (more uniform). "
             "Adds uniform component: T_smooth = T + smoothing_factor * mean(T). "
             "Recommended: 0.0 (no smoothing), 0.1-0.5 (light), 1.0-2.0 (moderate), 5.0+ (strong). "
             "Default: 0.0 (simple normalization)",
    )
    
    args = parser.parse_args()
    
    # Load T matrix
    print(f"Loading affinity matrix from: {args.T_matrix}")
    T = np.load(args.T_matrix)
    print(f"Loaded T matrix with shape: {T.shape}")
    
    # Cluster preferences
    clusters = cluster_preferences(
        T=T,
        k=args.k,
        random_state=args.random_state,
        verbose=args.verbose,
        normalize=args.normalize,
        use_exp=args.use_exp,
        temperature=args.temperature,
        smoothing_factor=args.smoothing_factor,
    )
    
    # Determine output path
    if args.output is None:
        import os
        T_basename = os.path.splitext(os.path.basename(args.T_matrix))[0]
        args.output = os.path.join(
            os.path.dirname(args.T_matrix),
            f"clusters_{T_basename}.npy"
        )
    
    # Save clusters
    save_clusters(clusters, args.output, verbose=args.verbose)
    
    print(f"\nClustering complete!")
