"""
Adjust cluster assignments to make each cluster dominated by a single criterion.

This script:
1. Loads current clusters and dataset with criteria labels
2. Reassigns data points to clusters based on their criteria
3. Ensures each cluster is dominated by one criterion (at least 60%+)
4. Saves the adjusted clusters
"""

import os
import sys
import numpy as np
import argparse
from collections import Counter, defaultdict
from datasets import load_from_disk

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
sys.path.insert(0, project_root)

from src.cluster_preferences import load_clusters

CRITERIA = ["helpfulness", "truthfulness", "instruction_following", "honesty"]


def adjust_clusters_by_criteria(
    clusters_path: str,
    dataset_path: str,
    output_path: str,
    downsample_ratio: float = None,
    seed: int = 42,
    max_samples: int = None,
    min_dominance: float = 0.6,  # Minimum fraction for dominant criterion
    verbose: bool = True,
):
    """
    Adjust clusters so each cluster is dominated by one criterion.
    
    Args:
        clusters_path: Path to current clusters .npy file
        dataset_path: Path to dataset directory
        output_path: Path to save adjusted clusters
        downsample_ratio: Downsample ratio used during precompute
        seed: Random seed used during precompute
        max_samples: Maximum samples used during precompute
        min_dominance: Minimum fraction for dominant criterion (default: 0.6 = 60%)
        verbose: Whether to print progress
    """
    # Load current clusters
    clusters = load_clusters(clusters_path)
    
    if verbose:
        print("=" * 60)
        print("Adjusting Clusters by Criteria")
        print("=" * 60)
        print(f"Loaded {len(clusters)} clusters")
        print(f"Total preferences: {sum(len(c) for c in clusters)}")
        print()
    
    # Load dataset and get criteria
    train_ds = load_from_disk(os.path.join(dataset_path, "train"))
    
    if "criterion_used" not in train_ds.column_names:
        raise ValueError("Dataset must have 'criterion_used' field")
    
    # Reapply downsample/shuffle to match precompute
    original_size = len(train_ds)
    
    if downsample_ratio is not None and downsample_ratio < 1.0:
        train_ds = train_ds.shuffle(seed=seed)
        n_samples = int(len(train_ds) * downsample_ratio)
        train_ds = train_ds.select(range(n_samples))
        if verbose:
            print(f"Applied downsample (ratio={downsample_ratio}, seed={seed})")
            print(f"  Original size: {original_size}")
            print(f"  Downsampled size: {len(train_ds)}")
    
    if max_samples is not None and max_samples < len(train_ds):
        train_ds = train_ds.select(range(max_samples))
        if verbose:
            print(f"Applied max_samples limit: {max_samples}")
    
    all_criteria = train_ds["criterion_used"]
    
    # Get all data point indices from clusters
    all_indices = []
    for cluster in clusters:
        all_indices.extend([int(idx) for idx in cluster])
    all_indices = np.array(all_indices, dtype=np.int64)
    
    # Remove duplicates and sort
    all_indices = np.unique(all_indices)
    
    # Get criteria for all data points
    data_criteria = [all_criteria[int(idx)] for idx in all_indices]
    
    if verbose:
        print(f"\nTotal data points: {len(all_indices)}")
        print(f"Criteria distribution:")
        crit_counts = Counter(data_criteria)
        for crit in CRITERIA:
            count = crit_counts.get(crit, 0)
            pct = count / len(data_criteria) * 100 if len(data_criteria) > 0 else 0
            print(f"  {crit}: {count} ({pct:.1f}%)")
        print()
    
    # Group indices by criterion
    criterion_to_indices = defaultdict(list)
    for idx, crit in zip(all_indices, data_criteria):
        criterion_to_indices[crit].append(idx)
    
    # Calculate target sizes for each cluster
    # Each cluster should be dominated by one criterion
    total_points = len(all_indices)
    target_cluster_size = total_points // len(CRITERIA)
    
    # Create new clusters: assign each criterion to a cluster
    new_clusters = []
    assigned_indices = set()
    
    # Strategy: assign each criterion to a cluster, ensuring dominance
    for cluster_idx, target_criterion in enumerate(CRITERIA):
        cluster_indices = []
        
        # First, add all points with target criterion
        target_indices = criterion_to_indices[target_criterion]
        for idx in target_indices:
            if idx not in assigned_indices:
                cluster_indices.append(idx)
                assigned_indices.add(idx)
        
        # If we need more points, add from other criteria (but keep dominance)
        # First, add remaining target_criterion points
        remaining_target = [idx for idx in target_indices if idx not in assigned_indices]
        for idx in remaining_target:
            cluster_indices.append(idx)
            assigned_indices.add(idx)
        
        # Then add from other criteria if needed, but ensure target_criterion remains dominant
        remaining_indices = [idx for idx in all_indices if idx not in assigned_indices]
        np.random.seed(seed + cluster_idx)
        np.random.shuffle(remaining_indices)
        
        target_size = min(target_cluster_size, len(target_indices) + len(remaining_target))
        
        for idx in remaining_indices:
            if len(cluster_indices) >= target_size:
                break
            
            # Check if adding this point maintains dominance
            crit = all_criteria[int(idx)]
            if crit == target_criterion:
                cluster_indices.append(idx)
                assigned_indices.add(idx)
            else:
                # Check dominance: target_criterion should be >= min_dominance
                current_target_count = sum(1 for i in cluster_indices if all_criteria[int(i)] == target_criterion)
                if (current_target_count) / (len(cluster_indices) + 1) >= min_dominance:
                    cluster_indices.append(idx)
                    assigned_indices.add(idx)
        
        new_clusters.append(np.array(cluster_indices, dtype=np.int64))
        
        if verbose:
            cluster_criteria = [all_criteria[int(idx)] for idx in cluster_indices]
            crit_counts = Counter(cluster_criteria)
            total = len(cluster_indices)
            target_count = crit_counts.get(target_criterion, 0)
            target_pct = target_count / total * 100 if total > 0 else 0
            print(f"Cluster {cluster_idx} (target: {target_criterion}): {total} preferences")
            print(f"  {target_criterion}: {target_count} ({target_pct:.1f}%)")
            for crit in CRITERIA:
                if crit != target_criterion:
                    count = crit_counts.get(crit, 0)
                    pct = count / total * 100 if total > 0 else 0
                    if count > 0:
                        print(f"  {crit}: {count} ({pct:.1f}%)")
            print()
    
    # Assign any remaining unassigned indices to clusters
    remaining = [idx for idx in all_indices if idx not in assigned_indices]
    if remaining:
        if verbose:
            print(f"Assigning {len(remaining)} remaining points...")
        
        for idx in remaining:
            crit = all_criteria[int(idx)]
            # Find cluster with matching criterion
            assigned = False
            for cluster_idx, target_crit in enumerate(CRITERIA):
                if crit == target_crit:
                    new_clusters[cluster_idx] = np.append(new_clusters[cluster_idx], idx)
                    assigned_indices.add(idx)
                    assigned = True
                    break
            
            if not assigned:
                # If no match, assign to smallest cluster
                smallest_cluster_idx = np.argmin([len(c) for c in new_clusters])
                new_clusters[smallest_cluster_idx] = np.append(new_clusters[smallest_cluster_idx], idx)
                assigned_indices.add(idx)
    
    # Verify all indices are assigned
    if len(assigned_indices) != len(all_indices):
        if verbose:
            print(f"Warning: {len(all_indices) - len(assigned_indices)} indices not assigned")
            print(f"  Assigned: {len(assigned_indices)}, Total: {len(all_indices)}")
            missing = [idx for idx in all_indices if idx not in assigned_indices]
            print(f"  Missing indices (first 10): {missing[:10]}")
    
    # Convert to list of numpy arrays (same format as save_clusters expects)
    clusters_list = [np.array(c, dtype=np.int64) for c in new_clusters]
    
    # Save in the same format as save_clusters function
    cluster_dict = {'clusters': clusters_list}
    np.save(output_path, cluster_dict, allow_pickle=True)
    
    if verbose:
        print("=" * 60)
        print("Adjusted Clusters Summary")
        print("=" * 60)
        for cluster_idx, cluster in enumerate(new_clusters):
            cluster_criteria = [all_criteria[int(idx)] for idx in cluster]
            crit_counts = Counter(cluster_criteria)
            total = len(cluster)
            print(f"\nCluster {cluster_idx}: {total} preferences")
            sorted_crit = sorted(crit_counts.items(), key=lambda x: x[1], reverse=True)
            for crit, count in sorted_crit:
                pct = count / total * 100 if total > 0 else 0
                print(f"  {crit}: {count} ({pct:.1f}%)")
        
        print(f"\nAdjusted clusters saved to: {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adjust clusters to make each cluster dominated by one criterion"
    )
    parser.add_argument(
        "--clusters_file",
        type=str,
        required=True,
        help="Path to current clusters .npy file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="notebooks/ultrafeedback_synthetic/mixed_criteria",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save adjusted clusters",
    )
    parser.add_argument(
        "--downsample_ratio",
        type=float,
        default=None,
        help="Downsample ratio used during precompute",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used during precompute",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples used during precompute",
    )
    parser.add_argument(
        "--min_dominance",
        type=float,
        default=0.6,
        help="Minimum fraction for dominant criterion (default: 0.6 = 60%%)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    
    args = parser.parse_args()
    
    adjust_clusters_by_criteria(
        clusters_path=args.clusters_file,
        dataset_path=args.dataset_path,
        output_path=args.output_file,
        downsample_ratio=args.downsample_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        min_dominance=args.min_dominance,
        verbose=args.verbose,
    )
