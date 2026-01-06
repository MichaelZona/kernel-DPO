"""
Analyze clustering results to check if preferences with same criteria are grouped together.

This script:
1. Loads the clusters from clustering results
2. Loads the original dataset to get criterion_used labels
3. Analyzes the distribution of criteria in each cluster
4. Reports how well the clustering matches the criteria
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


def infer_criterion_from_scores(
    dataset_path: str,
    indices: np.ndarray,
    seed: int = 42,
) -> list:
    """
    Infer which criterion was used for each preference by comparing score differences.
    
    For mixed_criteria dataset, we can infer the criterion by:
    1. Loading the original CSV data (if available)
    2. Or comparing which criterion has the largest score difference that matches the preference
    
    Args:
        dataset_path: Path to the dataset directory
        indices: Array of data point indices
        seed: Random seed used when creating the dataset
        
    Returns:
        List of inferred criteria for each index
    """
    # Try to load the dataset and check if criterion_used exists
    try:
        train_ds = load_from_disk(os.path.join(dataset_path, "train"))
        if "criterion_used" in train_ds.column_names:
            # Dataset has criterion_used field, use it directly
            criteria = [train_ds[int(idx)]["criterion_used"] for idx in indices]
            return criteria
    except Exception as e:
        print(f"Warning: Could not load dataset with criterion_used: {e}")
    
    # Fallback: Try to load from CSV if available
    # This would require the original CSV files with score columns
    print("Warning: criterion_used not found in dataset. Cannot infer criteria without original scores.")
    return None


def analyze_cluster_criteria(
    clusters_path: str,
    dataset_path: str,
    downsample_ratio: float = None,
    seed: int = 42,
    max_samples: int = None,
    verbose: bool = True,
) -> dict:
    """
    Analyze how well clusters match criteria.
    
    Args:
        clusters_path: Path to the clusters .npy file
        dataset_path: Path to the dataset directory (e.g., notebooks/ultrafeedback_synthetic/mixed_criteria)
        downsample_ratio: Downsample ratio used during precompute (to match indices)
        seed: Random seed used during precompute (to match shuffle)
        max_samples: Maximum samples used during precompute
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with analysis results
    """
    # Load clusters
    clusters = load_clusters(clusters_path)
    
    if verbose:
        print("=" * 60)
        print("Cluster-Criteria Analysis")
        print("=" * 60)
        print(f"Loaded {len(clusters)} clusters")
        print(f"Total preferences: {sum(len(c) for c in clusters)}")
        print()
    
    # Load dataset to get criterion_used
    try:
        train_ds = load_from_disk(os.path.join(dataset_path, "train"))
        
        if "criterion_used" not in train_ds.column_names:
            if verbose:
                print("Warning: Dataset does not have 'criterion_used' field.")
                print("Trying to infer from score differences...")
            
            # Try to infer from original CSV
            # This requires access to the original CSV files with score columns
            criteria_list = None
        else:
            # Reapply the same downsample and shuffle operations as in precompute
            # to match the indices used in clusters
            original_size = len(train_ds)
            
            if downsample_ratio is not None and downsample_ratio < 1.0:
                # Apply same shuffle and downsample as in data_loader
                # data_loader does: shuffle(seed) -> select(range(int(len * ratio)))
                train_ds = train_ds.shuffle(seed=seed)
                n_samples = int(len(train_ds) * downsample_ratio)
                train_ds = train_ds.select(range(n_samples))
                
                if verbose:
                    print(f"Applied downsample (ratio={downsample_ratio}, seed={seed})")
                    print(f"  Original size: {original_size}")
                    print(f"  Downsampled size: {len(train_ds)}")
            
            # Apply max_samples limit if specified (applied after downsample, same as precompute)
            if max_samples is not None and max_samples < len(train_ds):
                train_ds = train_ds.select(range(max_samples))
                if verbose:
                    print(f"Applied max_samples limit: {max_samples}")
                    print(f"  Final size: {len(train_ds)}")
            
            all_criteria = train_ds["criterion_used"]
            
            if verbose:
                print(f"Dataset size after processing: {len(all_criteria)}")
                print(f"Total indices in clusters: {sum(len(c) for c in clusters)}")
            
            # Get criteria for each cluster
            criteria_list = []
            for cluster in clusters:
                for idx in cluster:
                    idx_int = int(idx)
                    if idx_int < len(all_criteria):
                        criteria_list.append(all_criteria[idx_int])
                    else:
                        if verbose:
                            print(f"Warning: Index {idx_int} out of range for processed dataset (size {len(all_criteria)})")
                        criteria_list.append(None)
            
    except Exception as e:
        if verbose:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
        criteria_list = None
    
    if criteria_list is None:
        if verbose:
            print("\nCannot analyze criteria distribution without criterion_used or original scores.")
            print("Please ensure the dataset has 'criterion_used' field or provide original CSV files.")
        return None
    
    # Analyze each cluster
    cluster_criteria_dist = []
    cluster_purity = []
    
    if verbose:
        print("=" * 60)
        print("Cluster-wise Criteria Distribution")
        print("=" * 60)
        print()
    
    for cluster_idx, cluster in enumerate(clusters):
        # Get criteria for this cluster
        start_idx = sum(len(c) for c in clusters[:cluster_idx])
        cluster_criteria = [criteria_list[start_idx + i] for i in range(len(cluster))]
        # Filter out None values
        cluster_criteria = [c for c in cluster_criteria if c is not None]
        criterion_counts = Counter(cluster_criteria)
        
        cluster_criteria_dist.append(dict(criterion_counts))
        
        # Calculate purity: fraction of most common criterion
        if len(cluster_criteria) > 0:
            most_common_count = max(criterion_counts.values())
            purity = most_common_count / len(cluster_criteria)
            cluster_purity.append(purity)
        else:
            cluster_purity.append(0.0)
        
        if verbose:
            print(f"Cluster {cluster_idx}: {len(cluster)} preferences")
            print("-" * 60)
            print("Criteria Distribution:")
            # Sort by count (descending) for better readability
            sorted_criteria = sorted(criterion_counts.items(), key=lambda x: x[1], reverse=True)
            for crit, count in sorted_criteria:
                pct = count / len(cluster_criteria) * 100 if len(cluster_criteria) > 0 else 0
                bar_length = int(pct / 2)  # Scale bar to fit in 50 chars
                bar = "█" * bar_length
                print(f"  {crit:25s}: {count:3d} ({pct:5.1f}%) {bar}")
            
            # Show missing criteria
            missing = [c for c in CRITERIA if c not in criterion_counts]
            if missing:
                print(f"  Missing criteria: {', '.join(missing)}")
            
            print(f"\n  Purity: {purity:.2%} (most common: {criterion_counts.most_common(1)[0][0] if criterion_counts else 'N/A'})")
            print()
    
    # Overall statistics
    if verbose:
        print("=" * 60)
        print("Overall Statistics")
        print("=" * 60)
        print(f"Average cluster purity: {np.mean(cluster_purity):.2%}")
        print(f"Min cluster purity: {np.min(cluster_purity):.2%}")
        print(f"Max cluster purity: {np.max(cluster_purity):.2%}")
        print()
    
    return {
        "cluster_criteria_dist": cluster_criteria_dist,
        "cluster_purity": cluster_purity,
        "average_purity": np.mean(cluster_purity),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze clustering results to check criteria distribution"
    )
    parser.add_argument(
        "--clusters_file",
        type=str,
        required=True,
        help="Path to clusters .npy file (e.g., pre_compute/clusters_ultrafeedback_llama1b_1000samples.npy)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="notebooks/ultrafeedback_synthetic/mixed_criteria",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed analysis",
    )
    parser.add_argument(
        "--downsample_ratio",
        type=float,
        default=None,
        help="Downsample ratio used during precompute (to match indices). If None, assumes dataset was not downsampled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used during precompute (to match shuffle)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples used during precompute (to match indices)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Save detailed results to a text file",
    )
    
    args = parser.parse_args()
    
    # Analyze
    results = analyze_cluster_criteria(
        clusters_path=args.clusters_file,
        dataset_path=args.dataset_path,
        downsample_ratio=args.downsample_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )
    
    if results:
        print("=" * 60)
        print("Analysis complete!")
        print("=" * 60)
        
        # Save to file if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("Cluster-Criteria Analysis Results\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Clusters file: {args.clusters_file}\n")
                f.write(f"Dataset path: {args.dataset_path}\n")
                f.write(f"Total clusters: {len(results['cluster_criteria_dist'])}\n")
                f.write(f"Total preferences: {sum(sum(d.values()) for d in results['cluster_criteria_dist'])}\n")
                f.write("\n")
                
                # Write cluster-wise distribution
                f.write("=" * 60 + "\n")
                f.write("Cluster-wise Criteria Distribution\n")
                f.write("=" * 60 + "\n\n")
                
                for cluster_idx, (dist, purity) in enumerate(zip(results['cluster_criteria_dist'], results['cluster_purity'])):
                    total = sum(dist.values())
                    f.write(f"Cluster {cluster_idx}: {total} preferences\n")
                    f.write("-" * 60 + "\n")
                    f.write("Criteria Distribution:\n")
                    
                    sorted_criteria = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                    for crit, count in sorted_criteria:
                        pct = count / total * 100 if total > 0 else 0
                        f.write(f"  {crit:25s}: {count:3d} ({pct:5.1f}%)\n")
                    
                    missing = [c for c in CRITERIA if c not in dist]
                    if missing:
                        f.write(f"  Missing criteria: {', '.join(missing)}\n")
                    
                    most_common = max(dist.items(), key=lambda x: x[1])[0] if dist else 'N/A'
                    f.write(f"\n  Purity: {purity:.2%} (most common: {most_common})\n")
                    f.write("\n")
                
                # Write overall statistics
                f.write("=" * 60 + "\n")
                f.write("Overall Statistics\n")
                f.write("=" * 60 + "\n")
                f.write(f"Average cluster purity: {results['average_purity']:.2%}\n")
                f.write(f"Min cluster purity: {min(results['cluster_purity']):.2%}\n")
                f.write(f"Max cluster purity: {max(results['cluster_purity']):.2%}\n")
                f.write("\n")
                
                # Summary table: Cluster-wise criteria distribution
                f.write("=" * 60 + "\n")
                f.write("Summary: Cluster-wise Criteria Distribution\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"{'Cluster':<10} {'Total':<8} " + " ".join([f"{c[:8]:<10}" for c in CRITERIA]) + " Purity\n")
                f.write("-" * 60 + "\n")
                for cluster_idx, (dist, purity) in enumerate(zip(results['cluster_criteria_dist'], results['cluster_purity'])):
                    total = sum(dist.values())
                    crit_counts = [dist.get(c, 0) for c in CRITERIA]
                    crit_pcts = [f"{dist.get(c, 0)/total*100:.1f}%" if total > 0 else "0.0%" for c in CRITERIA]
                    f.write(f"{cluster_idx:<10} {total:<8} " + " ".join([f"{cnt:<10}" for cnt in crit_counts]) + f" {purity:.2%}\n")
                f.write("\n")
            
            print(f"\nResults saved to: {args.output_file}")