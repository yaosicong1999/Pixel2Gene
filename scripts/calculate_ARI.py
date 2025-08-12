import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    adjusted_mutual_info_score, v_measure_score,
    fowlkes_mallows_score, confusion_matrix
)
from scipy.optimize import linear_sum_assignment
from collections import Counter

def merge_small_clusters(labels, min_size=500):
    """
    Merge clusters with fewer than `min_size` elements into the closest larger cluster (by label ID).

    Args:
        labels (np.ndarray): Original label array (1D), may include -1.
        min_size (int): Minimum allowed cluster size. Smaller ones will be merged.

    Returns:
        np.ndarray: New labels with small clusters merged.
    """
    labels = labels.copy()
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    # Find small clusters
    small_clusters = [c for c, size in cluster_sizes.items() if size < min_size]
    large_clusters = [c for c in cluster_sizes if c not in small_clusters]

    print(f"Merging {len(small_clusters)} small clusters into larger ones...")

    for small_c in small_clusters:
        # Replace all occurrences of the small cluster
        mask = labels == small_c

        # Choose the nearest larger cluster (you can improve this if you have spatial data)
        # Here we just choose the largest cluster
        target_cluster = max(large_clusters, key=lambda c: cluster_sizes[c])
        labels[mask] = target_cluster
        print(f"  Cluster {small_c} (size={cluster_sizes[small_c]}) → Cluster {target_cluster}")

    return labels

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_labels(dir_path):
    return np.array(load_pickle(f"{dir_path}/labels.pickle")).flatten()


def print_cluster_sizes(name, labels):
    print(f"\n{name} cluster sizes:")
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u:>3}: {c:>6}")
    print(f"  Total (excluding -1): {counts.sum()}")


def plot_confusion_matrix(y_true, y_pred, title=None, out_path=None):
    mask = (y_true != -1) & (y_pred != -1)
    cm = confusion_matrix(y_true[mask], y_pred[mask])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap="Blues", square=True, cbar=True,
                xticklabels=False, yticklabels=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


def hungarian_match_accuracy(y_true, y_pred):
    mask = (y_true != -1) & (y_pred != -1)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize match
    acc = cm[row_ind, col_ind].sum() / cm.sum()
    return acc


def evaluate_metrics(name1, labels1, name2, labels2, plot=False):
    mask = (labels1 != -1) & (labels2 != -1)
    l1, l2 = labels1[mask], labels2[mask]
    print(f"\n--- Comparing {name1} vs {name2} ---")
    print(f"Valid count: {len(l1)} / {len(labels1)}")

    ari = adjusted_rand_score(l1, l2)
    nmi = normalized_mutual_info_score(l1, l2)
    ami = adjusted_mutual_info_score(l1, l2)
    v = v_measure_score(l1, l2)
    fmi = fowlkes_mallows_score(l1, l2)
    h_acc = hungarian_match_accuracy(l1, l2)

    print(f"Adjusted Rand Index (ARI):           {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Adjusted Mutual Information (AMI):   {ami:.4f}")
    print(f"V-measure:                           {v:.4f}")
    print(f"Fowlkes-Mallows Index (FMI):         {fmi:.4f}")
    print(f"Hungarian-matched Accuracy:          {h_acc:.4f}")

    if plot:
        plot_confusion_matrix(l1, l2, out_path=f"{name1} vs {name2}.jpg")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare clustering label sets")
    parser.add_argument("--oTrue", required=True, help="Directory containing original true labels.pickle")
    parser.add_argument("--dTrue", required=True, help="Directory containing denoised/reference labels.pickle")
    parser.add_argument("--dPred", required=True, help="Directory containing predicted labels.pickle")
    parser.add_argument("--plot", action="store_true", help="Whether to plot confusion matrices")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    labels_oTrue = load_labels(args.oTrue)
    labels_oTrue= merge_small_clusters(labels_oTrue, min_size=500)
    labels_dTrue = load_labels(args.dTrue)
    labels_dTrue= merge_small_clusters(labels_dTrue, min_size=500)
    labels_dPred = load_labels(args.dPred)
    labels_dPred= merge_small_clusters(labels_dPred, min_size=500)

    print_cluster_sizes("oTrue", labels_oTrue)
    print_cluster_sizes("dTrue", labels_dTrue)
    print_cluster_sizes("dPred", labels_dPred)

    evaluate_metrics("oTrue", labels_oTrue, "dTrue", labels_dTrue, args.plot)
    evaluate_metrics("oTrue", labels_oTrue, "dPred", labels_dPred, args.plot)
