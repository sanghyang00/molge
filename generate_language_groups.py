"""Cluster languages into expert groups from URIEL distances or centroids.

Usage:
  python generate_language_groups.py \
      --matrice_dir metadata/feature_matrices_noimpute \
      --language_centroid_dir metadata/language_centroids
"""

import os, json, torch, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from collections import Counter
from tqdm import tqdm
from preprocess.data_merging import code_to_language

import warnings
warnings.filterwarnings('ignore')

def get_embedding_from_dict(vectors):
    placeholder = []
    languages = []
    for k, v in vectors['prototypes'].items():
        languages.append(k)
        placeholder.append(v)
    embedding = torch.stack(placeholder)
    return embedding, languages


def convert_dist_to_affinity(dist_matrix, sigma=None, epsilon=1e-8):
    """
    Angular Distance Matrix -> Affinity Matrix (RBF Kernel)
    """
    N = dist_matrix.shape[0]
    
    if sigma is None:
        mask = ~np.eye(N, dtype=bool) & (dist_matrix > epsilon) & (~np.isnan(dist_matrix))
        valid_distances = dist_matrix[mask]
        
        valid_distances = valid_distances[valid_distances < 0.99]

        if valid_distances.size == 0:
            sigma = 1.0 
            # print("[Warning] No valid distances for sigma calculation. Using sigma=1.0")
        else:
            sigma = np.median(valid_distances)
            
        # print(f"[Sigma Auto] Based on {valid_distances.size} pairs. Sigma (Median): {sigma:.4f}")

    dist_sq = dist_matrix ** 2
    affinity_matrix = np.exp(-dist_sq / (2 * sigma ** 2))
    
    np.fill_diagonal(affinity_matrix, 0)
    
    affinity_matrix[np.isnan(dist_matrix)] = 0
    
    return affinity_matrix

def cluster_data_from_embedding(
    X,
    method="km",
    n_clusters=8,
    normalize=True,
    random_state=42,
    **kwargs,
):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D of shape (L, F)")

    if normalize:
        X_proc = Normalizer(norm='l2').fit_transform(X)
    else:
        X_proc = X

    method = method.lower()
    
    if method == "km":
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state, **kwargs)
    elif method == "ahc":
        model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    elif method == "sc":
        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=random_state,
            assign_labels="kmeans",
            affinity='nearest_neighbors',
            **kwargs,
        )
    else:
        raise ValueError("method must be one of: km, ahc, sc")

    labels = model.fit_predict(X_proc)

    if n_clusters > 1 and len(np.unique(labels)) > 1:
        score = silhouette_score(X_proc, labels, metric='cosine')
    else:
        score = -1.0

    return labels, score


def cluster_data_from_matrix(
    distance_mat,
    method="ahc",        # one of: "ahc", "sc"
    n_clusters=8,
    random_state=42,
    **kwargs,               # passed through to the chosen estimator
):
    """ """
    distance_mat = np.asarray(distance_mat)
    if distance_mat.ndim != 2 or distance_mat.shape[0] != distance_mat.shape[1]:
        raise ValueError("distance_mat must be 2D square matrix of shape (L, L)")
    
    L = distance_mat.shape[0]
    
    method = method.lower()
    if method == "ahc":
        linkage = kwargs.pop('linkage', 'complete')
        ahc = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='precomputed', 
            linkage=linkage
        )
        labels = ahc.fit_predict(distance_mat)
    elif method == "sc":
        sigma = kwargs.pop('sigma', None)
        sc = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed', 
            random_state=random_state,
            assign_labels='kmeans'
        )
        affinity_mat = convert_dist_to_affinity(distance_mat)
        labels = sc.fit_predict(affinity_mat)
    else:
        raise ValueError("method must be one of: 'ahc', 'sc'")
    
    if n_clusters > 1 and len(np.unique(labels)) > 1:
        from sklearn.metrics import silhouette_score
        score = silhouette_score(distance_mat, labels, metric='precomputed')
    else:
        score = -1.0
    
    return labels, score


def search_best_k(
    data,
    method='km', 
    min_k=8, 
    max_k=32, 
    normalize=True,
    random_state=42, 
    plot=True, 
    figsize=(12, 4), 
    **kwargs
):
    """ """
    data = np.asarray(data)
    
    if data.ndim != 2:
        raise ValueError("data must be 2D array")
    
    is_matrix = (data.shape[0] == data.shape[1])
    
    if is_matrix:
        if method.lower() not in ['ahc', 'sc']:
            raise ValueError("For distance matrix, method must be 'ahc' or 'sc'")
        data_type = "matrix"
        cluster_func = cluster_data_from_matrix
    else:
        if method.lower() not in ['km', 'ahc', 'sc']:
            raise ValueError("For embedding, method must be 'km', 'ahc', or 'sc'")
        data_type = "embedding"
        cluster_func = cluster_data_from_embedding
    
    scores = {}
    best_score = -999
    best_k = -1
    best_labels = None
    
    cached_affinity_mat = None
    if data_type == "matrix" and method.lower() == "sc":
        sigma = kwargs.get('sigma', None)
        print(f"[SC + Matrix] Computing affinity matrix (sigma={'auto' if sigma is None else f'{sigma:.4f}'})...")
        cached_affinity_mat = convert_dist_to_affinity(data, sigma=sigma)
        print(f"[SC + Matrix] Affinity matrix computed (shape: {cached_affinity_mat.shape})")
        kwargs = {k: v for k, v in kwargs.items() if k != 'sigma'}
    
    print(f"Searching best K from {min_k} to {max_k} using {method.upper()} ({data_type})...")
    
    for k in tqdm(range(min_k, max_k+1)):
        if data_type == "embedding":
            labels, score = cluster_func(
                data, 
                method=method, 
                n_clusters=k,
                normalize=normalize,
                random_state=random_state,
                **kwargs
            )
        else:  # matrix
            if method.lower() == "sc" and cached_affinity_mat is not None:
                
                sc = SpectralClustering(
                    n_clusters=k,
                    affinity='precomputed',
                    random_state=random_state,
                    assign_labels='kmeans'
                )
                labels = sc.fit_predict(cached_affinity_mat)
                
                if k > 1 and len(np.unique(labels)) > 1:
                    score = silhouette_score(data, labels, metric='precomputed')
                else:
                    score = -1.0
            else:
                labels, score = cluster_func(
                    data, 
                    method=method, 
                    n_clusters=k,
                    random_state=random_state,
                    **kwargs
                )
        
        scores[k] = score
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
        
        # print(f"  K={k:2d}: Silhouette Score = {score:.4f}")
    
    print(f"\nBest K: {best_k} (Silhouette Score: {best_score:.4f})")
    
    if plot:
        plt.figure(figsize=figsize)
        k_values = list(scores.keys())
        score_values = list(scores.values())
        
        plt.plot(k_values, score_values, marker='o', linewidth=2, markersize=8)
        plt.axvline(x=best_k, color='r', linestyle='--', linewidth=2, 
                   label=f'Best K={best_k} (Score={best_score:.4f})')
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title(f'Silhouette Score vs Number of Clusters ({method.upper()}, {data_type})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    return best_k, best_score, scores, best_labels


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrice_dir', type=str, default='metadata/feature_matrices_noimpute')
    parser.add_argument('--language_centroid_dir', type=str, default='metadata/language_centroids')
    args = parser.parse_args()
    
    eps = 1e-8
    matrice_dir = args.matrice_dir

    languages_geographic = np.load(os.path.join(matrice_dir, f'geographic_languages.npy'))
    distance_mat_geographic = np.load(os.path.join(matrice_dir, f'geographic_dist.npy'))
    mask_geographic = ~np.eye(distance_mat_geographic.shape[0], dtype=bool)
    distance_mat_geographic[mask_geographic & (distance_mat_geographic == 0)] = eps

    languages_genetic = np.load(os.path.join(matrice_dir, f'genetic_languages.npy'))
    distance_mat_genetic = np.load(os.path.join(matrice_dir, f'genetic_dist.npy'))
    mask_genetic = ~np.eye(distance_mat_genetic.shape[0], dtype=bool)
    distance_mat_genetic[mask_genetic & (distance_mat_genetic == 0)] = eps

if __name__ == "__main__":
    main()
