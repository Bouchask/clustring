import os
import numpy as np
import matplotlib.pyplot as plt
# Only import 3D if needed, but safe to keep
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

class Clustering:
    def __init__(self, X, pca_2d, results_dir="results"):
        self.X = X
        self.pca = pca_2d # Can be None if PCA was rejected
        self.k_range = range(2, 11)
        self.plots_dir = f"{results_dir}/plots"
        os.makedirs(self.plots_dir, exist_ok=True)

    def compute_metrics(self):
        metrics = []
        report = ["=== CLUSTERING METRICS REPORT ===\n"]

        print(f"   Computing for K={self.k_range.start} to {self.k_range.stop-1}...")
        
        for k in self.k_range:
            # 1. K-Means
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.X)

            # 2. Metrics
            sil = silhouette_score(self.X, labels)
            ch = calinski_harabasz_score(self.X, labels)
            db = davies_bouldin_score(self.X, labels)

            # 3. BIC (GMM)
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(self.X)
            bic = gmm.bic(self.X)

            metrics.append((k, sil, ch, db, bic))
            line = f"K={k} | Silhouette={sil:.4f} | CH={ch:.2f} | DB={db:.4f} | BIC={bic:.2f}"
            report.append(line)
            
            # 4. Generate Plot ONLY if PCA exists
            if self.pca is not None:
                self.plot_3d_visual(labels, k)

        with open("results/metrics_report.txt", "w") as f:
            f.write("\n".join(report))
        
        return metrics

    def select_best_k(self, metrics):
        # Voting logic:
        # Silhouette -> Max
        # CH -> Max
        # DB -> Min
        # BIC -> Min
        
        sil_k = max(metrics, key=lambda x: x[1])[0]
        ch_k = max(metrics, key=lambda x: x[2])[0]
        db_k = min(metrics, key=lambda x: x[3])[0]
        bic_k = min(metrics, key=lambda x: x[4])[0]
        
        # Simple mode vote
        votes = [sil_k, ch_k, db_k, bic_k]
        best_k = max(votes, key=votes.count)
        
        return best_k

    def apply_best_solution(self, k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(self.X)
        return labels

    def plot_3d_visual(self, labels, k):
        # Safety check
        if self.pca is None:
            return

        z = labels 
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        
        # Use first 2 dims of PCA + Labels as Z
        sc = ax.scatter(
            self.pca[:, 0],
            self.pca[:, 1],
            z,
            c=labels,
            cmap='viridis',
            s=40
        )
        ax.set_title(f"Cluster Visual K={k} (PCA Reduced)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("Cluster Label")
        
        plt.savefig(f"{self.plots_dir}/clusters_3d_K_{k}.png")
        plt.close()