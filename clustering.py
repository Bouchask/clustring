import os
import numpy as np
import matplotlib.pyplot as plt
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
        self.pca = pca_2d
        self.k_range = range(2, 11)
        self.plots_dir = f"{results_dir}/plots"
        os.makedirs(self.plots_dir, exist_ok=True)

    # =====================
    # METRICS (AUTO K)
    # =====================
    def compute_metrics(self):
        report = []
        metrics = []

        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.X)

            sil = silhouette_score(self.X, labels)
            ch = calinski_harabasz_score(self.X, labels)
            db = davies_bouldin_score(self.X, labels)

            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(self.X)
            bic = gmm.bic(self.X)

            metrics.append((k, sil, ch, db, bic))
            report.append(
                f"K={k} | Silhouette={sil:.4f} | CH={ch:.2f} | DB={db:.4f} | BIC={bic:.2f}"
            )

        # Save TXT report
        with open("results/metrics_report.txt", "w") as f:
            f.write("=== CLUSTERING METRICS REPORT ===\n\n")
            f.write("\n".join(report))

        return metrics

    # =====================
    # BEST K (MATH RULE)
    # =====================
    def select_best_k(self, metrics):
        sil_k = max(metrics, key=lambda x: x[1])[0]
        ch_k = max(metrics, key=lambda x: x[2])[0]
        db_k = min(metrics, key=lambda x: x[3])[0]
        bic_k = min(metrics, key=lambda x: x[4])[0]

        best_k = max([sil_k, ch_k, db_k, bic_k], key=[sil_k, ch_k, db_k, bic_k].count)

        return best_k

    # =====================
    # 2D → 3D VISUAL (NO PCA CHANGE)
    # =====================
    def plot_3d_visual(self, labels, k):
        z = labels  # cluster index as depth (visual trick)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.pca[:, 0],
            self.pca[:, 1],
            z,
            c=labels
        )
        ax.set_title(f"3D Visual Clustering (K={k})")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("Cluster ID")

        plt.savefig(f"{self.plots_dir}/clusters_3d_K_{k}.png")
        plt.close()

    # =====================
    # APPLY FINAL SOLUTION
    # =====================
    def apply_best_solution(self, best_k):
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(self.X)

        # 2D plot
        plt.figure()
        plt.scatter(self.pca[:, 0], self.pca[:, 1], c=labels)
        plt.title(f"KMeans – PCA 2D (K={best_k})")
        plt.savefig(f"{self.plots_dir}/clusters_2d_K_{best_k}.png")
        plt.close()

        # 3D visual
        self.plot_3d_visual(labels, best_k)

        return labels
