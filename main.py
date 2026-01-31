import os
from preprocissing import DataProcessor
from clustering import Clustering

class MasterPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        os.makedirs("results", exist_ok=True)

    def run(self):
        print("1️⃣ Preprocessing...")
        processor = DataProcessor(self.file_path)
        clean_df, X, pca_2d = processor.preprocess()

        print("2️⃣ Computing metrics...")
        cluster = Clustering(X, pca_2d)
        metrics = cluster.compute_metrics()

        best_k = cluster.select_best_k(metrics)
        print(f"✅ Best K selected mathematically: {best_k}")

        print("3️⃣ Applying best solution...")
        clean_df["Cluster"] = cluster.apply_best_solution(best_k)

        clean_df.to_csv("results/final_clusters.csv", index=False)
        print("🎓 DONE — MASTER LEVEL OUTPUT READY")

if __name__ == "__main__":
    MasterPipeline(r"C:\Users\VoxxF\Desktop\Projet\clustring\data\raw\ev_charging_patterns.csv").run()
