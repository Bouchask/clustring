import os
from preprocissing import DataProcessor
from clustering import Clustering

class MasterPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self):
        print("1️⃣ Preprocessing & Suitability Check...")
        processor = DataProcessor(self.file_path, results_dir=self.results_dir)
        
        # Now returns a flag 'use_pca'
        clean_df, X_final, X_vis, use_pca = processor.preprocess()
        
        if use_pca:
            print("   ✅ DECISION: PCA Applied. Clustering on Compressed Data.")
        else:
            print("   🚫 DECISION: PCA Rejected. Clustering on Raw Data.")

        print("2️⃣ Computing Metrics...")
        cluster = Clustering(X_final, X_vis, results_dir=self.results_dir)
        metrics = cluster.compute_metrics()

        best_k = cluster.select_best_k(metrics)
        print(f"✅ Best K selected: {best_k}")

        print("3️⃣ Finalizing...")
        clean_df["Cluster"] = cluster.apply_best_solution(best_k)

        output_path = os.path.join(self.results_dir, "final_clusters.csv")
        clean_df.to_csv(output_path, index=False)
        print("🎓 DONE — Results in 'results/' folder")

if __name__ == "__main__":
    # Update this path to your file location
    file_path = r"C:\Users\VoxxF\Desktop\Projet\clustring\data\raw\ev_charging_patterns.csv"
    MasterPipeline(file_path).run()