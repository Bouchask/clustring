import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# --- KMO & Bartlett Import Handling ---
try:
    from factor_analyzer.factor_analyzer import calculate_kmo
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
    HAS_FACTOR_ANALYZER = True
except ImportError:
    HAS_FACTOR_ANALYZER = False

class DataProcessor:
    def __init__(self, file_path, results_dir="results"):
        self.df = pd.read_csv(file_path)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.label = LabelEncoder()
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def preprocess(self):
        # 1. Clean Data
        df = self.df.dropna().drop_duplicates()

        # 2. Feature Engineering
        if "Charging Start Time" in df.columns:
            df["Charging Start Time"] = pd.to_datetime(df["Charging Start Time"])
            df["Charging End Time"] = pd.to_datetime(df["Charging End Time"])
            df["Duration_Sec"] = (
                df["Charging End Time"] - df["Charging Start Time"]
            ).dt.total_seconds()
            df.drop(columns=["Charging Start Time", "Charging End Time"], inplace=True)

        df.drop(columns=[c for c in ["User ID", "Charging Station ID"] if c in df.columns],
                inplace=True)

        # 3. Encoding
        obj_cols = df.select_dtypes(include="object").columns
        encoded_dfs = []
        for col in obj_cols:
            if df[col].nunique() <= 2:
                df[col] = self.label.fit_transform(df[col])
            else:
                enc = self.encoder.fit_transform(df[[col]])
                enc_df = pd.DataFrame(
                    enc,
                    columns=self.encoder.get_feature_names_out([col]),
                    index=df.index
                )
                encoded_dfs.append(enc_df)
                df.drop(columns=[col], inplace=True)
        
        if encoded_dfs:
            df = pd.concat([df] + encoded_dfs, axis=1)

        # 4. Scaling
        X_scaled = self.scaler.fit_transform(df)

        # 5. CHECK PCA SUITABILITY (The "Smart Switch")
        use_pca, X_final, X_vis = self._check_and_apply_pca(X_scaled)

        return df, X_final, X_vis, use_pca

    def _check_and_apply_pca(self, X_scaled):
        """
        Calculates metrics and decides whether to use PCA or Raw Data.
        Rule: If votes >= 2, use PCA. Else, use Raw.
        """
        metrics_file = os.path.join(self.results_dir, "preprocessing_metrics.txt")
        votes = 0
        report_lines = []
        
        report_lines.append("=== DATA SUITABILITY REPORT ===")
        
        # --- Metric 1: KMO ---
        kmo_val = 0
        if HAS_FACTOR_ANALYZER:
            try:
                _, kmo_val = calculate_kmo(X_scaled)
                if kmo_val > 0.5: # Threshold 0.5
                    votes += 1
                    report_lines.append(f"✅ KMO Test: {kmo_val:.3f} (Passed > 0.5)")
                else:
                    report_lines.append(f"❌ KMO Test: {kmo_val:.3f} (Failed < 0.5)")
            except:
                report_lines.append("⚠️ KMO Test: Error in calculation")
        else:
            report_lines.append("⚠️ KMO Test: Skipped (Lib missing)")

        # --- Metric 2: Bartlett ---
        if HAS_FACTOR_ANALYZER:
            try:
                _, p_val = calculate_bartlett_sphericity(X_scaled)
                if p_val < 0.05:
                    votes += 1
                    report_lines.append(f"✅ Bartlett Test: p={p_val:.4f} (Passed < 0.05)")
                else:
                    report_lines.append(f"❌ Bartlett Test: p={p_val:.4f} (Failed > 0.05)")
            except:
                report_lines.append("⚠️ Bartlett Test: Error")
        else:
            report_lines.append("⚠️ Bartlett Test: Skipped")

        # --- Metric 3: Explained Variance (2D Check) ---
        pca_check = PCA(n_components=2)
        pca_check.fit(X_scaled)
        var_2d = np.sum(pca_check.explained_variance_ratio_)
        # Threshold: If 2D explains > 30% of data, it's decent for PCA vis
        if var_2d > 0.30: 
            votes += 1
            report_lines.append(f"✅ Explained Var (2D): {var_2d:.1%} (Passed > 30%)")
        else:
            report_lines.append(f"❌ Explained Var (2D): {var_2d:.1%} (Failed < 30%)")

        # --- DECISION ---
        report_lines.append(f"\nTotal Votes for PCA: {votes}/3")
        
        X_final = X_scaled
        X_vis = None
        use_pca = False

        if votes >= 2:
            report_lines.append(">>> DECISION: ✅ PERFORM PCA (Clustering on Principal Components)")
            # Apply PCA for Clustering (Keep 95% variance)
            pca_full = PCA(n_components=0.95)
            X_final = pca_full.fit_transform(X_scaled)
            # Apply PCA for Vis (2 components)
            X_vis = X_final[:, :2] if X_final.shape[1] >= 2 else X_final
            use_pca = True
        else:
            report_lines.append(">>> DECISION: 🚫 SKIP PCA (Clustering on Original Scaled Data)")
            # X_final remains X_scaled
            # X_vis is None (No plotting)
            use_pca = False

        # Save Report
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
            
        print("\n".join(report_lines))
        return use_pca, X_final, X_vis