import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# --- CONFIGURATION ---
INPUT_FILE = 'results/final_clusters.csv'
OUTPUT_DIR = 'results/plot'

# 1. Setup Environment
# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📂 Saving plots to: {os.path.abspath(OUTPUT_DIR)}")

# 2. Load Data
try:
    df = pd.read_csv(INPUT_FILE)
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Could not find {INPUT_FILE}. Please check the path.")
    exit()

# 3. Calculate Economic Profiles (Centroids)
# This helps understand what each cluster represents financially
centroids = df.groupby('Cluster')[['Duration_Sec', 'Energy Consumed (kWh)', 'Charging Cost (USD)']].mean()
centroids['Duration_Hours'] = centroids['Duration_Sec'] / 3600
print("\n=== CLUSTER PROFILES (CENTROIDS) ===")
print(centroids[['Duration_Hours', 'Energy Consumed (kWh)', 'Charging Cost (USD)']].round(2))

# 4. Initialize Plotting Style (Dark Mode for Professional Look)
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 9))

# ==========================================
# PLOT 1: ECONOMIC REALITY (2D)
# Business View: Time vs. Energy
# ==========================================
ax1 = fig.add_subplot(121)

# Using Seaborn for the 2D plot (Better handling of legends)
sns.scatterplot(
    data=df,
    x='Charging Duration (hours)',
    y='Energy Consumed (kWh)',
    hue='Cluster',
    palette='tab10',  # 'tab10' is perfect for distinct clusters (up to 10)
    s=80,             # Marker size
    alpha=0.8,        # Transparency
    edgecolor='w',    # White edge for contrast
    ax=ax1
)

ax1.set_title('ECONOMIC SEGMENTATION\n(Time Occupied vs. Energy Sold)', fontsize=14, color='cyan', pad=15)
ax1.set_xlabel('Duration (Hours)', fontsize=12, color='white')
ax1.set_ylabel('Energy Consumed (kWh)', fontsize=12, color='white')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(title='Cluster ID', bbox_to_anchor=(1.02, 1), loc='upper left')

# ==========================================
# PLOT 2: MATHEMATICAL SEPARATION (3D PCA)
# Technical View: How the algorithm sees the data
# ==========================================
# We must re-calculate PCA on the numeric features to visualize the 38-dim space in 3D
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
# Exclude the 'Cluster' column itself from the features!
features = df[numeric_cols].drop(columns=['Cluster'], errors='ignore')

# Standardize & Reduce
X_scaled = StandardScaler().fit_transform(features)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

ax2 = fig.add_subplot(122, projection='3d')

scatter_3d = ax2.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    X_pca[:, 2],
    c=df['Cluster'],
    cmap='tab10',  # Must match the 2D palette
    s=50,
    alpha=0.8,
    edgecolor='k'
)

ax2.set_title('ALGORITHMIC SEPARATION\n(3D Principal Component Analysis)', fontsize=14, color='cyan', pad=15)
ax2.set_xlabel('PC 1 (Variance)', fontsize=10)
ax2.set_ylabel('PC 2', fontsize=10)
ax2.set_zlabel('PC 3', fontsize=10)

# Add colorbar for 3D plot
cbar = plt.colorbar(scatter_3d, ax=ax2, pad=0.1, shrink=0.7)
cbar.set_label('Cluster ID')

# ==========================================
# SAVE & SHOW
# ==========================================
plt.tight_layout()

# Save the combined dashboard
save_path = os.path.join(OUTPUT_DIR, 'cluster_analysis_dashboard.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Dashboard saved to: {save_path}")

# Optional: Save individual plots if needed
# (You can comment this out if you only want the combined one)
fig2 = plt.figure(figsize=(10, 8))
ax_iso = fig2.add_subplot(111, projection='3d')
ax_iso.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=df['Cluster'], cmap='tab10', s=50)
ax_iso.set_title(f'Isolated 3D View (K={df["Cluster"].nunique()})')
fig2.savefig(os.path.join(OUTPUT_DIR, 'isolated_3d_pca.png'), dpi=300)
print(f"✅ Isolated 3D plot saved.")

plt.show()