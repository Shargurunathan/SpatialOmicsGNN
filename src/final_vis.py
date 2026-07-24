import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==========================================================
# CONFIG — adjust these paths/names to match your files
# ==========================================================
GCN_FEATURES_PATH = "outputs/features/gcn_output.pt"     
COORDS_PATH = "outputs/patch_coordinates.csv"
PATCH_FOLDER = "outputs/tissue_patches"
EDGES_PATH = "outputs/graphs/edges.pt"
N_CLUSTERS = 5          # try 4-8, tune by eye
MARGIN = 512             # same border trim as vis_graph_fixed.py
MIN_DEGREE = 2

# ==========================================================
# STEP 1: Load GCN output embeddings [num_patches, 128]
# ==========================================================
gcn_features = torch.load(GCN_FEATURES_PATH)
if isinstance(gcn_features, torch.Tensor):
    gcn_features = gcn_features.detach().cpu().numpy()

print("GCN embedding shape:", gcn_features.shape)

# ==========================================================
# STEP 2: Load + align coordinates (same logic as vis_graph_fixed.py)
# ==========================================================
coordinates = pd.read_csv(COORDS_PATH)

patch_files = sorted(
    os.listdir(PATCH_FOLDER),
    key=lambda name: int(name.split("_")[1].split(".")[0])
)

tissue_coordinates = coordinates[
    coordinates["Patch_Name"].isin(patch_files)
].copy()

tissue_coordinates = (
    tissue_coordinates
    .set_index("Patch_Name")
    .loc[patch_files]
    .reset_index()
)

assert len(tissue_coordinates) == gcn_features.shape[0], (
    "Mismatch: number of coordinates does not match number of GCN "
    "embeddings. Make sure GCN_FEATURES_PATH corresponds to the same "
    "patch order as patch_coordinates.csv / tissue_patches folder."
)

# ==========================================================
# STEP 3: Same border trim + low-degree cleanup as before
# ==========================================================
x_min, x_max = tissue_coordinates["X"].min(), tissue_coordinates["X"].max()
y_min, y_max = tissue_coordinates["Y"].min(), tissue_coordinates["Y"].max()

in_bounds = (
    (tissue_coordinates["X"] > x_min + MARGIN) &
    (tissue_coordinates["X"] < x_max - MARGIN) &
    (tissue_coordinates["Y"] > y_min + MARGIN) &
    (tissue_coordinates["Y"] < y_max - MARGIN)
).values

edges = torch.load(EDGES_PATH)
G_full = nx.Graph()
G_full.add_edges_from(edges)

keep_ids = np.where(in_bounds)[0].tolist()
H = G_full.subgraph(keep_ids).copy()

low_degree_nodes = [n for n, d in H.degree() if d < MIN_DEGREE]
H.remove_nodes_from(low_degree_nodes)

final_ids = sorted(H.nodes())
print("Final patches after cleanup:", len(final_ids))

# Keep only cleaned patches for both coordinates and embeddings
tissue_coordinates_clean = tissue_coordinates.iloc[final_ids].reset_index(drop=True)
gcn_features_clean = gcn_features[final_ids]

# ==========================================================
# STEP 4: Cluster the 128-dim embeddings with KMeans
# ==========================================================
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(gcn_features_clean)

# ==========================================================
# STEP 5: Reduce 128-dim -> 2D with PCA (for the embedding-space plot)
# ==========================================================
pca = PCA(n_components=2, random_state=42)
embedding_2d = pca.fit_transform(gcn_features_clean)

print("PCA explained variance ratio:", pca.explained_variance_ratio_)

# ==========================================================
# STEP 6: Plot side by side
#   Left  = clusters shown in PCA embedding space
#   Right = SAME clusters shown at their real spatial position on the WSI
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# --- Left: PCA embedding space ---
scatter1 = axes[0].scatter(
    embedding_2d[:, 0], embedding_2d[:, 1],
    c=cluster_labels, cmap="tab10", s=8, alpha=0.8
)
axes[0].set_title("GCN Embeddings — PCA Projection (colored by cluster)")
axes[0].set_xlabel("PC 1")
axes[0].set_ylabel("PC 2")

# --- Right: spatial layout on the slide ---
scatter2 = axes[1].scatter(
    tissue_coordinates_clean["X"], -tissue_coordinates_clean["Y"],
    c=cluster_labels, cmap="tab10", s=8, alpha=0.8
)
axes[1].set_title("Same Clusters — Mapped Back to Tissue Location")
axes[1].set_xlabel("X (WSI coordinate)")
axes[1].set_ylabel("Y (WSI coordinate)")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.savefig("outputs/graphs/gcn_embedding_clusters.png", dpi=200, bbox_inches="tight")
plt.show()

print("\nSaved final output image to outputs/gcn_embedding_clusters.png")
print("If regions on the right (spatial) form clean, contiguous colored ")
print("patches instead of random speckling, that shows the GCN embeddings ")
print("captured real spatial/tissue structure.")