import os
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------------
# Load all original patch coordinates
# ----------------------------------
coordinates = pd.read_csv("outputs/patch_coordinates.csv")

# ----------------------------------
# Get tissue patch filenames
# ----------------------------------
patch_folder = "outputs/tissue_patches"

patch_files = sorted(
    os.listdir(patch_folder),
    key=lambda name: int(name.split("_")[1].split(".")[0])
)

# ----------------------------------
# Keep coordinates only for tissue patches
# ----------------------------------
tissue_coordinates = coordinates[
    coordinates["Patch_Name"].isin(patch_files)
].copy()

# Make coordinate order exactly match feature extraction / edge creation
tissue_coordinates = (
    tissue_coordinates
    .set_index("Patch_Name")
    .loc[patch_files]
    .reset_index()
)

# ----------------------------------
# STEP 1: Drop border/margin patches
# ----------------------------------
# Scanner border strips sit right at the max X or max Y edge of the
# scanned region. We trim a margin (in pixels) from all 4 sides.
margin = 512  # 2 patch-widths; increase if border still shows up

x_min, x_max = tissue_coordinates["X"].min(), tissue_coordinates["X"].max()
y_min, y_max = tissue_coordinates["Y"].min(), tissue_coordinates["Y"].max()

in_bounds = (
    (tissue_coordinates["X"] > x_min + margin) &
    (tissue_coordinates["X"] < x_max - margin) &
    (tissue_coordinates["Y"] > y_min + margin) &
    (tissue_coordinates["Y"] < y_max - margin)
)

tissue_coordinates = tissue_coordinates[in_bounds].reset_index(drop=True)
print("Patches remaining after border trim:", len(tissue_coordinates))

# Keep only the patch files that survived the border trim
kept_files = set(tissue_coordinates["Patch_Name"])

# ----------------------------------
# Load tissue-only graph edges
# ----------------------------------
edges = torch.load("outputs/graphs/edges.pt")

G_full = nx.Graph()
G_full.add_edges_from(edges)

# Node positions for ALL original tissue patches (needed to look up old IDs)
# original patch_files list still has the full un-trimmed order
pos_full = {}
for i, name in enumerate(patch_files):
    row = coordinates[coordinates["Patch_Name"] == name].iloc[0]
    pos_full[i] = (row["X"], -row["Y"])

# Map old node index -> keep or drop, based on border trim
keep_ids = [i for i, name in enumerate(patch_files) if name in kept_files]

H = G_full.subgraph(keep_ids).copy()

# ----------------------------------
# STEP 2: Drop isolated / low-degree nodes
# ----------------------------------
# Real tissue patches have several spatial neighbors. Stray artifacts
# usually end up with degree 0 or 1. Remove those.
min_degree = 2
low_degree_nodes = [n for n, d in H.degree() if d < min_degree]
H.remove_nodes_from(low_degree_nodes)

print("Nodes after removing low-degree/isolated points:", H.number_of_nodes())
print("Edges remaining:", H.number_of_edges())

pos = {n: pos_full[n] for n in H.nodes()}

# ----------------------------------
# Draw the cleaned graph
# ----------------------------------
plt.figure(figsize=(12, 12))

nx.draw(
    H,
    pos=pos,
    node_size=3,
    node_color="red",
    edge_color="gray",
    width=0.3,
    alpha=0.6,
    with_labels=False
)

plt.title("Spatial Graph of Tissue Patches (CMU-1) — Cleaned")
plt.gca().set_aspect("equal")
plt.savefig("outputs/graphs/graph_cleaned.png", dpi=200, bbox_inches="tight")
plt.show()