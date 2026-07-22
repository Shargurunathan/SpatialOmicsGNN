# Import os to work with folders and filenames
import os

# Import pandas to read and work with CSV files
import pandas as pd

# Import PyTorch to save the edge list
import torch

# Import NearestNeighbors to find spatially nearby patches
from sklearn.neighbors import NearestNeighbors


# -------------------------------------------------
# STEP 1: Load coordinates of all original patches
# -------------------------------------------------

# This CSV contains coordinates for all 23,220 patches
coordinates = pd.read_csv("outputs/patch_coordinates.csv")

print("Total coordinates before filtering:", len(coordinates))


# -------------------------------------------------
# STEP 2: Get only tissue patch filenames
# -------------------------------------------------

# Folder containing the 4,880 selected tissue patches
patch_folder = "outputs/tissue_patches"

# Get all tissue patch filenames and sort them numerically
#
# Example:
# patch_1.png
# patch_2.png
# patch_3.png
# ...
patch_files = sorted(
    os.listdir(patch_folder),
    key=lambda name: int(name.split("_")[1].split(".")[0])
)

print("Number of tissue patches:", len(patch_files))


# -------------------------------------------------
# STEP 3: Keep coordinates only for tissue patches
# -------------------------------------------------

# Select only rows whose Patch_Name exists in tissue_patches folder
tissue_coordinates = coordinates[
    coordinates["Patch_Name"].isin(patch_files)
].copy()


# -------------------------------------------------
# STEP 4: Make coordinate order match feature order
# -------------------------------------------------

# Feature extraction processes patches in the order of patch_files.
#
# Therefore, coordinate rows must follow exactly the same order.
tissue_coordinates = (
    tissue_coordinates
    .set_index("Patch_Name")
    .loc[patch_files]
    .reset_index()
)

print("Number of tissue coordinates:", len(tissue_coordinates))


# -------------------------------------------------
# STEP 5: Select X and Y positions
# -------------------------------------------------

points = tissue_coordinates[["X", "Y"]]


# -------------------------------------------------
# STEP 6: Create k-NN model
# -------------------------------------------------

# n_neighbors = 4 means:
#
# 1 neighbour = the patch itself
# 3 neighbours = three closest surrounding tissue patches
knn = NearestNeighbors(n_neighbors=4)


# Store all 4,880 tissue patch coordinates in k-NN
knn.fit(points)


# -------------------------------------------------
# STEP 7: Find nearest neighbours
# -------------------------------------------------

# distances → spatial distance to neighbouring patches
# indices   → node indices of neighbouring patches
distances, indices = knn.kneighbors(points)


# -------------------------------------------------
# STEP 8: Create graph edges
# -------------------------------------------------

edges = []

# Go through every tissue patch
for i in range(len(indices)):

    # Skip indices[i][0] because it is the node itself
    for j in indices[i][1:]:

        # Add connection:
        # current node i → neighbouring node j
        edges.append((int(i), int(j)))


# -------------------------------------------------
# STEP 9: Check the result
# -------------------------------------------------

print("\nTotal nodes:", len(tissue_coordinates))
print("Total edges:", len(edges))

print(
    "Minimum node index:",
    min(min(edge) for edge in edges)
)

print(
    "Maximum node index:",
    max(max(edge) for edge in edges)
)

print("First 10 edges:")
print(edges[:10])


# -------------------------------------------------
# STEP 10: Save corrected edges
# -------------------------------------------------

torch.save(edges, "outputs/graphs/edges.pt")

print("\nEdge list saved successfully!")