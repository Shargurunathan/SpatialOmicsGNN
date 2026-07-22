# Import PyTorch
import torch

# Import the PyTorch Geometric Data class
from torch_geometric.data import Data

# -----------------------------
# Load the extracted feature vectors
# -----------------------------
features = torch.load("outputs/features.pt")

# Convert the list of feature tensors into one tensor
# Shape: (4880, 2048, 1, 1)
x = torch.cat(features, dim=0).squeeze(-1).squeeze(-1)

# -----------------------------
# Load the real edge list
# -----------------------------
edges = torch.load("outputs/edges.pt")

# Convert edge list into a tensor
edge_index = torch.tensor(edges, dtype=torch.long)

# PyTorch Geometric expects shape (2, Number_of_Edges)
edge_index = edge_index.t().contiguous()

# -----------------------------
# Create the graph
# -----------------------------
graph = Data(
    x=x,
    edge_index=edge_index
)

# -----------------------------
# Print graph information
# -----------------------------
print(graph)
print()

print("Number of nodes :", graph.num_nodes)
print("Number of edges :", graph.num_edges)
print("Feature shape :", graph.x.shape)
print("Edge shape :", graph.edge_index.shape)
