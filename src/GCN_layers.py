import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ----------------------------------
# Load extracted ResNet50 features
# ----------------------------------
features = torch.load("outputs/features/features.pt")

# Combine all patch features
# Before: list of tensors, each shape [1, 2048, 1, 1]
# After: [4880, 2048]
x = torch.cat(features, dim=0).squeeze(-1).squeeze(-1)


# ----------------------------------
# Load graph edges
# ----------------------------------
edges = torch.load("outputs/features/edges.pt")

# Convert edges into tensor
edge_index = torch.tensor(edges, dtype=torch.long)

# Convert shape:
# [69660, 2] → [2, 69660]
edge_index = edge_index.t().contiguous()


# ----------------------------------
# Create PyTorch Geometric graph
# ----------------------------------
graph = Data(
    x=x,
    edge_index=edge_index
)


# ----------------------------------
# Define the GCN model
# ----------------------------------
class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # First GCN layer
        # 2048 input features → 512 output features
        self.conv1 = GCNConv(2048, 512)

        # Second GCN layer
        # 512 input features → 128 output features
        self.conv2 = GCNConv(512, 128)


    def forward(self, data):

        # Get node features and edges
        x = data.x
        edge_index = data.edge_index

        # First GCN layer
        x = self.conv1(x, edge_index)

        # Activation function
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        return x


# ----------------------------------
# Create the GCN model
# ----------------------------------
model = GCN()



# Pass graph through the GCN
# ----------------------------------
output = model(graph)



# ----------------------------------
# Print results
# ----------------------------------
print("Input feature shape :", graph.x.shape)
print("Edge shape :", graph.edge_index.shape)
print("Output feature shape :", output.shape)

torch.save(output.detach(), "outputs/features/gcn_output.pt")
print("Saved GCN output to outputs/gcn_output.pt")