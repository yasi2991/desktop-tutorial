import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Convert adjacency matrix to edge_index format
edge_index = torch.nonzero(torch.tensor(adjacency_matrix)).t().contiguous()

# Feature matrix and target tensor
x = torch.tensor(features.values, dtype=torch.float)
y_t = torch.tensor(output_matrix.values, dtype=torch.long)  # Target matrix with 0s, 1s, and nulls (assume nulls are -1)

# Mask to filter out null values
mask = (y_t != -1)  # True for valid 0s and 1s, False for nulls

# Create the PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index)
data.train_mask = mask

# Define the GAT model for binary classification
class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

# Initialize model, optimizer, and loss function
model = GATModel(in_channels=x.shape[1], hidden_channels=16, out_channels=2)  # 2 classes: 0 and 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[mask], y_t[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation
def test():
    model.eval()
    with torch.no_grad():
        out = model(data)
        test_loss = loss_fn(out[mask], y_t[mask])
        pred = out[mask].max(dim=1)[1]
        accuracy = (pred == y_t[mask]).sum().item() / mask.sum().item()
    return test_loss.item(), accuracy

# Run training
epochs = 200
for epoch in range(epochs):
    loss = train()
    if epoch % 10 == 0:
        test_loss, accuracy = test()
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')