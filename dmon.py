'''
Has classes of all the models investigated. 
They include, GCN, GAT, MinCut, GraphSAGE, DiffPool, DMon

'''


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,dense_mincut_pool, dense_diff_pool
from torch_geometric.utils import to_dense_adj
from torch.nn import Linear





class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
                 heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads))
            
        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
                 dropout=0.5, aggregator='mean'):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
    
class MinCut(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        
        in_channels = data.x.shape[1]
        hidden_channels = 32
        out_channels = len(set(data.y))
        n_clusters= 10
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.pool = Linear(hidden_channels, n_clusters)
        
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Cluster assignments
        s = self.pool(x)
        
        adj = to_dense_adj(edge_index)
        _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, s)
        
        # Final classification
        out = self.classifier(x)
        
        return F.log_softmax(out, dim=-1), mc_loss, o_loss
    

class DiffPool(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        
        in_channels = data.x.shape[1]
        hidden_channels = 32
        out_channels = len(set(data.y))
        n_clusters= 10
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.pool = Linear(hidden_channels, n_clusters)
        
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # clsuterings
        s = self.pool(x)
        
        # losses
        adj = to_dense_adj(edge_index)
        _, _, mc_loss, o_loss = dense_diff_pool(x, adj, s)
        
        # Final classification
        out = self.classifier(x)
        
        return F.log_softmax(out, dim=-1), mc_loss, o_loss


    
class DMoN(torch.nn.Module):
    def __init__(self, in_channels, num_clusters, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = DMoNPooling([hidden_channels, hidden_channels], num_clusters)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels) 
        self.pool2 = DMoNPooling([hidden_channels, hidden_channels], num_clusters)
        
    def forward(self, x, edge_index):
        x = F.selu(self.conv1(x, edge_index))
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)
        
        ca, x, adj, sl1, ol1, cl1 = self.pool1(x, adj, mask)
        
        x = F.selu(self.conv2(x, adj))
        ca, x, adj, sl2, ol2, cl2 = self.pool2(x, adj)
        
        # Return cluster assignments and combined losses
        return ca.squeeze(0), sl1 + sl2 + ol1 + ol2 + cl1 + cl2

    

