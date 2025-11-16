import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
import torch.nn as nn
from torch_scatter import scatter
from model.TransformerBlock import TransformerBlock


class SEQ(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, head):
        super(SEQ, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    
        self.conv1 = Linear(in_dim, hidden_dim)
        self.conv2 = Linear(hidden_dim, hidden_dim)
        
        self.transformer_block = TransformerBlock(hidden_dim, hidden_dim, head)

    def forward(self, x, edge_index, batch, inter_f):
 
        init_avg_h = scatter(x, dim=0, index=batch.unsqueeze(-1).expand(-1, x.size(-1)), dim_size=batch.max().item() + 1, reduce='mean')

        pre = x
        x = self.bn1(x)
        x = self.dropout(F.relu(self.conv1(x)))

        pre = x
        x = self.bn2(x)
        x = self.dropout(F.relu(self.conv2(x)))

        inter_h = inter_f[batch]
        hg = self.transformer_block(x, inter_h, edge_index, batch)
        readout = global_add_pool(hg, batch)
        return readout, init_avg_h