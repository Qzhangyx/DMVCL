

from torch import nn
import torch
import torch.nn.functional as F
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, head=1):
        super(TransformerBlock, self).__init__()
        self.head = head


        self.trans_q_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_k_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_v_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

   
        self.concat_trans = nn.Linear(hidden_dim * head, hidden_dim, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.layernorm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
    def forward(self, x, inter_h, edge_index, batch):

        multi_output = []
        for i in range(self.head):
            q = self.trans_q_list[i](x)  # Query
            k = self.trans_k_list[i](inter_h)  # Key
            v = self.trans_v_list[i](x)  # Value
            att = torch.sum(torch.mul(q, k) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float)), dim=-1, keepdim=True)
            alpha = F.softmax(att, dim=0)
            tp = v * alpha  
            multi_output.append(tp)
 
        multi_output = torch.cat(multi_output, dim=1)
        multi_output = self.concat_trans(multi_output)
        multi_output = self.layernorm(multi_output + x)
        multi_output = self.layernorm(self.ff(multi_output) + multi_output)
        return multi_output