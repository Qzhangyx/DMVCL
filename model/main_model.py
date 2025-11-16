
from torch import nn
import torch
import torch.nn as nn
from model import domain_structure_att, domain_sequence_att
from model import inter_model

class CL_interpro_model(nn.Module):
    def __init__(self, inter_size, inter_hid, graph_size, graph_hid, seq_size, seq_hid, label_num, head):
        super(CL_interpro_model, self).__init__()
        self.inter_embedding = inter_model.inter_model(inter_size, inter_hid)

        self.GNN = domain_structure_att.GCN(graph_size, graph_hid, label_num, head)
        self.SEQ = domain_sequence_att.SEQ(seq_size, seq_hid, label_num, head)



        self.classify = nn.Sequential(
            nn.BatchNorm1d(graph_size+graph_hid),
            nn.Linear(graph_size+graph_hid, (graph_size+graph_hid)*2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear((graph_size+graph_hid)*2, (graph_size+graph_hid)*2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear((graph_size+graph_hid)*2, label_num)
        )

        self.classify1 = nn.Sequential(
            nn.BatchNorm1d(graph_hid),
            nn.Linear(graph_hid, (graph_hid)*2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear((graph_hid)*2, (graph_hid)*2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear((graph_hid)*2, label_num)
        )


        self.classify3 = nn.Sequential(
            nn.BatchNorm1d(seq_hid+graph_hid+seq_size),
            nn.Linear(seq_hid+graph_hid+seq_hid, (seq_hid)*2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear((seq_hid)*2, seq_hid),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(seq_hid, label_num)
        )


    def forward(self, x, edge_index, batch, inter_feature, esm_representations, x1):

        inter_feature = self.inter_embedding(inter_feature)
        graph_feature, graph_init_feature = self.GNN(x1, edge_index, batch, inter_feature)
        seq_feature, seq_init_feature = self.SEQ(x, edge_index, batch, inter_feature)
        y = self.classify3(torch.cat((esm_representations, graph_feature, seq_feature), 1))
        return graph_feature, seq_feature, y
    



