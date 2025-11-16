from torch import nn
import torch.nn.functional as F
import torch.nn as nn

# domain encoder
class inter_model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(inter_model, self).__init__()
        
        self.embedding_layer = nn.EmbeddingBag(input_size, hidden_size, mode='sum', include_last_offset=True)

        self.linearLayer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU()
        )
    
    def forward(self, inter_feature):
        inter_feature = F.relu(self.embedding_layer(*inter_feature))
        inter_feature = self.linearLayer(inter_feature)

        return inter_feature