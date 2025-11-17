import math
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        if self.alpha is not None:
            alpha_t = self.alpha[target.long()]
            focal_loss = alpha_t * focal_loss

        return focal_loss
    
def compute_class_weights(label): 
    class_count = torch.sum(label.to(torch.float32), dim=0)
    class_weights = class_count.sum() / class_count
    class_weights = torch.where(torch.isinf(class_weights), torch.zeros_like(class_weights), class_weights)
    return class_weights
# inter_CL
class MatchLoss(nn.Module): 
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, feature_left, feature_right, match_type="graph"):
        device = feature_left.device
        assert match_type in {"node", "graph"}, print("match_type error")
        if match_type == "node":
            similarity = F.cosine_similarity(feature_left, feature_right, dim=1)
            similarity = torch.exp(similarity / self.T)
            loss = torch.mean(-torch.log(similarity))
        else:
            n = len(feature_left)  
            similarity = F.cosine_similarity(feature_left.unsqueeze(1), feature_right.unsqueeze(0), dim=2)
            similarity = torch.exp(similarity / self.T)
            mask_pos = torch.eye(n, n, dtype=bool).to(device)
            sim_pos = torch.masked_select(similarity, mask_pos)
            sim_total_row = torch.sum(similarity, dim=0)
            #sim_total_row = torch.sum(similarity, dim=0)-sim_pos
            loss_row = torch.div(sim_pos, sim_total_row)
            loss_row = -torch.log(loss_row)

            sim_total_col = torch.sum(similarity, dim=1)
            loss_col = torch.div(sim_pos, sim_total_col)
            loss_col = -torch.log(loss_col)

            loss = loss_row + loss_col
            loss = torch.sum(loss) / (2 * n)

        return loss
# intra_CL
class ContrastiveLoss(nn.Module):
    def __init__(self, tau=0.1, alpha=0.5):
        super().__init__()
        self.tau = tau  
        self.alpha = alpha  
        self.eps = 1e-8  

    def forward(self, features, labels, dataset_class):
      
        if dataset_class == "gpsfun":
            global_counts = gpsfun_class_number
        elif dataset_class == "deepmtc":
            global_counts = deepmtc_class_number
        
        global_weights = 1 / torch.sqrt(global_counts.float() + 1)  
        class_counts = labels.sum(0) 
        current_weights = 1 / torch.sqrt(class_counts.float() + 1)  
        fused_weights = self.alpha * global_weights + (1 - self.alpha) * current_weights 
        features = F.normalize(features, p=2, dim=1)  
        
        sim = torch.mm(features, features.T) / self.tau  
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach() 

        pos_mask = (torch.mm(labels, labels.T) > 0)  
        neg_mask = (torch.mm(labels, labels.T) == 0)  
        pos_mask = pos_mask & ~torch.eye(labels.shape[0], dtype=bool, device=labels.device)  

        sample_weights = labels @ fused_weights.unsqueeze(1)  # (L,1)
        sample_weights = sample_weights / (sample_weights.mean() + self.eps)  

        pos_sim_pairs = sim[pos_mask]  
        pos_indices = torch.where(pos_mask)

        pos_weights = torch.sqrt(sample_weights[pos_indices[0]] * sample_weights[pos_indices[1]])
        pos_term = (torch.exp(pos_sim_pairs) * pos_weights).sum()

        neg_sim_pairs = sim[neg_mask]  
        neg_indices = torch.where(neg_mask)
    
        neg_weights = (sample_weights[neg_indices[0]] + sample_weights[neg_indices[1]]) / 2
        neg_term = (torch.exp(neg_sim_pairs) * neg_weights).sum()

        loss = -torch.log(pos_term / (pos_term + neg_term + self.eps))
        
        return loss
    
