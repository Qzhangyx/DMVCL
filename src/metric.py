import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_curve,roc_curve, auc, f1_score
import torch

def sample_accuracy(y_true, y_pred, threshold=0.5):
 
    y_pred = (y_pred > threshold).float()
   
    correct_samples = torch.all(y_pred == y_true, dim=1).float().mean().item()
    return correct_samples


def calculate_metrics(y_true, y_pred):

    acc = sample_accuracy(y_true, y_pred)
    y_bin = (y_pred > 0.5).float().numpy()
    

    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()


    precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_flat)
    micro_aupr = auc(recall, precision)

    fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
    micro_auc = auc(fpr, tpr)

    y_pred_flat_binary = (y_pred_flat > 0.5).astype(int)
    micro_f1 = f1_score(y_true_flat, y_pred_flat_binary)

    aupr_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0 and np.sum(1 - y_true[:, i]) > 0:  
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            aupr_list.append(auc(recall, precision))
    macro_aupr = np.mean(aupr_list)

    auc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0 and np.sum(1 - y_true[:, i]) > 0: 
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc_list.append(auc(fpr, tpr))
    macro_auc = np.mean(auc_list)

    f1_list = []
    for i in range(y_true.shape[1]):
        y_pred_binary = (y_pred[:, i] > 0.5).astype(int)
        if np.sum(y_true[:, i]) > 0: 
            f1_list.append(f1_score(y_true[:, i], y_pred_binary))
    macro_f1 = np.mean(f1_list)

    jaccard = jaccard_score(y_true, y_bin, average='samples')
    

    return micro_auc, micro_aupr, micro_f1, macro_auc, macro_aupr, macro_f1, acc, jaccard




