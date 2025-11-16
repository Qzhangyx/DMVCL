
import torch
import numpy as np

import numpy as np
import os
from torch_geometric.data import Data, Dataset, DataLoader
from scipy.sparse import csr_matrix
import pickle as pkl

# domain encoder
def get_interpro_data(interpro, device):
    rows = []
    cols = []
    data = []
    for i in list(range(interpro.shape[0])):
        tp = interpro[i,:]
        vals_idx = torch.argwhere(tp>0).reshape(-1)
        val = tp[vals_idx]

        rows += [i]*len(vals_idx)
        cols += vals_idx.tolist()
        data += val.tolist()

    col_nodes = 18847    #8535          
    interpro_matrix = csr_matrix((data, (rows, cols)), shape=(interpro.shape[0], col_nodes))

    inter_features = (torch.from_numpy(interpro_matrix.indices).to(device).long(), 
                        torch.from_numpy(interpro_matrix.indptr).to(device).long(), 
                        torch.from_numpy(interpro_matrix.data).to(device).float())
    return inter_features

