import torch
from torch_geometric.data import Data, Dataset

class interpro_data(Data):
    def __cat_dim__(self, key, item, store):
        if key in ['esm_representations', "interpro", 'SL_label', "ID"]:
            return None
        else:
            return super().__cat_dim__(key, item)   

# demo_dataset
# Thumuluri’s dataset
class protein_dataset(Dataset):
    def __init__(self, demo_path, transform=None, pre_transform=None):
  
        super(protein_dataset, self).__init__(demo_path, transform, pre_transform)
        self.chain_ids = torch.load(demo_path+"protein_ids_1.pt")
        self.demo_path = demo_path

    @property
    def raw_file_names(self):
        return self.chain_ids
    
    def len(self):
       
        return len(self.chain_ids)

    def get(self, idx):
        chain_ID = self.chain_ids[idx]
        data = torch.load(self.demo_path+"{}.pt".format(chain_ID))
        
        data = interpro_data(
            h_V_geo = data["h_V_geo"],
            prottrans_feat = data["prottrans_feat"],
            interpro = data["ipr"],
            esm_tokens = data["esm_token"],
            esm_representations = data["esm_seq"],
            edge_index = data["edge_index"],
            edge_feature = data["h_E"],
            one_hot_seq = data["onehot"],
            SL_label = data["label"],
            DSSP = data["dssp"],
            ID = chain_ID
        )
        return data


# Bai’s dataset
# 
class random_sl_new_data_deepmtc(Dataset):
    def __init__(self, data_class, transform=None, pre_transform=None):
        self.data_class = data_class
  
        super(random_sl_new_data_deepmtc, self).__init__(data_class, transform, pre_transform)
        self.chain_ids = torch.load("/home/nas2/biod/zq/ZQ/SL/data/Dataset/id_data_feature/pt_data_gpsfun/txt_file/{}_ids.pt".format(self.data_class))

    @property
    def raw_file_names(self):
        return self.chain_ids
    
    def len(self):
        
        return len(self.chain_ids)

    def get(self, idx):
        chain_ID = self.chain_ids[idx]
        data = torch.load("/home/nas2/biod/zq/ZQ/SL/data/pt_data_20250623/{}.pt".format(chain_ID))  
        #print("/home/nas2/biod/zq/ZQ/Protein_Function_Prediction/GPSFun/datasets/subcellular_location/{}_pt_data/{}.pt".format(self.data_class, chain_ID))
        data = interpro_data(
            h_V_geo = data["h_V_geo"],
            prottrans_feat = data["prottrans_feat"],
            interpro = data["interpro"],
            esm_tokens = data["esm_token"],
            esm_representations = data["esm_seq"],
            edge_index = data["edge_index"],
            edge_feature = data["h_E"],
            one_hot_seq = data["onehot"],
            SL_label = data["label"],
            DSSP = data["dssp"],
            edge_feature_45 = data['edge_feature_45'],
            edge_index_45 = data["edge_index_45"],
            MF_BP_CC_label = data["MF_BP_CC_label"].unsqueeze(0),
            ID = chain_ID
        )
        return data
    



