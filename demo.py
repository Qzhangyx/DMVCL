
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch, os
from model import get_inter_feature, main_model
from src import data_loader, model_utils, metric
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
current_pid = os.getpid()
print(f"THE PROCESS IS:{current_pid}")
batch_size = 1
epoches = 50
learning_rate = 0.0001
demo_dataset_path = "./src/demo_dataset/"
X_demo = data_loader.protein_dataset(demo_dataset_path)
demo_loader = DataLoader(X_demo, batch_size=batch_size, shuffle=False, drop_last=True)
model = main_model.CL_interpro_model(inter_size = 18847, inter_hid=1280, graph_size=20+184+9, graph_hid=1280, seq_size=1280, seq_hid=1280, label_num=10, head=4).to(device)
optim = torch.optim.Adam(params = model.parameters(),lr = learning_rate, weight_decay=0.0001)
ckp = "./src/ckp/dataset0_ckp.pt"
model, optim, current_epoch, min_val_loss = model_utils.load_ckp(ckp, model, optim, device = device)
model.eval()
with torch.no_grad():
    for data in demo_loader:    
        esm_tokens, esm_representations, edge_index, one_hot_seq, interpro = data.esm_tokens.to(torch.float32).to(device), data.esm_representations.to(device), data.edge_index.to(device), data.one_hot_seq.to(device), data.interpro.to(device)
        inter_features = get_inter_feature.get_interpro_data(interpro, device)
        label = data.SL_label.float()
        batch = data.batch.to(device)
        node_feat = torch.cat([data.one_hot_seq, data.h_V_geo, data.DSSP], dim=-1).to(device)
        ID = data["ID"]
        structure_embedding, sequence_embedding, y_pred = model(esm_tokens, edge_index, batch, inter_features, esm_representations, node_feat)
        y_pred = torch.sigmoid(y_pred).to(torch.float32).detach()[0]
        print("ID:%s, Cytoplasm:%.3f, Nucleus:%.3f, Extracellular:%.3f, Cell membrane:%.3f, Mitochondrion:%.3f, Plastid:%.3f, Endoplasmic reticulum:%.3f, Lysosome/Vacuole:%.3f, Golgi apparatus:%.3f, Peroxisome:%.3f" %(ID[0], y_pred[0],y_pred[1],y_pred[2], y_pred[3], y_pred[4], y_pred[5], y_pred[6], y_pred[7], y_pred[8], y_pred[9]))