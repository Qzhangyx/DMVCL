from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch, os
from model import get_inter_feature, main_model
from src import data_loader, data_utils, model_utils, metric_sl, loss_CL
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")  # Set the device to GPU if available, otherwise CPU
DATA_PATH = './dataset'  # Path to the dataset
MODEL_PATH = "./src/ckp/"  # Path to save model checkpoints

# Hyperparameters
batch_size = 32
epoches = 50
learning_rate = 0.0001

# Load the training and validation datasets
X_train = data_loader.protein_dataset(DATA_PATH, mode='train')
X_val = data_loader.protein_dataset(DATA_PATH, mode='val')
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(X_val, batch_size=batch_size, shuffle=False, drop_last=False)

# Initialize the model

#inter_size=8535   #inter_size Changing the dataset requires modification
model = main_model.CL_interpro_model(
    inter_size=18847, inter_hid=1280, graph_size=20+184+9, graph_hid=1280, 
    seq_size=1280, seq_hid=1280, label_num=10, head=4
).to(device)

# Optimizer
optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.0001)

# Loss functions
Inter_Loss = loss_CL.MatchLoss()  # Loss for inter-modal consistency
Intro_Loss = loss_CL.ContrastiveLoss()  # Contrastive loss for intra-modal consistency
loss_function = torch.nn.BCELoss()  # Binary cross-entropy loss for classification

def train(train_loader, model): 
    """
    Training function to update model parameters based on the training data.
    """
    model.train()  # Set model to training mode
    total_loss = 0
    for data in tqdm(train_loader):
        # Move data to device (GPU or CPU)
        esm_tokens, esm_representations, edge_index, one_hot_seq, interpro = data.esm_tokens.to(torch.float32).to(device), data.esm_representations.to(device), data.edge_index.to(device), data.one_hot_seq.to(device), data.interpro.to(device)

        inter_features = data_loader.get_inter_feature(interpro, device)  # Get interaction features
        label = data.SL_label.float().to(device)  # Get labels
        batch = data.batch.to(device)  # Get batch indices
        node_feat = torch.cat([data.one_hot_seq, data.h_V_geo, data.DSSP], dim=-1).to(device)  # Concatenate node features

        # Forward pass
        structure_embedding, sequence_embedding, y_pred = model(esm_tokens, edge_index, batch, inter_features, esm_representations, node_feat)
        y_pred = torch.sigmoid(y_pred)  # Apply sigmoid activation for binary classification

        # Compute losses
        intra_loss = Intro_Loss(structure_embedding, label, "gpsfun") + Intro_Loss(sequence_embedding, label, "gpsfun")  # Intra-modal contrastive loss
        inter_loss = Inter_Loss(structure_embedding, sequence_embedding)  # Inter-modal consistency loss

        # Compute class weights for handling class imbalance
        class_count = torch.sum(label.to(torch.float32), dim=0)
        class_weights = class_count.sum() / class_count
        class_weights = torch.where(torch.isinf(class_weights), torch.zeros_like(class_weights), class_weights)  # Avoid infinite weights

        model_loss = loss_function(y_pred, label)  # Binary cross-entropy loss
        loss_ = (model_loss * class_weights).mean()  # Weighted loss

        # Total loss
        loss = 0.4 * intra_loss + 0.4 * inter_loss + loss_
        # loss = 0.5 * intra_loss + 0.5 * inter_loss + loss_

        # Backward pass and optimization
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(val_loader, model):
    """
    Validation function to evaluate model performance on the validation set.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            # Move data to device (GPU or CPU)
            esm_tokens, esm_representations, edge_index, one_hot_seq, interpro = data.esm_tokens.to(torch.float32).to(device), data.esm_representations.to(device), data.edge_index.to(device), data.one_hot_seq.to(device), data.interpro.to(device)

            inter_features = data_loader.get_inter_feature(interpro, device)  # Get interaction features
            label = data.SL_label.float().to(device)  # Get labels
            batch = data.batch.to(device)  # Get batch indices
            node_feat = torch.cat([data.one_hot_seq, data.h_V_geo, data.DSSP], dim=-1).to(device)  # Concatenate node features

            # Forward pass
            structure_embedding, sequence_embedding, y_pred = model(esm_tokens, edge_index, batch, inter_features, esm_representations, node_feat)
            y_pred = torch.sigmoid(y_pred)  # Apply sigmoid activation for binary classification

            model_loss = loss_function(y_pred, label)  # Binary cross-entropy loss
            total_loss += model_loss.item()

    return total_loss / len(val_loader)

model_path = "./ckp"
# Training loop
best_val_loss = float('inf')
for epoch in range(epoches):
    Train_loss = train(train_loader, model)  # Train the model for one epoch
    Val_loss = validate(val_loader, model)  # Validate the model

    print(f"Epoch [{epoch+1}/{epoches}], Train Loss: {Train_loss:.4f}, Val Loss: {Val_loss:.4f}")

    # Save the model if it has the best validation loss so far
    if Val_loss < best_val_loss:
        best_val_loss = Val_loss
        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': best_val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
        }
        model_utils.save_ckp(checkpoint, False, model_path + "current_checkpoint_{}.pt".format(epoch), model_path + "best_model.pt")