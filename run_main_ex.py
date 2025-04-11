from custom_dataloader import GraphDataset
from models.EquivariantGNN import EGTF
from utils.splitter import scaffold_split, random_scaffold_split
from utils.etc import time_check

import time
import torch
import torch.nn as nn
from torchmetrics import AUROC
# from utils.etc import AUCMeter as AUROC
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as gDataLoader



def eval(model, val_dl, val_dataset, loss_func, val_eval):
    model.eval()
    val_loss = 0.0

    for data in val_dl:
        x = data.x.to(torch.float32).to(device)
        coord = torch.cat([torch.Tensor(c) for c in data.coord]).to(device)
        batch = data.batch.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(torch.float32).to(device)
        y = data.y.to(device)

        pred = model(x, coord, batch, edge_index, edge_attr) # node_feats, coords_feats, batch, edge_index=None, edge_feats
        loss = loss_func(pred, y)
        val_eval.update(preds=pred, target=y)

        val_loss += loss.item() * len(y)
    
    val_metric = val_eval.compute()
    val_loss = val_loss/len(val_dataset)
    torch.cuda.empty_cache()

    return val_metric, val_loss




dataset_name = 'bace'

bace = MoleculeNet(root='./data', name=dataset_name)
print("Load dataset...")
dataset = GraphDataset(bace, dataset_name, add_H=False)
print("Done loading dataset!")

train_idx, val_idx, test_idx = random_scaffold_split(dataset=dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

batch_size = 2**8

train_dl = gDataLoader(train_dataset, batch_size, shuffle=True)
val_dl = gDataLoader(val_dataset, batch_size, shuffle=True)
test_dl = gDataLoader(test_dataset, batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EGTF(64, 256, 4).to(device)


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params:,}")


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = nn.BCELoss()


epoch = 200

start_time = time.time()

for _e in range(epoch):
    model.train()

    train_eval = AUROC(task="binary")
    val_eval = AUROC(task="binary")

    train_loss = 0.0

    for data in train_dl:
        x = data.x.to(torch.float32).to(device)
        coord = torch.cat([torch.Tensor(c) for c in data.coord]).to(device)
        batch = data.batch.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(torch.float32).to(device)
        y = data.y.to(device)

        optimizer.zero_grad()
        pred = model(x, coord, batch, edge_index, edge_attr) # node_feats, coords_feats, batch, edge_index=None, edge_feats
        loss = loss_func(pred, y)
        train_eval.update(preds=pred, target=y)

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y)

    train_metric = train_eval.compute()
    train_loss = train_loss/len(train_dataset)

    print(f"Epoch {_e+1}\n-Train\nAUC socre: {train_metric:.5f} Train loss: {train_loss:.5f}")

    val_score, val_loss = eval(model, val_dl, val_dataset, loss_func, val_eval)
    print(f"-Valid\nAUC score: {val_score:.5f} Valid loss: {val_loss:.5f}\n")


print(f"Training: {time_check(time.time() - start_time)}")
