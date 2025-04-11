import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.preprocess_mol import smiles2FP, ValidSmiles, smiles2coord
from utils.smiles2graph import smiles_to_graph


class GraphDataset(Dataset):
    def __init__(self, data, data_name, add_H=True):
        '''
        data: pandas DataFrame (smiles, target)
        '''
        super().__init__()
        self.data_list = []
        self.smiles = []
        self.y = []
        self.data_name = data_name

        if isinstance(data, pd.DataFrame):
            iterator = data.iterrows()
        elif isinstance(data, MoleculeNet):
            iterator = data

        for mol in tqdm(iterator):
            # 1. diff. data sources
            if data_name == "BBB" or data_name == 'bbb':
                smiles = mol[1]["Unnamed: 0"]
                y = torch.Tensor([mol[1]['BBclass']])
            elif data_name == "logP" or data_name == 'logp':
                smiles = mol[1]["smiles"]
                y = torch.Tensor([mol[1]['logp']])
            elif data_name == "BACE" or data_name == 'bace':
                smiles = mol.smiles
                y = mol.y

            # 2. check valid smiles
            if not ValidSmiles(smiles):
                continue
            
            # 3. convert smiles to graph
            graph = smiles_to_graph(smiles, add_H=add_H)
            coord = smiles2coord(smiles, add_H=add_H)
            if type(coord) != np.ndarray or graph == None:
                continue
            
            self.data_list.append(Data(
                x=graph['x'],
                edge_attr=graph['edge_attr'],
                edge_index=graph['edge_index'],
                y=y,
                smiles=smiles,
                coord=coord
            ))
            self.y.append(y)
            self.smiles.append(smiles)
        
        self.y = torch.Tensor(self.y)

        self.x = torch.cat([data.x for data in self.data_list])
        self.edge_attr = torch.cat([data.edge_attr for data in self.data_list])
        self.edge_index = torch.cat([data.edge_index for data in self.data_list], dim=1)
        
        # self.coord = torch.cat([data.coord for data in self.data_list])


    def __len__(self):
        return len(self.y)

    def __repr__(self):
        return f"{self.data_name} Graph({len(self.y):,})"

    def __getitem__(self, idx):
        return self.data_list[idx]
