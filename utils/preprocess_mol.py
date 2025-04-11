from rdkit.Chem import AllChem, MACCSkeys
from rdkit import Chem
import torch
import pandas as pd
import numpy as np



def ValidSmiles(smiles):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def smiles2FP(smiles, return_type='list'):
    mol = Chem.MolFromSmiles(smiles)
    # MACCS 생성
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    # Gen ECFP
    ecfp = Chem.RDKFingerprint(mol)
    if return_type == 'list':
        return ecfp.ToList() + maccs_fp.ToList()[1:]
    elif return_type == "split_list":
        return [ecfp.ToList(), maccs_fp.ToList()[1:]]
    elif return_type == 'dict':
        return {"ecfp": ecfp.ToList(), "maccs": maccs_fp.ToList()[1:]}


def preprocess_fingerprint(data):
    fp_dataset = []
    for mol in data:
        fp_dataset.append(smiles2FP(mol.smiles))

    fp_df = pd.DataFrame(fp_dataset)
    tensor_fp = torch.tensor(fp_df.values, dtype=torch.float32)

    return tensor_fp



def smiles2coord(smiles, add_H=True, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    
    if add_H:
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None

    AllChem.UFFOptimizeMolecule(mol)

    ret = mol.GetConformer().GetPositions()
    if type(ret) != np.ndarray:
        print(f"Failed to generate coordinates for {smiles}")
        return None
    return ret
