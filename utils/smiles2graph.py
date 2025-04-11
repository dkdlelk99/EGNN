from rdkit import Chem
from rdkit.Chem import BondType, BondStereo
import torch

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

custom_bond_rank = {
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 12,
}

allowable_bond_features = {
    "stereo": [
        BondStereo.STEREONONE,
        BondStereo.STEREOANY,
        BondStereo.STEREOZ,
        BondStereo.STEREOE,
    ]
}


# Safe indexing
def safe_index(mapping, value):
    try:
        return mapping.index(value)
    except ValueError:
        return len(mapping) - 1
# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2


# 예외(단일 원소 분자) 처리용 함수
def create_empty_graph():
    # 빈 edge_index (2 x 0)
    edge_index = torch.empty((2, 0), dtype=torch.int64)
    
    # 빈 edge_attr (0 x 3)
    edge_attr = torch.empty((0, 3), dtype=torch.int64)
    
    return edge_index, edge_attr


def atom_to_feature_vector(atom):
    atom_feature = [
            atom.GetAtomicNum(),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            int(atom.GetHybridization()),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature


def mol2edge_index(smiles, add_H=True):
    mol = Chem.MolFromSmiles(smiles)
    
    if add_H:
        mol = Chem.AddHs(mol)

    edge_index = []
    for bond in mol.GetBonds():
        # 결합을 연결하는 원자 인덱스
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
        edge_index.sort()
    
    return torch.tensor(edge_index).T


# Bond to feature vector
def bond_to_feature_vector(bond):
    return [
        custom_bond_rank.get(bond.GetBondType(), 0),  # 사용자 정의 결합 타입
        safe_index(allowable_bond_features["stereo"], bond.GetStereo()),  # 입체 정보
        int(bond.GetIsConjugated()),  # 공액 결합 여부
    ]

# Edge index 생성
def mol2edge_index(smiles, add_H=True):
    mol = Chem.MolFromSmiles(smiles)
    
    if add_H:
        mol = Chem.AddHs(mol)
        
    edge_index = []
    bonds = []  # 결합 객체 저장

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # 원자 인덱스 정렬
        edge_index.append([min(start, end), max(start, end)])
        edge_index.append([max(start, end), min(start, end)])
        bonds.append(bond)
        bonds.append(bond)

    # 정렬 기준: edge_index의 첫 번째 열, 그다음 두 번째 열
    edge_index_with_bonds = sorted(zip(edge_index, bonds), key=lambda x: (x[0][0], x[0][1]))
    edge_index = [pair[0] for pair in edge_index_with_bonds]
    bonds = [pair[1] for pair in edge_index_with_bonds]

    return torch.tensor(edge_index).T, bonds

# Edge attributes 생성
def mol2edge_attr(bonds):
    edge_attr = []
    for bond in bonds:
        bond_features = bond_to_feature_vector(bond)
        edge_attr.append(bond_features)
    return torch.tensor(edge_attr, dtype=torch.int64)



def smiles_to_graph(smiles, add_H=True):
    mol = Chem.MolFromSmiles(smiles)
    
    if add_H:
        mol = Chem.AddHs(mol)
    
    node_feature = []
    for atom in mol.GetAtoms():
        node_feature.append(atom_to_feature_vector(atom))
        
    if mol.GetNumBonds() == 0:  # 결합이 없는 경우
        edge_index, edge_attr = create_empty_graph()
        return {'x': torch.tensor(node_feature), 'edge_index': edge_index, 'edge_attr': edge_attr}
    
    # 기존 edge_index 및 edge_attr 생성 로직
    edge_index, bonds = mol2edge_index(smiles, add_H)
    edge_attr = mol2edge_attr(bonds)
    return {'x': torch.tensor(node_feature), 'edge_index': edge_index, 'edge_attr': edge_attr}
