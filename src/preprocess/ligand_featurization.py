import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

def encode_ligand_to_Data(mol: Chem.rdchem.Mol):    
    
    lpp = LigandPreprocessor(mol)
    
    # 1. generate ligand_feature (54dim)
    lig_feature = lpp.get_lig_feature(mol, to_tensor=True)
    
    # 2. generate adj_matrix
    lig_edge_indices, lig_edge_attr = lpp.get_edge_info(mol)

    # pyg style adj가 아니라 numpy style adj (nxn) 인 경우 tensor로 변환
    if isinstance(lig_edge_indices, np.ndarray):
        lig_edge_indices = torch.from_numpy(lig_edge_indices)
    
    # 3. generate atom_pos
    lig_pos = lpp.get_atom_position(mol)
    
    # 4. generate atomic_number z
    lig_z = lpp.get_atomic_number(mol)
    
    # 최종 Data객체 생성    
    ligand_data = Data(x=lig_feature, edge_index=lig_edge_indices, edge_attr=lig_edge_attr, pos=lig_pos, z=lig_z)
    
    return ligand_data
    
    
class LigandPreprocessor:
    def __init__(self, mol, num_conformers=5):
        self.mol = mol
        self.num_conformers = num_conformers
        
        self.SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "X"]
        self.DEGREES = [0, 1, 2, 3, 4, 5]
        self.HYBRIDIZATIONS = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ]
        self.FORMALCHARGES = [-2, -1, 0, 1, 2, 3, 4]
        
        pt = """
            H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE
            LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE
            NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR
            K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR
            RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE
            CS,BA,LU,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
        """
        
        self.PERIODIC_TABLE = dict()
        for i, per in enumerate(pt.split()):
            for j, ele in enumerate(per.split(",")):
                self.PERIODIC_TABLE[ele] = (i, j)
        self.PERIODS = [0, 1, 2, 3, 4, 5]
        self.GROUPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        
    def one_of_k_encoding(self, x, allowable_set: list) -> list[bool]:
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set: list) -> list[bool]:
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
    
    def get_period_group(self, atom) -> list[bool]:
        period, group = self.PERIODIC_TABLE[atom.GetSymbol().upper()]
        return self.one_of_k_encoding(period, self.PERIODS) + self.one_of_k_encoding(group, self.GROUPS)
    
    def atom_feature(self, mol, atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        return np.array(
            self.one_of_k_encoding_unk(atom.GetSymbol(), self.SYMBOLS)
            + self.one_of_k_encoding_unk(atom.GetDegree(), self.DEGREES)
            + self.one_of_k_encoding_unk(atom.GetHybridization(), self.HYBRIDIZATIONS)
            + self.one_of_k_encoding_unk(atom.GetFormalCharge(), self.FORMALCHARGES)
            + self.get_period_group(atom)
            + [atom.GetIsAromatic()]
        )  # (9, 6, 7, 7, 24, 1) --> total 54        
    
    def get_lig_feature(self, mol, to_tensor=False):
        '''
        mol 객체의 원자를 순회하면서 atom_feature 함수를 활용해 분자 feature 생성
        to_tensor -> return을 numpy 대신 tensor
        '''
        n_atoms = mol.GetNumAtoms()
        atom_features = []
        
        for atom_idx in range(n_atoms):
            atom_features.append(self.atom_feature(mol, atom_idx))
            
        lig_feature = np.array(atom_features)
        
        if to_tensor:
            lig_feature = torch.from_numpy(lig_feature).to(dtype=torch.float32)
            
        return lig_feature
        
    def get_edge_info(self, mol):
        adj = Chem.GetAdjacencyMatrix(mol, useBO=True) # [n_atoms, n_atoms]
        
        edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long) # [2, n_edges]
        edge_attr = torch.tensor(adj[adj.nonzero()], dtype=torch.float) # [n_edges, ]
            
        return edge_index, edge_attr

    def get_atom_position(self, mol, to_tensor=True, numConfs=None):
        # 3D conformer 로 변환

        # mol.RemoveAllConformers() # standardize conformation
        # AllChem.EmbedMolecules(mol)
        # AllChem.MMFFOptimizeMolecule(mol)
        # 원하는 conformer 갯수만큼 슬라이싱
        if numConfs is None:
            numConfs = self.num_conformers
            
        pos_list = []
        for n in range(numConfs):
            conf = mol.GetConformer(n)
            pos = conf.GetPositions()
            pos_list.append(pos)
        
        stacked_pos = np.stack(pos_list, axis=1)
            
        if to_tensor:
            stacked_pos = torch.from_numpy(stacked_pos).to(dtype=torch.float32)
        
        return stacked_pos

    def get_atomic_number(self, mol, to_tensor=True):
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
        if to_tensor:
            atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
            
        return atomic_numbers