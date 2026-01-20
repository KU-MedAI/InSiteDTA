import numpy as np
import os
import torch
import argparse
import pickle
from tqdm import tqdm
from pathlib import Path
from openbabel import openbabel

import warnings
from Bio import BiopythonWarning

warnings.simplefilter("ignore", BiopythonWarning)
openbabel.obErrorLog.SetOutputLevel(0)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--voxel_size", type=float, default=2)
    parser.add_argument("--n_voxels", type=int, default=32)
    parser.add_argument("--label_radius", type=float, default=2.0, help="Radius for pocket labeling (0 for exact voxel)")
    parser.add_argument("--device", default=0)
    args = parser.parse_args()
    return args


class ProteinVoxelizer:
    """
    A class to convert protein structures into 3D voxel grids with rich chemical features.
    Updated for Dual-Binding Site Logic & I/O Safety.
    """

    def __init__(self, voxel_size=2.0, n_voxels=32):
        self.voxel_size = voxel_size
        self.n_voxels = n_voxels

        # Define 21 feature channels
        self.feature_channels = {
            "C": 0, "N": 1, "O": 2, "S": 3, "P": 4, "F": 5, "CL": 6, "BR": 7, "I": 8, "H": 9, "METAL": 10,
            "HYB_SP": 11, "HYB_SP2": 12, "HYB_SP3": 13,
            "HYDROPHOBIC": 14, "AROMATIC": 15, "ACCEPTOR": 16, "DONOR": 17, "RING": 18, "POSITIVE": 19, "NEGATIVE": 20,
        }
        self.n_features = len(self.feature_channels)
        self.metal_ions = {"ZN", "MG", "CA", "FE", "NA", "K", "MN", "CO", "CU", "NI"}

        # Residue properties
        self.hydrophobic_residues = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "TYR", "CYS"}
        self.aromatic_residues = {"PHE", "TRP", "TYR", "HIS"}
        self.acceptor_residues = {"ASP", "GLU", "ASN", "GLN", "HIS", "SER", "THR", "TYR", "CYS"}
        self.donor_residues = {"ARG", "LYS", "HIS", "TRP", "ASN", "GLN", "SER", "THR", "TYR", "CYS"}
        self.ring_residues = {"PHE", "TRP", "TYR", "HIS", "PRO"}
        self.positive_residues = {"ARG", "LYS", "HIS"}
        self.negative_residues = {"ASP", "GLU"}
        
        self.unidentified_atoms = []

    def read_protein_features(self, pdb_file: str) -> list[dict]:
        """Read atomic coordinates and features using OpenBabel."""
        # [Safety] Handle None or empty path
        if not pdb_file:
            return []

        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("pdb")
        mol = openbabel.OBMol()

        # [Fix] Convert Path object to string for OpenBabel C++ bindings
        if not obConversion.ReadFile(mol, str(pdb_file)):
            # print(f"Error: Could not read PDB file: {pdb_file}")
            return []

        atoms_data = []
        for atom in openbabel.OBMolAtomIter(mol):
            residue = atom.GetResidue()
            if not residue: continue

            element = openbabel.GetSymbol(atom.GetAtomicNum()).upper()
            atom_type = "METAL" if element in self.metal_ions else element

            if atom_type in self.feature_channels:
                atom_data = {
                    "element": atom_type,
                    "coords": np.array([atom.GetX(), atom.GetY(), atom.GetZ()]),
                    "hyb": atom.GetHyb(),
                    "resname": residue.GetName(),
                    "resnum": residue.GetNum(),
                    "atom_name": residue.GetAtomID(atom).strip(),
                }
                atoms_data.append(atom_data)
            else:
                if atom_type not in self.unidentified_atoms:
                    # print(f'Warning: atom type "{atom_type}" not found.')
                    self.unidentified_atoms.append(atom_type)
        return atoms_data

    def _precompute_atom_features(self, atoms: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Convert atom list to coord and feature matrices."""
        n_atoms = len(atoms)
        coords = np.zeros((n_atoms, 3), dtype=np.float32)
        features = np.zeros((n_atoms, self.n_features), dtype=np.float32)

        hyb_map = {1: "HYB_SP", 2: "HYB_SP2", 3: "HYB_SP3"}
        prop_map_items = [
            ("HYDROPHOBIC", self.hydrophobic_residues), ("AROMATIC", self.aromatic_residues),
            ("ACCEPTOR", self.acceptor_residues), ("DONOR", self.donor_residues),
            ("RING", self.ring_residues), ("POSITIVE", self.positive_residues),
            ("NEGATIVE", self.negative_residues),
        ]

        for i, atom in enumerate(atoms):
            coords[i] = atom["coords"]
            if atom["element"] in self.feature_channels:
                features[i, self.feature_channels[atom["element"]]] = 1.0
            if atom["hyb"] in hyb_map:
                features[i, self.feature_channels[hyb_map[atom["hyb"]]]] = 1.0
            for prop, res_set in prop_map_items:
                if atom["resname"] in res_set:
                    features[i, self.feature_channels[prop]] = 1.0
        return coords, features

    def get_voxel_centers(self, start_point: np.ndarray) -> np.ndarray:
        x = (np.arange(self.n_voxels) * self.voxel_size + start_point[0] + self.voxel_size / 2)[:, None, None]
        y = (np.arange(self.n_voxels) * self.voxel_size + start_point[1] + self.voxel_size / 2)[None, :, None]
        z = (np.arange(self.n_voxels) * self.voxel_size + start_point[2] + self.voxel_size / 2)[None, None, :]
        
        voxel_centers = np.zeros((self.n_voxels, self.n_voxels, self.n_voxels, 3))
        voxel_centers[..., 0] = x
        voxel_centers[..., 1] = y
        voxel_centers[..., 2] = z
        return voxel_centers

    def get_start_point(self, atoms: list[dict]) -> np.ndarray:
        if not atoms: return np.zeros(3)
        coords = np.array([atom["coords"] for atom in atoms])
        center = np.mean(coords, axis=0)
        return center - (self.voxel_size * self.n_voxels / 2)

    # === [Main Voxelization Function] ===
    def voxelize_gpu_v2(
        self,
        protein_path: str,
        pocket_path: str,
        r_cutoff: float = 4.0,
        device: str = "cuda",
        batch_size: int = 1024,
        defined_center: np.ndarray = None,  # <--- [New] 강제 중심 설정용 인자
        label_radius: float = 2.0  # <--- [New] Pocket labeling radius (0 for exact voxel)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # 1. Load Data
        protein_atoms = self.read_protein_features(protein_path)
        pocket_atoms = self.read_protein_features(pocket_path)

        if not protein_atoms:
            return (
                np.zeros((self.n_voxels, self.n_voxels, self.n_voxels, self.n_features)),
                np.zeros((self.n_voxels, self.n_voxels, self.n_voxels, 1)),
                np.zeros(3)
            )

        protein_coords_raw = np.array([atom["coords"] for atom in protein_atoms])
        pocket_coords_raw = np.array([atom["coords"] for atom in pocket_atoms]) if pocket_atoms else np.array([])
        grid_size = self.voxel_size * self.n_voxels

        # 2. Center Calculation Logic
        if defined_center is not None:
            # [Logic] 외부에서 주어진 중심(Unified Center) 사용
            center = defined_center
            start_point = center - (grid_size / 2)
        else:
            # [Logic] 기존 Intelligent Center (단백질 중심 -> 포켓 쪽 이동)
            center = np.mean(protein_coords_raw, axis=0)
            max_attempts = 100
            for attempt in range(max_attempts):
                start_point = center - (grid_size / 2)
                end_point = start_point + grid_size
                if pocket_coords_raw.size == 0 or (
                    np.all(pocket_coords_raw >= start_point) and np.all(pocket_coords_raw <= end_point)
                ):
                    break
                pocket_center = np.mean(pocket_coords_raw, axis=0)
                shift_fraction = (attempt + 1) / max_attempts
                center = (1 - shift_fraction) * np.mean(protein_coords_raw, axis=0) + shift_fraction * pocket_center

        # 3. GPU Voxelization
        voxel_center_coords = self.get_voxel_centers(start_point)
        flat_voxel_centers = torch.tensor(voxel_center_coords.reshape(-1, 3), device=device, dtype=torch.float32)

        # Protein Processing
        p_coords, p_features = self._precompute_atom_features(protein_atoms)
        p_coords_tensor = torch.tensor(p_coords, device=device, dtype=torch.float32)
        p_features_tensor = torch.tensor(p_features, device=device, dtype=torch.float32)
        flat_voxels = torch.zeros((self.n_voxels**3, self.n_features), device=device)

        num_atoms = p_coords_tensor.shape[0]
        for i in range(0, num_atoms, batch_size):
            batch_c = p_coords_tensor[i : i + batch_size]
            batch_f = p_features_tensor[i : i + batch_size]
            dists = torch.cdist(batch_c, flat_voxel_centers)
            mask = dists <= r_cutoff
            densities = torch.exp(-dists**2 / 2) * mask.float()
            flat_voxels += (densities.T @ batch_f)

        voxel = flat_voxels.view(self.n_voxels, self.n_voxels, self.n_voxels, self.n_features)

        # Label Processing
        label = torch.zeros((self.n_voxels, self.n_voxels, self.n_voxels, 1), device=device)
        if len(pocket_coords_raw) > 0:
            if label_radius > 0:
                # [Logic] Radius-based labeling (Original)
                flat_label = torch.zeros((self.n_voxels**3, 1), device=device)
                pkt_coords_tensor = torch.tensor(pocket_coords_raw, device=device, dtype=torch.float32)
                for i in range(0, pkt_coords_tensor.shape[0], batch_size):
                    batch_c = pkt_coords_tensor[i : i + batch_size]
                    dists = torch.cdist(batch_c, flat_voxel_centers)
                    within = (dists <= label_radius).any(dim=0).float().unsqueeze(1)
                    flat_label = torch.max(flat_label, within)
                label = flat_label.view(self.n_voxels, self.n_voxels, self.n_voxels, 1)

            else:
                # [Logic] Exact voxel matching instead of radius-based (label_radius == 0)
                indices = np.floor((pocket_coords_raw - start_point) / self.voxel_size).astype(int)
                
                # Filter out-of-bounds
                mask = (
                    (indices[:, 0] >= 0) & (indices[:, 0] < self.n_voxels) &
                    (indices[:, 1] >= 0) & (indices[:, 1] < self.n_voxels) &
                    (indices[:, 2] >= 0) & (indices[:, 2] < self.n_voxels)
                )
                valid_indices = indices[mask]
                
                if len(valid_indices) > 0:
                    idx_t = torch.tensor(valid_indices, device=device, dtype=torch.long)
                    label[idx_t[:, 0], idx_t[:, 1], idx_t[:, 2], 0] = 1.0

        return voxel.cpu().numpy(), label.cpu().numpy(), center
        
    def voxelize_inference(
        self,
        protein_path: str,
        r_cutoff: float = 4.0,
        device: str = "cuda",
        batch_size: int = 1024,
        defined_center: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Voxelization for inference (process protein only, without pocket)
        
        Returns:
            voxel: (n_voxels, n_voxels, n_voxels, n_features) numpy array
            center: (3,) numpy array - grid center point used
        """
        # 1. Load Protein Data
        protein_atoms = self.read_protein_features(protein_path)

        if not protein_atoms:
            return (
                np.zeros((self.n_voxels, self.n_voxels, self.n_voxels, self.n_features)),
                np.zeros(3)
            )

        protein_coords_raw = np.array([atom["coords"] for atom in protein_atoms])
        grid_size = self.voxel_size * self.n_voxels

        # 2. Center Calculation
        if defined_center is not None:
            center = defined_center
        else:
            center = np.mean(protein_coords_raw, axis=0)
        
        start_point = center - (grid_size / 2)

        # 3. GPU Voxelization
        voxel_center_coords = self.get_voxel_centers(start_point)
        flat_voxel_centers = torch.tensor(
            voxel_center_coords.reshape(-1, 3), 
            device=device, 
            dtype=torch.float32
        )

        # Protein Processing
        p_coords, p_features = self._precompute_atom_features(protein_atoms)
        p_coords_tensor = torch.tensor(p_coords, device=device, dtype=torch.float32)
        p_features_tensor = torch.tensor(p_features, device=device, dtype=torch.float32)
        flat_voxels = torch.zeros((self.n_voxels**3, self.n_features), device=device)

        num_atoms = p_coords_tensor.shape[0]
        for i in range(0, num_atoms, batch_size):
            batch_c = p_coords_tensor[i : i + batch_size]
            batch_f = p_features_tensor[i : i + batch_size]
            dists = torch.cdist(batch_c, flat_voxel_centers)
            mask = dists <= r_cutoff
            densities = torch.exp(-dists**2 / 2) * mask.float()
            flat_voxels += (densities.T @ batch_f)

        voxel = flat_voxels.view(self.n_voxels, self.n_voxels, self.n_voxels, self.n_features)

        return voxel.cpu().numpy(), center

    # === [Post-Processing Methods] ===
    def retrieve_pdb_path(self, data_vault: str, pdb_code: str) -> tuple[str, str]:
        pdb_dir = [f for f in os.listdir(data_vault) if pdb_code in f]
        if len(pdb_dir) != 1: raise FileNotFoundError(f"Entry error for {pdb_code}")
        pdb_dir_abs = os.path.join(data_vault, pdb_dir[0])
        protein_path = [f for f in os.listdir(pdb_dir_abs) if f.endswith("_protein.pdb")][0]
        pocket_path = [f for f in os.listdir(pdb_dir_abs) if f.endswith("_pocket.pdb")][0]
        return os.path.join(pdb_dir_abs, protein_path), os.path.join(pdb_dir_abs, pocket_path)

    def get_predicted_pocket_atoms_from_pred(self, original_pdb, pred, center=None, threshold=0.5):
        if isinstance(pred, torch.Tensor): pred = pred.cpu().detach().numpy()
        if len(pred.shape) == 5: pred = pred.squeeze(0)
        
        # Channel processing
        if pred.shape[0] == 22: pred = pred[-1:, ...]
        elif pred.shape[-1] == 22: pred = pred[..., -1:].transpose((3, 0, 1, 2))
        elif pred.shape[-1] == 1: pred = pred.transpose((3, 0, 1, 2))
        
        pred = pred.transpose((1, 2, 3, 0)) # (N, N, N, 1)
        pred_mask = (pred > threshold).squeeze()

        original_atoms = self.read_protein_features(original_pdb)
        if center is None: center = self.get_start_point(original_atoms) + (self.voxel_size * self.n_voxels / 2)
        
        start_point = center - (self.voxel_size * self.n_voxels / 2)
        voxel_centers = self.get_voxel_centers(start_point)
        bs_coords = voxel_centers[pred_mask]
        
        if len(bs_coords) == 0: return []
        
        atom_coords = np.array([a['coords'] for a in original_atoms])
        diff = bs_coords[:, None, :] - atom_coords[None, :, :]
        within = np.all(np.abs(diff) <= self.voxel_size/2, axis=2)
        _, atom_indices = np.where(within)
        
        result = []
        for idx in np.unique(atom_indices):
            a = original_atoms[idx]
            result.append((a['element'], a['coords'], a['resname'], a['resnum'], a['atom_name']))
        return result

    def get_pocket_aminoacids(self, original_pdb, predicted_atoms):
        if not predicted_atoms: return ""
        target_ids = set([a[3] for a in predicted_atoms])
        lines = []
        with open(original_pdb, 'r') as f:
            for line in f:
                if (line.startswith('ATOM') or line.startswith('HETATM')):
                    try:
                        if int(line[22:26].strip()) in target_ids: lines.append(line)
                    except: continue
        lines.append('TER\n')
        return "".join(lines)
        


def main():
    args = parse_arguments()

    pdb_vault = args.target_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    pdb_ids = sorted(os.listdir(pdb_vault))

    voxelizer = ProteinVoxelizer(voxel_size=args.voxel_size, n_voxels=args.n_voxels)

    for pdb_id in tqdm(pdb_ids, desc="Processing PDBs"):
        protein_path = os.path.join(pdb_vault, pdb_id, f"{pdb_id}_protein.pdb")
        pocket_path = os.path.join(pdb_vault, pdb_id, f"{pdb_id}_pocket.pdb")
        ligand_path = os.path.join(pdb_vault, pdb_id, f"{pdb_id}_ligand.sdf")

        if not all(os.path.exists(p) for p in [protein_path, pocket_path, ligand_path]):
            print(f"Warning: Missing files for PDB ID {pdb_id}. Skipping.")
            continue

        voxel_file_name = os.path.join(save_dir, f"{pdb_id}_voxel_label_dim22.pkl")
        center_file_name = os.path.join(save_dir, f"{pdb_id}_center_coords.pkl")

        if os.path.exists(voxel_file_name):
            continue

        voxel, label, center = voxelizer.voxelize_gpu_v2(
            protein_path=protein_path,
            pocket_path=pocket_path,
            r_cutoff=4.0,
            device=f"cuda:{args.device}",
            batch_size=8192,
            label_radius=args.label_radius,
        )

        # Concatenate features and label (21 + 1 = 22 channels)
        voxel_label_dim22 = np.concatenate((voxel, label), axis=3).astype(np.float16)

        # Save results
        with open(voxel_file_name, "wb") as fp:
            pickle.dump(voxel_label_dim22, fp)

        with open(center_file_name, "wb") as fp:
            pickle.dump(center, fp)

if __name__ == "__main__":
    main()