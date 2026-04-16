import os, json, sys, random
import numpy as np

from tqdm import tqdm
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from .dataset import CustomDataset

def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def collate_as_dict(items: list) -> dict[str, any]:
    """
    Efficiently collates a list of dictionaries into batched tensors and objects.

    Args:
        items: List of dictionaries containing arrays, tensors, strings, or Data objects
    Returns:
        Dictionary mapping keys to batched tensors or Batch objects
    """
    if not items:
        return {}

    # Pre-compute batch size and get reference keys from first item
    batch_size = len(items)
    reference_item = items[0]

    # Initialize result containers for each key based on first item's types
    batches = {}
    for key, value in reference_item.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            # Pre-allocate maximum sized array by checking all items
            max_dims = [value.shape]
            for item in items[1:]:
                if key in item and isinstance(item[key], (np.ndarray, torch.Tensor)):
                    max_dims.append(item[key].shape)
            max_shape = np.maximum.reduce(max_dims)
            batches[key] = np.zeros((batch_size, *max_shape))

        elif isinstance(value, str):
            batches[key] = ["" for _ in range(batch_size)]

        elif isinstance(value, Data):
            # Create list to collect Data objects for batch conversion
            batches[key] = []

        elif isinstance(value, float):
            batches[key] = torch.zeros(batch_size, dtype=torch.float32)

        else:
            batches[key] = np.zeros(batch_size)

    # Single pass to fill batches
    for batch_idx, item in enumerate(items):
        if item is None:
            continue

        for key, value in item.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                # Efficient slicing using pre-computed shapes
                slices = tuple(slice(0, dim) for dim in value.shape)
                batches[key][(batch_idx, *slices)] = value

            elif isinstance(value, Data):
                # Collect Data objects for batch conversion
                batches[key].append(value)

            elif isinstance(batches[key], torch.Tensor):
                batches[key][batch_idx] = torch.tensor(value, dtype=torch.float32)

            else:
                batches[key][batch_idx] = value

    # Post-process batches
    result = {}
    for key, batch in batches.items():
        if isinstance(reference_item[key], np.ndarray):
            # Convert numpy arrays to PyTorch tensors
            result[key] = torch.from_numpy(batch).float()
        elif isinstance(reference_item[key], Data):
            # Convert collected Data objects to Batch
            result[key] = Batch.from_data_list(batch)
        elif isinstance(batch, np.ndarray):
            result[key] = torch.from_numpy(batch).float()
        else:
            result[key] = batch

    return result


class MasterDataLoader:
    def __init__(self, data_cfg, seed, batch_size, num_workers, num_conformers=5):

        self.index_dict = {}
        if data_cfg["index_file"]:
            with open(data_cfg["index_file"], "r") as fp:
                self.index_dict = json.load(fp)

        self.vox_dir = data_cfg["vox_dir"]
        self.lig_dir = data_cfg["lig_dir"]

        self.tr_keys = data_cfg["tr_keys"]
        self.vl_keys = data_cfg["vl_keys"]
        self.ts_keys = data_cfg["ts_keys"]

        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_conformers = num_conformers

    def get_zip_paths(self, keys: list, desc: Literal["training", "validation", "test"]) -> list[tuple[str, str]]:
        # Find preprocessed voxel and ligand files and matching them

        vox_lig_pairs = []
        valid_vox_count = 0
        valid_lig_count = 0

        for k in tqdm(keys, desc=f"Scanning {desc} keys"):
            vox_path = os.path.join(self.vox_dir, f"{k}_voxel.pkl")
            lig_path = os.path.join(self.lig_dir, f"{k}_ligand.pkl")
            
            if os.path.exists(vox_path):
                valid_vox_count += 1
                
            if os.path.exists(lig_path):
                valid_lig_count += 1

            if os.path.exists(vox_path) and os.path.exists(lig_path):
                vox_lig_pairs.append((vox_path, lig_path))

        print(f"  Target : {len(keys)} keys")
        print(f"  Voxel : {valid_vox_count} / {len(keys)} found from {self.vox_dir}")
        print(f"  Ligand : {valid_lig_count} / {len(keys)} found from {self.lig_dir}")
        print(f"  Matched : {len(vox_lig_pairs)} pairs\n")
        
        return vox_lig_pairs
        
    def get_tr_vl_dataset(self):
        tr_zip_paths = self.get_zip_paths(
            self.tr_keys,
            desc="training",
        )
        vl_zip_paths = self.get_zip_paths(
            self.vl_keys,
            desc="validation",
        )

        tr_dataset = CustomDataset(
            zip_paths=tr_zip_paths, index_dict=self.index_dict, desc="training", num_conformers=self.num_conformers
        )
        vl_dataset = CustomDataset(
            zip_paths=vl_zip_paths, index_dict=self.index_dict, desc="validation", num_conformers=self.num_conformers
        )
        return tr_dataset, vl_dataset

    def get_tr_vl_loader(self):
        tr_dataset, vl_dataset = self.get_tr_vl_dataset()
        
        if self.seed != -1:
            tr_g = seed_generator(self.seed)
            vl_g = seed_generator(self.seed + 1)
        else:
            tr_g, vl_g = None, None

        tr_loader = DataLoader(
            tr_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_as_dict,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=tr_g,
            pin_memory=True,
            shuffle=True,
        )

        vl_loader = DataLoader(
            vl_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_as_dict,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=vl_g,
            pin_memory=True,
            shuffle=False,
        )

        return tr_loader, vl_loader

    def get_ts_dataset(self):
        ts_zip_paths = self.get_zip_paths(
            self.ts_keys,
            desc="test",
        )

        ts_dataset = CustomDataset(
            zip_paths=ts_zip_paths, index_dict=self.index_dict, desc="test", num_conformers=self.num_conformers
        )
        return ts_dataset

    def get_ts_loader(self):
        ts_dataset = self.get_ts_dataset()

        if self.seed != -1:
            ts_g = seed_generator(self.seed + 2)
        else:
            ts_g = None
            
        ts_loader = DataLoader(
            ts_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_as_dict,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=ts_g,
            pin_memory=True,
            shuffle=False,
        )

        return ts_loader
