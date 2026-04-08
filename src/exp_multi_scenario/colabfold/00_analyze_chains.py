"""Analyze how many protein chains are within 10A of the ligand in coreset_crystal.

Reads {pdb_id}_protein.pdb and {pdb_id}_ligand.sdf for each complex,
then reports per-complex chain counts and overall distribution.

Usage:
    python 00_analyze_chains.py
    python 00_analyze_chains.py --coreset_dir ../../data/coreset_crystal --cutoff 10.0
"""

import argparse
import os

import numpy as np
from rdkit import Chem


CUTOFF = 10.0


def get_arguments():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--coreset_dir", type=str,
                        default=os.path.join(script_dir, "../../data/coreset_crystal"))
    parser.add_argument("--cutoff", type=float, default=CUTOFF)
    return parser.parse_args()


def get_ligand_coords(sdf_path: str) -> np.ndarray:
    """Return heavy atom coordinates of the ligand from SDF.

    Uses sanitize=False to handle SDFs with aromatic bond type (1.5),
    which cause Kekulize failures under normal sanitization.
    Hydrogens are excluded by atomic number check instead of removeHs.
    """
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False, sanitize=False)
    if mol is None:
        return np.empty((0, 3))
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1:   # skip H
            continue
        coords.append(conf.GetAtomPosition(i))
    return np.array(coords) if coords else np.empty((0, 3))


def parse_protein_chains(pdb_path: str) -> dict[str, np.ndarray]:
    """Return {chain_id: heavy_atom_coords} from protein PDB (ATOM records only)."""
    chains: dict[str, list] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):   # HETATM(물 등) 제외
                continue
            element = line[76:78].strip()
            if element == "H":
                continue
            chain = line[21].strip()
            if not chain:                      # 공백 chain 제외
                continue
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            chains.setdefault(chain, []).append(coord)
    return {ch: np.array(coords) for ch, coords in chains.items()}


def chains_within_cutoff(chain_coords: dict[str, np.ndarray],
                          ligand_coords: np.ndarray,
                          cutoff: float) -> list[str]:
    """Return chain IDs that have at least one atom within cutoff of any ligand atom."""
    if len(ligand_coords) == 0:
        return []
    nearby = []
    for chain_id, coords in chain_coords.items():
        dists = np.linalg.norm(coords[:, None, :] - ligand_coords[None, :, :], axis=2)
        if dists.min() <= cutoff:
            nearby.append(chain_id)
    return sorted(nearby)


def main():
    args = get_arguments()
    coreset_dir = os.path.realpath(args.coreset_dir)

    results = []
    errors = []

    for pdb_id in sorted(os.listdir(coreset_dir)):
        d = os.path.join(coreset_dir, pdb_id)
        if not os.path.isdir(d):
            continue

        pdb_path = os.path.join(d, f"{pdb_id}_protein.pdb")
        sdf_path = os.path.join(d, f"{pdb_id}_ligand.sdf")

        if not os.path.exists(pdb_path) or not os.path.exists(sdf_path):
            errors.append(pdb_id)
            continue

        ligand_coords = get_ligand_coords(sdf_path)
        chain_coords = parse_protein_chains(pdb_path)
        nearby = chains_within_cutoff(chain_coords, ligand_coords, args.cutoff)

        results.append({
            "pdb_id": pdb_id,
            "total_chains": len(chain_coords),
            "nearby_chains": len(nearby),
            "nearby_chain_ids": nearby,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    from collections import Counter
    dist = Counter(r["nearby_chains"] for r in results)

    print(f"Coreset: {len(results)} complexes  (errors: {len(errors)})")
    print(f"Cutoff:  {args.cutoff} Å\n")
    print(f"{'Nearby chains':<16} {'Count':>6}  {'Ratio':>7}")
    print("-" * 32)
    for n in sorted(dist):
        cnt = dist[n]
        print(f"{n:<16} {cnt:>6}  {cnt/len(results)*100:>6.1f}%")

    print()
    multi = [r for r in results if r["nearby_chains"] > 1]
    print(f"Multi-chain (>1 chain within {args.cutoff}Å): {len(multi)} / {len(results)} "
          f"({len(multi)/len(results)*100:.1f}%)")

    print("\nPDB IDs with >1 nearby chain:")
    for r in multi:
        print(f"  {r['pdb_id']}: {r['nearby_chains']} chains {r['nearby_chain_ids']}")

    if errors:
        print(f"\nSkipped (missing files): {errors}")


if __name__ == "__main__":
    main()
