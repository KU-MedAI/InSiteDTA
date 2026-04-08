"""Convert Boltz-2 YAML inputs to ColabFold FASTA for protein-only structure prediction.

Reads Boltz-2 style YAML files (one per complex), extracts protein chain sequences,
and writes a single ColabFold-compatible FASTA file where multi-chain sequences are
joined by ':'.

Usage:
    python 01_yaml_to_fasta.py \
        --yaml_dir ../baselines/boltz-2/inputs \
        --output ./colabfold_input.fasta
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Boltz-2 YAML inputs to ColabFold FASTA"
    )
    parser.add_argument(
        "--yaml_dir", type=str, required=True,
        help="Directory containing Boltz-2 YAML files (one per complex)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output FASTA file path for ColabFold",
    )
    return parser.parse_args()


def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def extract_protein_sequences(yaml_data: dict) -> list[tuple[str, str]]:
    """Extract (chain_id, sequence) pairs for protein entries only.

    Boltz-2 can express homodimers as id: [B, E] — these are expanded
    into individual chains sharing the same sequence.

    Returns:
        List of (chain_id, sequence) tuples, sorted by chain_id.
    """
    proteins = []
    for entry in yaml_data.get("sequences", []):
        if "protein" not in entry:
            continue
        prot = entry["protein"]
        chain_id = prot.get("id", "?")
        sequence = prot.get("sequence", "")
        if not sequence:
            continue
        if isinstance(chain_id, list):
            for cid in chain_id:
                proteins.append((str(cid), sequence))
        else:
            proteins.append((str(chain_id), sequence))
    return sorted(proteins, key=lambda x: x[0])


def build_colabfold_entry(pdb_id: str, chain_sequences: list[tuple[str, str]]) -> str:
    """Build a single FASTA entry for ColabFold.

    Multi-chain: sequences joined by ':'.
    """
    joined = ":".join(seq for _, seq in chain_sequences)
    return f">{pdb_id}\n{joined}\n"


def collect_all_entries(yaml_dir: str) -> list[dict]:
    """Scan YAML directory and build ColabFold entries for all complexes."""
    yaml_files = sorted(Path(yaml_dir).glob("*.yaml"))
    if not yaml_files:
        yaml_files = sorted(Path(yaml_dir).glob("*.yml"))

    entries = []
    skipped = []
    for yf in yaml_files:
        pdb_id = yf.stem
        yaml_data = load_yaml(str(yf))
        proteins = extract_protein_sequences(yaml_data)

        if not proteins:
            skipped.append(pdb_id)
            continue

        entries.append({
            "pdb_id": pdb_id,
            "n_chains": len(proteins),
            "fasta_entry": build_colabfold_entry(pdb_id, proteins),
        })

    if skipped:
        print(f"[WARN] {len(skipped)} YAMLs skipped (no protein sequences): {skipped}")

    return entries


def write_fasta(entries: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(entry["fasta_entry"])


def print_stats(entries: list[dict]) -> None:
    dist = Counter(e["n_chains"] for e in entries)
    total = len(entries)

    print(f"\nTotal complexes: {total}")
    print(f"{'Chains':<10} {'Count':>6}  {'Ratio':>7}")
    print("-" * 26)
    for n in sorted(dist):
        print(f"{n:<10} {dist[n]:>6}  {dist[n]/total*100:>6.1f}%")

    multi = sum(1 for e in entries if e["n_chains"] > 1)
    print(f"\nMulti-chain: {multi}/{total} ({multi/total*100:.1f}%)")


def main():
    args = parse_args()

    print(f"Scanning YAMLs: {args.yaml_dir}")
    entries = collect_all_entries(args.yaml_dir)

    if not entries:
        print("No valid entries found. Exiting.")
        sys.exit(1)

    write_fasta(entries, args.output)
    print(f"FASTA written: {args.output} ({len(entries)} entries)")

    print_stats(entries)


if __name__ == "__main__":
    main()