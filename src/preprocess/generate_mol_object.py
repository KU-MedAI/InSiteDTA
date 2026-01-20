import os
import argparse
import pickle
import pandas as pd
from tqdm import tqdm
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem


def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--smi_dir", type=str, help="SMILES 파일이 있는 디렉토리")
    parser.add_argument("--csv_path", type=str, help="pdbbind ligand info 디렉토리")
    parser.add_argument("--out_dir", type=str, required=True, help="결과물을 저장할 디렉토리")
    parser.add_argument("--target_numConfs", type=int, default=5, help="생성을 시도할 conformation 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    
    args = parser.parse_args()
    return args


def fix_smiles_with_openbabel(smiles):
    """Open Babel을 사용한 SMILES 수정"""
    try:
        # Open Babel로 분자 읽기
        mol = pybel.readstring("smi", smiles)
        
        # 수소 추가 및 최적화
        mol.addh()
        mol.make3D()  # 3D 구조 생성으로 유효성 검증
        
        # 정규화된 SMILES 출력
        canonical_smiles = mol.write("can").strip()  # canonical SMILES
        return canonical_smiles
    except:
        print("Failed to fix smiles with openbabel")
        return None


def get_smiles_paths(smi_dir):
    if not os.path.exists(smi_dir):
        raise FileNotFoundError(f"Cannot find smiles directory: {smi_dir}")
    smiles_paths = [os.path.join(smi_dir, f) for f in os.listdir(smi_dir) if f.endswith(".smi")]
    smiles_paths.sort()
    
    return smiles_paths


def generate_mol_object(smiles):   
    # mol 객체화 (with 유효성 검사)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            fixed_smiles = fix_smiles_with_openbabel(smiles)
            mol = Chem.MolFromSmiles(fixed_smiles)
            if mol is None:
                raise ValueError(f"Warning: failed to convert SMILES to mol object. Smiles: {smiles}")
            
    return mol


def generate_conformers(mol, target_numConfs=1, total_max_attempts=10, seed=42):
    try:
        mol = Chem.RemoveHs(mol)
        mol = Chem.AddHs(mol)

        # 여러 번 시도해서 원하는 개수 채우기
        curr_numConfs = 0
        attempts = 0
        while curr_numConfs < target_numConfs and attempts < total_max_attempts:
            AllChem.EmbedMultipleConfs(
                mol=mol,
                numConfs=target_numConfs + attempts*5,
                maxAttempts=1000 + attempts*300,
                randomSeed=42 + attempts,
                numThreads=min(40, os.cpu_count()//4),
                clearConfs=False
            )
            curr_numConfs = len(mol.GetConformers())
            attempts += 1

        if curr_numConfs < target_numConfs:
            print(f"Conformation 생성에 실패")
            return []
        
    except:
        print(f"Failed to generate conformations")
        return []
    
    return mol


def main(csv_path, smi_dir, out_dir, target_numConfs):
    if csv_path is not None and smi_dir is not None:
        raise ValueError("csv_path 와 smi_dir 중 하나로만 smiles 정보를 제공해 주세요")
    
    # BLACKLIST load
    blacklist_file = "failed_smiles_ids.txt"
    
    if not os.path.exists(blacklist_file):
        with open(blacklist_file, 'w') as fp:
            pass

    with open(blacklist_file, 'r') as fp:
        failed_smiles_ids = fp.read().splitlines()
        
    # smiles 경로 추출
    smiles_ids = []
    if smi_dir is not None:
        smiles_paths = get_smiles_paths(smi_dir)
        for smiles_path in smiles_paths:
            with open(smiles_path, 'r') as fp:
                smiles = fp.read().split()[0]
            smiles_id = os.path.basename(smiles_path).replace(".smi", "")
            smiles_ids.append((smiles, smiles_id))
        
    elif csv_path is not None:
        df = pd.read_csv(csv_path)
        for i in range(df.shape[0]):
            if df.columns[0] == "PDB_ID":
                smiles_id = df.iloc[i, 0] + "_ligand" # ID_ligand 형태
                smiles = df.iloc[i, 9] # canonical smiles
                smiles_ids.append((smiles, smiles_id))
            else:
                raise ValueError("Dataframe column이 맞지 않습니다")
        
        
    print(f"detected {len(smiles_ids)} of smiles files.")
    
    os.makedirs(out_dir, exist_ok=True)
    
    exist_count = 0
    for i, (smiles, smiles_id) in enumerate(smiles_ids):
        if smiles_id in failed_smiles_ids:
            exist_count += 1
            continue
        
        print(f"\rProcessing {smiles_id}...({exist_count + i}/{len(smiles_ids)})", end="", flush=True)
        out_path = os.path.join(out_dir, smiles_id + ".pkl")
        if os.path.exists(out_path):
            continue
            # with open(out_path, 'rb') as fp:
            #     try:
            #         mol = pickle.load(fp)
            #         continue
            #     except:
            #         pass
                
        mol = generate_mol_object(smiles)    
        mol_confs = generate_conformers(mol, target_numConfs=target_numConfs)

        if mol_confs == []:
            print(f"Warning: failed to generate 5 conformers. File: {smiles_id}, smiles: {smiles}")
            with open(blacklist_file, 'a') as fp:
                fp.write(smiles_id + "\n")
            continue
        
        with open(out_path, 'wb') as fp:
            pickle.dump(mol_confs, fp)
    
if __name__ == "__main__":
    args = get_arguments()
    main(args.csv_path, args.smi_dir, args.out_dir, args.target_numConfs)