# InSiteDTA — Multi-Scenario Evaluation Pipeline

ISMB 2026 논문 리비전을 위한 multi-model, multi-split, multi-scenario 평가 파이프라인.
Branch: `feat/multi-scenario-eval-pipeline`

---

## 1. 프로젝트 구조

```
InSiteDTA/                              ← repo root (모든 명령은 여기서 실행)
├── CLAUDE.md                           ← 이 파일
├── src/
│   ├── data/
│   │   ├── coreset_{scenario}/         ← 시나리오별 평가 데이터 (crystal, redocked, p2rank, alphafold, boltz2)
│   │   │   └── {pdb_id}/              ← {pdb_id}_protein.pdb, {pdb_id}_pocket.pdb, {pdb_id}_ligand.{sdf}
│   │   ├── datasplit_preset/           ← split 정의 JSON (data_config_PDBbind2016.json, data_config_CleanSplit0.json)
│   │   └── index/                      ← affinity_index, smiles_csv 등
│   └── exp_multi_scenario/
│       ├── configs/
│       │   └── eval_config.yaml        ← 통합 설정 (splits, models, scenarios)
│       ├── baselines/
│       │   ├── eval_interface.py       ← 공유 유틸 (CLI, metrics, result JSON 빌드)
│       │   ├── DeepDTA/  → symlink     ← /data/hawon/InSiteDTA_baselines/DeepDTA
│       │   ├── CheapNet/ → symlink     ← /data/hawon/InSiteDTA_baselines/CheapNet
│       │   ├── Pafnucy/  → symlink     ← /data/hawon/InSiteDTA_baselines/Pafnucy
│       │   ├── CAPLA/    → symlink     ← /data/hawon/InSiteDTA_baselines/CAPLA
│       │   └── PLANET/   → symlink     ← /data/hawon/InSiteDTA_baselines/PLANET
│       └── results/
│           └── {Model}/{split}/{scenario}.json
│
/data/hawon/InSiteDTA_baselines/        ← 대용량 데이터/체크포인트 (NFS)
├── raw/                                ← PDBbind 2020 raw data
├── DeepDTA/                            ← DeepDTA 코드 + hw_ 파일 + ckpts
├── CheapNet/                           ← CheapNet 코드 + hw_ 파일 + ckpts
├── Pafnucy/                            ← Pafnucy 코드 + hw_ 파일 + ckpts + chimera + tfbio
├── PLANET/                             ← PLANET 코드 + hw_ 파일 + PLANET.param (pretrained)
├── CAPLA/                              ← CAPLA 코드 + hw_ 파일 + hw_capla.py (클린 모델)
└── GEMS/                               ← GEMS 코드 + hw_ 파일 + .venv
```

---

## 2. 베이스라인 모델 목록


| Model     | 유형             | 환경      | Python 경로                                         | 상태     |
| --------- | -------------- | ------- | ------------------------------------------------- | ------ |
| InSiteDTA | Complex-based  | uv venv | `/data/hawon/venvs/insitedta/bin/python`          | 평가 완료  |
| DeepDTA   | Sequence-based | uv venv | `/data/hawon/venvs/deepdta/bin/python`            | 평가 완료  |
| CheapNet  | Complex-based  | conda   | `/home/tech/anaconda3/envs/cheapcross/bin/python` | 평가 진행중 |
| Pafnucy   | Complex-based  | conda   | `conda run -n pafnucy_a100`                       | 파이프라인 작성 완료 |
| PLANET    | Complex-free   | conda   | `conda run -n planet`                             | 파이프라인 작성 완료 |
| CAPLA     | Complex-free   | conda   | `conda run -n capla`                              | 파이프라인 작성 완료 |
| GEMS      | Complex-based  | uv venv | `/data/hawon/InSiteDTA_baselines/GEMS/.venv/bin/python` | 파이프라인 작성 완료 |


---

## 3. 베이스라인 파이프라인 규칙

### 3.1 파일 명명 규칙

- 우리가 추가한 모든 파일/디렉토리는 `hw_` prefix 사용 (원본과 구분)
- 각 모델의 파이프라인 파일 구성:
  ```
  hw_01_input_prep.py     ← 데이터 전처리 (모델별 입력 형식 생성)
  hw_02_train.py          ← 학습 (split_config 기반 train/val/test 분할)
  hw_eval_wrapper.py      ← 평가 (eval_interface 준수)
  hw_run_all.sh           ← 실행 명령어 모음 (참조용)
  ```

### 3.2 eval_wrapper 작성 규칙 (eval_interface.py 준수)

**필수 import:**

```python
from eval_interface import (
    build_parser, load_eval_config, load_split_config,
    compute_metrics, aggregate_metrics,
    build_result, save_result, print_metrics,
)
```

**CLI 인자:**

- `build_parser(model_name)`으로 기본 인자 생성 후 모델별 인자 추가
- 기본 인자: `--eval_config`, `--split`, `--scenario`, `--output_json`, `--device`, `--batch_size`
- 모델별 추가 인자: `--ckpt_dir`, `--graph_dir`, `--raw_csv` 등

**메트릭 계산:**

- 반드시 `compute_metrics(preds, targets)` 사용 → PCC, RMSE, MAE 반환
- per-checkpoint 로그에 세 지표 모두 출력: `PCC = ... RMSE = ... MAE = ...` (띄어쓰기 있을것)
- 여러 checkpoint 결과는 `aggregate_metrics()` → mean, std

**로그 헤더/푸터 (for문 실행 시 구분용):**

```python
# main() 시작
print(f"\n===== {ModelName} | split={args.split} | scenario={args.scenario} =====")
# ... 평가 로직 ...
# main() 종료
print(f"===== {ModelName} | split={args.split} | scenario={args.scenario} | DONE =====\n")
```

**결과 JSON 출력:**

```python
result = build_result(model_name, args.split, args.scenario,
                      ckpt_paths, agg_metrics, per_sample)
save_result(result, args.output_json)
print_metrics(result)
```

**결과 파일 경로:** `src/exp_multi_scenario/results/{Model}/{split}/{scenario}.json`

**결과 JSON 스키마:**

```json
{
  "model": "ModelName",
  "split": "original",
  "scenario": "crystal",
  "ckpts_used": ["path/to/ckpt1.pt", ...],
  "metrics": {
    "PCC":  {"mean": 0.78, "std": 0.01},
    "RMSE": {"mean": 1.20, "std": 0.05},
    "MAE":  {"mean": 0.95, "std": 0.03}
  },
  "per_sample": [ # predictions from last ckpt
    {"pdb_id": "1a30", "pred": 5.12, "true": 5.30}, ...
  ]
}
```

### 3.3 main() 구조 규칙

`main()`은 최대한 **함수 호출의 나열**로 구성. 로직을 main에 직접 쓰지 않는다.

```python
def main():
    args       = parse_args()
    eval_cfg   = load_eval_config(args.eval_config)
    split_cfg  = load_split_config(eval_cfg, args.split)
    # ... model-specific data loading ...
    # ... checkpoint discovery ...
    # ... inference loop ...
    result     = build_result(...)
    save_result(result, args.output_json)
    print_metrics(result)
```

### 3.4 데이터 경로 규칙

- **coreset 데이터**: `src/data/coreset_{scenario}/{pdb_id}/` (repo 내, 공유)
  - 파일: `{pdb_id}_protein.pdb`, `{pdb_id}_pocket.pdb`, `{pdb_id}_ligand.sdf`, `{pdb_id}_ligand.mol2` 단 mol2 는 일부만 존재
  - coreset 디렉토리 내 파일 생성/수정 금지 → 전처리 결과는 모델별 output_dir에 저장
- **split 정의**: `src/data/datasplit_preset/data_config_{split}.json`
  - `tr_keys`, `vl_keys`, `ts_keys` 포함
- **affinity 라벨**: `src/data/index/affinity_index_pdbbind2020.json`
- **PDBbind raw data**: `/data/hawon/InSiteDTA_baselines/raw/`
- **모델별 전처리 결과/checkpoint**: `/data/hawon/InSiteDTA_baselines/{Model}/hw_`*

### 3.5 Split 규칙


| Split      | Train data                   | Test data             | 특징                        |
| ---------- | ---------------------------- | --------------------- | ------------------------- |
| original   | PDBbind 2016 general+refined | CASF2016 coreset 285개 | 표준 벤치마크                   |
| cleansplit | CleanSplit0 (overlap 제거)     | CASF2016 coreset 285개 | protein/ligand overlap 제거 |


- 두 split의 test set은 **동일한 285개** CASF2016 coreset
- 차이는 training data 구성만 다름

### 3.6 Scenario 규칙


| Scenario  | Protein source  | Ligand source        | 비고                                           |
| --------- | --------------- | -------------------- | -------------------------------------------- |
| crystal   | 실험 구조 (PDB)     | 실험 위치                | ground truth                                 |
| redocked  | 실험 구조 (PDB)     | 정답 binding site에 도킹  | ligand pose만 다름                              |
| p2rank    | 실험 구조 (PDB)     | P2Rank 예측 pocket에 도킹 | pocket, ligand pose 다름                       |
| alphafold | AlphaFold 예측 구조 | P2Rank 예측 pocket에 도킹 | AlphaFold multimer로 protein multichain 구조 예측 |
| boltz2    | Boltz-2 예측 구조   | Boltz-2 예측 구조        | Boltz-2 는 protein-ligand complex 구조 예측       |


- Sequence-based 모델 (DeepDTA): scenario에 무관하게 동일 입력
- Complex-based, free 모델: scenario별로 3D 입력이 달라짐 → 전처리 필수

### 3.7 Complex-based 모델 전처리 규칙

- **pocket 생성**: 반드시 베이스라인의 원본 코드가 사용한 방법과 동일하게 생성
  - 예시) CheapNet: PyMOL `byres ligand around 5` → `Pocket_5A.pdb`
- **전처리 결과 저장 위치 예시**: `/data/hawon/InSiteDTA_baselines/{Model}/hw_{graphs}/{scenario}/`
  - coreset 원본 디렉토리를 오염시키지 않음
- **RDKit parse fallback**: `sanitize=False` fallback 포함 (일부 pocket에서 필요)

### 3.8 학습 코드 (hw_02_train.py) 규칙

- `--split_config`: data split JSON 경로 (tr_keys/vl_keys/ts_keys)
- `--seed -1`: 랜덤 시드 (실제 사용된 seed가 파일명에 포함)
- checkpoint 저장: `hw_ckpts/{split}/` 디렉토리
  - 파일명에 seed 포함: `{model}_seed{actual_seed}.pt`
- original split에 기존 학습된 checkpoint이 있으면 재사용 가능
- 하이퍼파라미터는 레퍼런스 논문 기본값과 동일하게 설정

### 3.9 환경 관리

- 각 모델은 독립된 가상환경 사용 (uv venv 또는 conda)
- `eval_config.yaml`에 python 경로 명시
- conda 환경 사용 시: `conda run -n {env_name} python ...`으로 실행
- 모든 명령은 **InSiteDTA repo root**에서 실행

### 3.10 Symlink 규칙

- `src/exp_multi_scenario/baselines/{Model}/` → `/data/hawon/InSiteDTA_baselines/{Model}/`
- 모든 모델 디렉토리는 symlink으로 연결 (일관성 유지)
- `hw_eval_wrapper.py`는 물리적으로 baselines 디렉토리에 위치하되 symlink을 통해 접근

### 3.11 Import 경로 규칙

각 eval_wrapper에서 레포 원본 코드와 eval_interface를 import하기 위해:

```python
BASELINES_DIR = "/home/tech/Hawon/InSiteDTA/src/exp_multi_scenario/baselines"
MODEL_ROOT    = "/data/hawon/InSiteDTA_baselines/{Model}"
sys.path.insert(0, os.path.join(MODEL_ROOT, "{subdir}"))
sys.path.insert(0, BASELINES_DIR)
```

- 절대 경로 사용 (symlink 해석 문제 방지)
- `__file__` 기반 상대 경로 사용 금지

---

## 4. 실행 명령어 (repo root에서 실행)

hw_01_input_prep.py는 학습/평가 데이터 모두 처리 가능. `--data_dir`에 PDBbind raw 또는 coreset 경로를 넣으면 됨.

### 4.1 DeepDTA

python: `/data/hawon/venvs/deepdta/bin/python`

```bash
# 전처리 (sequence-based → raw CSV만 필요, 한 번만 실행)
/data/hawon/venvs/deepdta/bin/python /data/hawon/InSiteDTA_baselines/DeepDTA/hw_01_input_prep.py \
    --yaml_dir /data/hawon/InSiteDTA_baselines/DeepDTA/hw_inputs \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output /data/hawon/InSiteDTA_baselines/DeepDTA/hw_deepdta_raw.csv

# 학습
/data/hawon/venvs/deepdta/bin/python /data/hawon/InSiteDTA_baselines/DeepDTA/hw_02_train.py \
    --raw_csv /data/hawon/InSiteDTA_baselines/DeepDTA/hw_deepdta_raw.csv \
    --split_config src/data/datasplit_preset/data_config_{split}.json \
    --output_dir /data/hawon/InSiteDTA_baselines/DeepDTA/hw_ckpts/{split} \
    --seed -1

# 평가
/data/hawon/venvs/deepdta/bin/python src/exp_multi_scenario/baselines/DeepDTA/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split {split} --scenario {scenario} \
    --ckpt_dir /data/hawon/InSiteDTA_baselines/DeepDTA/hw_ckpts/{split} \
    --raw_csv /data/hawon/InSiteDTA_baselines/DeepDTA/hw_deepdta_raw.csv \
    --output_json src/exp_multi_scenario/results/DeepDTA/{split}/{scenario}.json \
    --device 0
```

### 4.2 CheapNet

env: `conda run -n cheapcross`

```bash
# 전처리 — 학습용 (전체 PDBbind, 한 번만 실행)
conda run -n cheapcross python /data/hawon/InSiteDTA_baselines/CheapNet/hw_01_input_prep.py \
    --data_dir /data/hawon/InSiteDTA_baselines/raw \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir /data/hawon/InSiteDTA_baselines/CheapNet/hw_all_graphs

# 전처리 — 평가용 (scenario별)
conda run -n cheapcross python /data/hawon/InSiteDTA_baselines/CheapNet/hw_01_input_prep.py \
    --data_dir src/data/coreset_{scenario} \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir /data/hawon/InSiteDTA_baselines/CheapNet/hw_graphs/{scenario}

# 학습
conda run -n cheapcross python /data/hawon/InSiteDTA_baselines/CheapNet/hw_02_train.py \
    --data_dir /data/hawon/InSiteDTA_baselines/CheapNet/hw_all_graphs \
    --split_config src/data/datasplit_preset/data_config_{split}.json \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir /data/hawon/InSiteDTA_baselines/CheapNet/hw_ckpts/{split} \
    --seed -1

# 평가 (original — 기존 checkpoint)
conda run -n cheapcross python src/exp_multi_scenario/baselines/CheapNet/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split original --scenario {scenario} \
    --ckpt_dir /data/hawon/InSiteDTA_baselines/CheapNet/cross_dataset/Cross_best_models \
    --graph_dir /data/hawon/InSiteDTA_baselines/CheapNet/hw_graphs/{scenario} \
    --output_json src/exp_multi_scenario/results/CheapNet/original/{scenario}.json \
    --device 0

# 평가 (cleansplit 또는 hw_02로 학습한 checkpoint)
conda run -n cheapcross python src/exp_multi_scenario/baselines/CheapNet/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split {split} --scenario {scenario} \
    --ckpt_dir /data/hawon/InSiteDTA_baselines/CheapNet/hw_ckpts/{split} \
    --graph_dir /data/hawon/InSiteDTA_baselines/CheapNet/hw_graphs/{scenario} \
    --output_json src/exp_multi_scenario/results/CheapNet/{split}/{scenario}.json \
    --device 0
```

### 4.3 Pafnucy

env: `conda run -n pafnucy_a100`

```bash
# 전처리 — 학습용 (전체 PDBbind, 한 번만 실행)
conda run -n pafnucy_a100 python /data/hawon/InSiteDTA_baselines/Pafnucy/hw_01_input_prep.py \
    --data_dir       /data/hawon/InSiteDTA_baselines/raw \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir     /data/hawon/InSiteDTA_baselines/Pafnucy/hw_features/all \
    --chimera_path   /data/hawon/InSiteDTA_baselines/Pafnucy/chimera/bin/chimera

# 전처리 — 평가용 (scenario별)
conda run -n pafnucy_a100 python /data/hawon/InSiteDTA_baselines/Pafnucy/hw_01_input_prep.py \
    --data_dir       src/data/coreset_{scenario} \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir     /data/hawon/InSiteDTA_baselines/Pafnucy/hw_features/{scenario} \
    --chimera_path   /data/hawon/InSiteDTA_baselines/Pafnucy/chimera/bin/chimera

# 학습
conda run -n pafnucy_a100 python /data/hawon/InSiteDTA_baselines/Pafnucy/hw_02_train.py \
    --data_dir       /data/hawon/InSiteDTA_baselines/Pafnucy/hw_features/all \
    --split_config   src/data/datasplit_preset/data_config_{split}.json \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir     /data/hawon/InSiteDTA_baselines/Pafnucy/hw_ckpts/{split} \
    --seed -1 --device 0

# 평가 (pretrained checkpoint 검증)
conda run -n pafnucy_a100 python src/exp_multi_scenario/baselines/Pafnucy/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split original --scenario {scenario} \
    --ckpt_dir  /data/hawon/InSiteDTA_baselines/Pafnucy/hw_pretrained \
    --hdf_dir   /data/hawon/InSiteDTA_baselines/Pafnucy/hw_features/{scenario} \
    --output_json src/exp_multi_scenario/results/Pafnucy/original/{scenario}.json \
    --device 0

# 평가 (hw_02로 학습한 checkpoint)
conda run -n pafnucy_a100 python src/exp_multi_scenario/baselines/Pafnucy/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split {split} --scenario {scenario} \
    --ckpt_dir  /data/hawon/InSiteDTA_baselines/Pafnucy/hw_ckpts/{split} \
    --hdf_dir   /data/hawon/InSiteDTA_baselines/Pafnucy/hw_features/{scenario} \
    --output_json src/exp_multi_scenario/results/Pafnucy/{split}/{scenario}.json \
    --device 0
```

### 4.4 CAPLA

env: `conda run -n capla`

```bash
# 전처리 — 학습용 (전체 PDBbind, 한 번만 실행)
conda run -n capla python /data/hawon/InSiteDTA_baselines/CAPLA/hw_01_input_prep.py \
    --data_dir       /data/hawon/InSiteDTA_baselines/raw \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --smiles_csv     src/data/index/ligand_smiles_pdbbind2020.csv \
    --output_dir     /data/hawon/InSiteDTA_baselines/CAPLA/hw_features/all

# 전처리 — 평가용 (scenario별)
conda run -n capla python /data/hawon/InSiteDTA_baselines/CAPLA/hw_01_input_prep.py \
    --data_dir       src/data/coreset_{scenario} \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --smiles_csv     src/data/index/ligand_smiles_pdbbind2020.csv \
    --output_dir     /data/hawon/InSiteDTA_baselines/CAPLA/hw_features/{scenario}

# 학습
conda run -n capla python /data/hawon/InSiteDTA_baselines/CAPLA/hw_02_train.py \
    --data_dir       /data/hawon/InSiteDTA_baselines/CAPLA/hw_features/all \
    --split_config   src/data/datasplit_preset/data_config_{split}.json \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir     /data/hawon/InSiteDTA_baselines/CAPLA/hw_ckpts/{split} \
    --seed -1 --device 0

# 평가 (pretrained checkpoint 검증)
conda run -n capla python src/exp_multi_scenario/baselines/CAPLA/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split original --scenario {scenario} \
    --ckpt_dir  /data/hawon/InSiteDTA_baselines/CAPLA/src/CAPLA/saveModel/CAPLA_bestModel \
    --feat_dir  /data/hawon/InSiteDTA_baselines/CAPLA/hw_features/{scenario} \
    --output_json src/exp_multi_scenario/results/CAPLA/original/{scenario}.json \
    --device 0

# 평가 (hw_02로 학습한 checkpoint)
conda run -n capla python src/exp_multi_scenario/baselines/CAPLA/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split {split} --scenario {scenario} \
    --ckpt_dir  /data/hawon/InSiteDTA_baselines/CAPLA/hw_ckpts/{split} \
    --feat_dir  /data/hawon/InSiteDTA_baselines/CAPLA/hw_features/{scenario} \
    --output_json src/exp_multi_scenario/results/CAPLA/{split}/{scenario}.json \
    --device 0
```

### 4.5 PLANET

env: `conda run -n planet`

```bash
# 전처리 — 학습용 (전체 PDBbind, 한 번만 실행)
conda run -n planet python /data/hawon/InSiteDTA_baselines/PLANET/hw_01_input_prep.py \
    --data_dir       /data/hawon/InSiteDTA_baselines/raw \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir     /data/hawon/InSiteDTA_baselines/PLANET/hw_pickles/all \
    --n_jobs 8

# 전처리 — 평가용 (scenario별)
conda run -n planet python /data/hawon/InSiteDTA_baselines/PLANET/hw_01_input_prep.py \
    --data_dir       src/data/coreset_{scenario} \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir     /data/hawon/InSiteDTA_baselines/PLANET/hw_pickles/{scenario} \
    --n_jobs 8

# 학습
conda run -n planet python /data/hawon/InSiteDTA_baselines/PLANET/hw_02_train.py \
    --data_dir       /data/hawon/InSiteDTA_baselines/PLANET/hw_pickles/all \
    --split_config   src/data/datasplit_preset/data_config_{split}.json \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir     /data/hawon/InSiteDTA_baselines/PLANET/hw_ckpts/{split} \
    --seed -1 --device 0

# 평가 (pretrained checkpoint 검증)
conda run -n planet python src/exp_multi_scenario/baselines/PLANET/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split original --scenario {scenario} \
    --ckpt_dir  /data/hawon/InSiteDTA_baselines/PLANET \
    --pkl_dir   /data/hawon/InSiteDTA_baselines/PLANET/hw_pickles/{scenario} \
    --output_json src/exp_multi_scenario/results/PLANET/original/{scenario}.json \
    --device 0

# 평가 (hw_02로 학습한 checkpoint)
conda run -n planet python src/exp_multi_scenario/baselines/PLANET/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split {split} --scenario {scenario} \
    --ckpt_dir  /data/hawon/InSiteDTA_baselines/PLANET/hw_ckpts/{split} \
    --pkl_dir   /data/hawon/InSiteDTA_baselines/PLANET/hw_pickles/{scenario} \
    --output_json src/exp_multi_scenario/results/PLANET/{split}/{scenario}.json \
    --device 0
```

### 4.6 GEMS

python: `/data/hawon/InSiteDTA_baselines/GEMS/.venv/bin/python`

```bash
# 전처리 — 학습용 (전체 PDBbind, 한 번만 실행)
/data/hawon/InSiteDTA_baselines/GEMS/.venv/bin/python \
    /data/hawon/InSiteDTA_baselines/GEMS/hw_01_input_prep.py \
    --data_dir /data/hawon/InSiteDTA_baselines/raw \
    --output_dir /data/hawon/InSiteDTA_baselines/GEMS/hw_flat/all

# 전처리 — 평가용 (scenario별)
/data/hawon/InSiteDTA_baselines/GEMS/.venv/bin/python \
    /data/hawon/InSiteDTA_baselines/GEMS/hw_01_input_prep.py \
    --data_dir src/data/coreset_{scenario} \
    --output_dir /data/hawon/InSiteDTA_baselines/GEMS/hw_flat/{scenario}
    # redocked, p2rank 등 multi-pose SDF: --first_ligand_only 추가

# 학습 (3 seeds × 2 splits)
/data/hawon/InSiteDTA_baselines/GEMS/.venv/bin/python \
    /data/hawon/InSiteDTA_baselines/GEMS/hw_02_train.py \
    --data_dir /data/hawon/InSiteDTA_baselines/GEMS/hw_flat/all \
    --split_config src/data/datasplit_preset/data_config_{split}.json \
    --affinity_index src/data/index/affinity_index_pdbbind2020.json \
    --output_dir /data/hawon/InSiteDTA_baselines/GEMS/hw_ckpts/{split} \
    --seed -1 --device 0

# 평가
/data/hawon/InSiteDTA_baselines/GEMS/.venv/bin/python \
    src/exp_multi_scenario/baselines/GEMS/hw_eval_wrapper.py \
    --eval_config src/exp_multi_scenario/configs/eval_config.yaml \
    --split {split} --scenario {scenario} \
    --ckpt_dir /data/hawon/InSiteDTA_baselines/GEMS/hw_ckpts/{split} \
    --graph_dir /data/hawon/InSiteDTA_baselines/GEMS/hw_flat/{scenario} \
    --output_json src/exp_multi_scenario/results/GEMS/{split}/{scenario}.json \
    --device 0
```

---

## 5. 모델별 세부 사항

### DeepDTA

- **유형**: Sequence-based (protein seq + SMILES → pKd)
- **환경**: uv venv (`/data/hawon/venvs/deepdta/bin/python`)
- **scenario 무관**: 입력이 서열이므로 scenario별 차이 없음
- **추가 인자**: `--ckpt_dir`, `--raw_csv`
- **vocab**: train+val+test 전체에서 빌드 (character-level, set 기반 → 실행마다 순서 다름)
- **checkpoint**: `hw_ckpts/{split}/deepdta_seed{N}.pt` + `vocab_seed{N}.json` + `config_seed{N}.json`
- **기존 ckpt**: `hw_ckpts/pdbbind2016/` (3 seeds), `hw_ckpts/cleansplit/` (3 seeds)

### CheapNet

- **유형**: Complex-based GNN (pocket graph + ligand graph + interaction edges)
- **환경**: conda cheapcross (`/home/tech/anaconda3/envs/cheapcross/bin/python`)
- **전처리 필수**: coreset → PyMOL pocket → obabel ligand → .rdkit → .pyg
- **추가 인자**: `--ckpt_dir`, `--graph_dir`
- **기존 ckpt (original)**: `cross_dataset/Cross_best_models/repeat{0,1,2}/model/` (repeat 구조)
- **pocket 생성**: 반드시 PyMOL 사용 (`generate_pocket` from `preprocessing.py`)
  - `preprocessing.py`의 `generate_pocket`에 `save_dir`, `input_ligand_format` 파라미터 추가됨

### Pafnucy

- **유형**: Complex-based 3D CNN (voxel grid 21×21×21×19)
- **환경**: conda pafnucy_a100 (nvidia-tensorflow 1.15.5)
- **원본 레포**: `/data/hawon/InSiteDTA_baselines/pafnucy/` (소문자)
- **파이프라인 디렉토리**: `/data/hawon/InSiteDTA_baselines/Pafnucy/` (대문자, hw_ 파일 포함)
- **전처리 필수**: Chimera로 pocket PDB→MOL2 (addh, addcharge), obabel로 ligand SDF→MOL2, tfbio Featurizer로 HDF5 생성
- **추가 인자**: `--ckpt_dir`, `--hdf_dir`, `--charge_scaler`
- **pocket 출처**: PDBbind 제공 `{pdb_id}_pocket.pdb` 사용 (PyMOL 생성 아님)
- **tfbio**: pip 미설치, `/data/hawon/InSiteDTA_baselines/Pafnucy/tfbio/`에서 sys.path로 import
- **chimera**: `/data/hawon/InSiteDTA_baselines/Pafnucy/chimera/bin/chimera`
- **charge_scaler**: 학습 시 training set partial charge의 std로 계산, `charge_scaler.json`으로 저장. pretrained 기본값 = 0.425896
- **checkpoint**: TF1 SavedModel 형식 (`.meta` + `.index` + `.data-*`)
  - pretrained: `hw_pretrained/batch5-2017-06-05T07:58:47-best.*`
  - 학습: `hw_ckpts/{split}/pafnucy_seed{N}.ckpt.*` + `charge_scaler.json`
- **하이퍼파라미터**: Adam lr=1e-5, batch=20, 20 epochs × 24 rotations, dropout 0.5, L2 λ=0.001
- **아키텍처**: Conv(64,128,256) 5×5×5 + MaxPool 2×2×2 → Dense(1000,500,200) → 1

### CAPLA

- **유형**: Complex-free (protein seq 40D + pocket seq 40D + SMILES → pKd)
- **환경**: conda capla (Python 3.6, PyTorch 1.10)
- **원본 레포**: `/data/hawon/InSiteDTA_baselines/CAPLA/src/CAPLA/`
- **파이프라인 디렉토리**: `/data/hawon/InSiteDTA_baselines/CAPLA/` (hw_ 파일 포함)
- **전처리 필수**: protein PDB → DSSP (mkdssp v3) → 40D features (CSV), pocket PDB → 잔기 필터링
- **추가 인자**: `--ckpt_dir`, `--feat_dir`
- **40D feature 구성**: physicochemical 4D + Shen 7-cluster 7D + DSSP 8-state 8D + AA one-hot 21D
  - Shen 7-cluster 순서: c2_1=AGV, c2_2=ILFP, c2_3=YMTS, c2_4=HNQW, c2_5=RK, c2_6=DE, c2_7=C
  - Cys는 polar로 분류 (non_polar 아님)
- **DSSP**: mkdssp v3.0.0 사용 (원 저자는 v2 추정, 재학습으로 일관성 유지)
- **hw_capla.py**: 원본 capla.py + self_attention.py의 클린 버전 (debug np.save 제거)
- **SMILES 인코딩**: character-level, 64-char vocab (CHAR_SMI_SET), max_len=150
- **pocket 출처**: coreset `{pdb_id}_pocket.pdb` 사용 → global features에서 해당 잔기 필터링
- **checkpoint**: PyTorch state_dict
  - pretrained: `src/CAPLA/saveModel/CAPLA_bestModel/best_model.pt`
  - 학습: `hw_ckpts/{split}/capla_seed{N}.pt`
- **하이퍼파라미터**: AdamW lr=1e-3 (default), batch=256, 40 epochs, save_best after epoch 35, MSELoss(sum)
- **Mixed precision**: torch.cuda.amp (원본 apex.amp O1 대체)
- **아키텍처**: DilatedConv(seq) + Conv(pkt) + Cross-Attention(smi↔pkt) + DilatedConv(smi) → FNN(256,128,1)
- **max lengths**: seq=1000, pkt=63, smi=150

### PLANET

- **유형**: Complex-free (pocket graph + ligand 2D graph → pKd, multi-task)
- **환경**: conda planet (Python 3.6, PyTorch 1.8.1, RDKit 2020.09)
- **원본 레포**: `/data/hawon/InSiteDTA_baselines/PLANET/`
- **파이프라인 디렉토리**: 동일 (hw_ 파일 포함)
- **전처리 필수**: protein PDB + ligand SDF → ComplexPocket pickle (ligand centroid 기준 12Å pocket)
- **추가 인자**: `--ckpt_dir`, `--pkl_dir`
- **Protein features**: BLOSUM62 20D per residue + Cα 좌표
- **Ligand features**: 2D molecular graph (atom 31D + bond 6D, from SDF)
- **Pocket 정의**: ligand centroid 기준 12Å 이내 잔기 (`near_pocket` 함수)
- **Multi-task**: ligand intra-distance prediction + protein-ligand contact map + affinity (pKd)
- **Beta schedule**: β=0 (step ≤ 500), β=1 (step > 500) — affinity loss weight
- **pretrained**: `PLANET.param` (PyTorch state_dict, zip format)
- **checkpoint**: `hw_ckpts/{split}/planet_seed{N}.pt`
- **하이퍼파라미터**: Adam lr=1e-4, batch=16, 250 epochs, ExponentialLR(0.8) after step 60000, clip_norm=200.0
- **아키텍처**: ProteinEGNN(3 iter) + LigandGAT(10 iter) + ProLig(1 iter) → affinity
- **feature_dims**: 300, nheads=8, key/value_dims=300
- **Decoy augmentation**: 학습 시 decoy molecule 사용 가능하나, 우리는 decoy 없이 학습 (decoy_flag=False)
- **MSE residue**: MSE 처리 — MSE 잔기는 MET으로 매핑 (chemutils.py Residue class)

### GEMS

- **유형**: Complex-based GNN (protein-ligand interaction graph → pKd)
- **환경**: uv venv (`/data/hawon/InSiteDTA_baselines/GEMS/.venv/bin/python`)
- **원본 레포**: `/data/hawon/InSiteDTA_baselines/GEMS/`
- **파이프라인 디렉토리**: 동일 (hw_ 파일 포함)
- **전처리 필수**: flat symlinks → ankh → esm2 → chemberta → graph_construction
- **추가 인자**: `--ckpt_dir`, `--graph_dir`
- **variant**: B6AEPL (ChemBERTa-77M + Ankh-base + ESM2-T6)
- **아키텍처**: GEMS18d — NodeTransform MLP → 2× MetaLayer (EdgeModel + GATv2Conv + GlobalModel) → FC(384→64→1)
- **node feature dim**: 60 (base) + 768 (ankh) + 320 (esm) = 1148
- **global feature**: ChemBERTa-77M embedding (384-dim)
- **affinity scaling**: pK / 16 → [0,1], 예측 시 ×16 으로 복원
- **하이퍼파라미터**: SGD (momentum=0.9), lr=0.001, weight_decay=0.001, dropout=0.5, RMSE loss, batch=256, 2000 epochs, early stopping patience=100, ReduceLROnPlateau (factor=0.1, patience=10, min_lr=5e-5)
- **checkpoint**: `hw_ckpts/{split}/gems_seed{N}.pt`
- **pretrained ckpts**: `model/GEMS18d_B6AEPL_kikdic_d0500_0_f{0-4}_best_stdict.pt` (5-fold, CleanSplit 기준)
- **패치된 파일**:
  - `hw_patched_chemberta.py`: sanitize=False fallback, --first_ligand_only, truncation max_length=512
  - `hw_patched_graph_construction.py`: sanitize=False fallback, H atom pos 필터링, multi-pose SDF 처리
- **flat directory 구조**: `hw_flat/{scenario}/` — `{id}.pdb` + `{id}.sdf` (symlink) + embeddings + graphs
- **temp files**: `hw_gems_data_dict.json`, `hw_eval_split.json` → GEMS_ROOT에 저장 (src/data/ 오염 방지)
- **multi-pose SDF**: redocked 등 multi-pose scenario는 `--first_ligand_only` 필수

---

## 6. 알려진 주의사항

1. **pocket 출처 구분**: complex-based 모델의 pocket은 모델 학습 시 사용한 방법과 동일하게 생성해야 함. coreset의 `{pdb_id}_pocket.pdb`를 그대로 사용하면 성능 차이 발생.
2. **split 메타데이터**: `--split` 인자와 실제 사용하는 checkpoint/data가 일치하는지 확인. JSON의 `"split"` 필드는 `args.split`에서 오므로, 잘못된 인자 조합 시 메타데이터 오류 발생.
3. **Python set 비결정성**: `set()`의 순회 순서가 실행마다 달라짐. vocab을 set으로 빌드하면 동일 seed여도 결과가 달라질 수 있음 (DeepDTA 해당). `PYTHONHASHSEED` 고정 또는 `sorted()` 적용으로 해결 가능.
4. **eval_config.yaml의 python 경로**: conda 환경은 `conda run -n` 으로 실행하므로 yaml의 python 경로와 실제 실행 방법이 다를 수 있음. 향후 orchestrator 작성 시 고려 필요.
5. **coreset 드랍 샘플**:
   - `coreset_alphafold`: 4agn, 4agq, 5aba 드랍 (P2Rank 예측 실패 → ligand.sdf 없음)
   - `coreset_boltz2`: 3dx2 드랍 (SMILES RDKit 파싱 실패)

---

## 7. 새 베이스라인 추가 워크플로우

1. **원본 레포 분석**: 모델 유형 (sequence/complex-based/free), 입력 형식, 학습 코드 구조 파악
2. **환경 구축**: 독립 venv/conda 생성, `eval_config.yaml`에 python 경로 추가
3. **symlink 생성**: `src/exp_multi_scenario/baselines/{Model}/` → `/data/hawon/InSiteDTA_baselines/{Model}/`
4. **hw_01_input_prep.py 작성**: 학습용 (전체 PDBbind) + 평가용 (coreset) 전처리
   - complex-based: pocket 생성 방식을 원본과 동일하게 유지
   - sequence-based: CSV/feature 추출만 필요
5. **hw_02_train.py 작성**: split_config 기반 train/val 분할, seed 관리, ckpt 컨벤션
6. **hw_eval_wrapper.py 작성**: eval_interface.py import, 드랍 샘플 리포트 포함
7. **original split 평가** → 논문 재현값과 비교하여 파이프라인 검증
8. **cleansplit 학습 + 전 scenario 평가**

### 참조할 파일

새 베이스라인 작성 시 아래 파일들을 레퍼런스로 읽을 것:

- `src/exp_multi_scenario/baselines/eval_interface.py` — 공유 유틸 (필수)
- `/data/hawon/InSiteDTA_baselines/DeepDTA/hw_eval_wrapper.py` — sequence-based 레퍼런스
- `/data/hawon/InSiteDTA_baselines/CheapNet/hw_eval_wrapper.py` — complex-based 레퍼런스
- `/data/hawon/InSiteDTA_baselines/CheapNet/hw_01_input_prep.py` — 전처리 레퍼런스
- `/data/hawon/InSiteDTA_baselines/CheapNet/hw_02_train.py` — 학습 레퍼런스
- `src/exp_multi_scenario/results/` 하위 JSON — 출력 형태 확인

