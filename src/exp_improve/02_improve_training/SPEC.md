# 02_improve_training: Self-Distillation on Improved Architecture

## 1. 목표

EMA self-distillation을 통해 PCC 개선.  
**전제**: 01_improve_architecture에서 확정된 best 아키텍처 위에 적용.  
각 실험 폴더의 `model/`은 01_improve_architecture 결과물로 채워진 후 학습 진행.

---

## 2. 현황 (1-seed, s312, 00_original 아키텍처 기준 예비 스크리닝)

| 실험 | PCC | 비고 |
|------|-----|------|
| 00_original (3-seed 평균) | 0.718 ± 0.004 | 기준선 |
| 0a aug-strong | 0.7182 | |
| 0b aug-struct-robust | 0.6794 | jitter+blur 과도 |
| 1a distill-pocket-soft | 0.7149 | |
| 1b distill-feat-match | 0.7112 | |

→ 모델 내부 불안정 요소(ligand encoder 기여도, pocket attention 구조 등) 미해결 상태에서의 수치로, 훈련 전략의 효과를 정확히 평가하기 어렵다고 판단. 01_improve_architecture로 구조를 먼저 정비한 후 재실험.

---

## 3. 설계

### 3.1 Teacher-Student 구조

- **Self-distillation**: 동일 아키텍처 (01_improve_architecture best)
- **Teacher 업데이트**: EMA (`θ_t = m * θ_t + (1-m) * θ_s`)
- **EMA momentum schedule**: cosine schedule, 0.996 → 1.0
- **Teacher는 gradient 없음** (eval mode, no_grad)
- **Scheduler**: T_0=100 (고정)

### 3.2 Student Augmentation

| Augmentation | Teacher | Student |
|---|---|---|
| Rotation | 동일 (p=0.3) | 동일 (p=0.3) |
| Channel dropout | X | O |
| Voxel Gaussian noise | X | O |
| Label noise | X | O (std=0.15) |

### 3.3 Distillation Targets

**(A) Pocket soft logits**
- `BCE_with_logits(student_logit, sigmoid(teacher_logit / τ))`, τ=1

**(B) Bottleneck_p2l feature matching**
- Cosine similarity loss (`1 - cos_sim`)

**(E) L2Pocket feature matching**
- `l2pocket` (ligand이 pocket을 바라보는 representation) teacher→student matching
- Cosine similarity loss

**(C) Pooled features (rotation-invariant, Phase 2+)**
- Teacher/Student 서로 다른 rotation, pooled_features matching

### 3.4 Loss 구성

```
L_total = L_original + α * L_distill
L_original = w_poc * L_poc(student, GT) + w_aff * L_aff(student, GT)
```

---

## 4. 실험 계획

1-seed screening (seed=312), PCC 기준 의사결정.

### Phase 0: Augmentation baseline (distillation 없이)

| ID | 폴더 | 실험 | 설명 |
|----|------|------|------|
| 0a | 0a_aug_strong | aug-strong | channel dropout + voxel noise |
| 0b | 0b_aug_struct_robust | aug-struct-robust | 0a + jitter + blur (이전 스크리닝에서 실패, 재검토 예정) |

### Phase 1: 단일 distillation target 비교

| ID | 폴더 | Distill target | T_0 |
|----|------|---------------|-----|
| 1a | 1a_distill_pocket_soft | Pocket logits (A) | 100 |
| 1b | 1b_distill_feat_match | Bottleneck_p2l (B) | 100 |
| 1e | (예정) 1e_distill_l2pocket | L2Pocket features (E) | 100 |

### Phase 2: 조합 및 확장

| ID | 실험 | 변경사항 |
|----|------|---------|
| 2a | best(Phase 1) + ligand perturbation | coord jitter + node noise |
| 2b | A+E 조합 | pocket-soft + l2pocket 동시 |
| 2c | rotation-invariant distillation | 서로 다른 rotation + pooled_features (C) |

### Phase 3: 최종 검증

| ID | 실험 | 변경사항 |
|----|------|---------|
| 3a | best variant 3-seed | seed=312, 309, 429 |

---

## 5. 파일 구조

```
02_improve_training/
├── SPEC.md
├── 0a_aug_strong/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── run_eval_multiscenario.sh
│   └── model/                 ← 비어있음: 01_improve_architecture 결과물 대기
├── 0b_aug_struct_robust/      ← 동일 구조
├── 1a_distill_pocket_soft/    ← 동일 구조
├── 1b_distill_feat_match/     ← 동일 구조
└── (1e_distill_l2pocket/)     ← 예정
```

---

## 6. 결과 저장

- Checkpoint: `src/exp_improve/ckpts/02_improve_training/{exp_folder}/{exp_name}.pt`
- Results: 같은 디렉토리에 `{exp_name}_results.json`
- wandb 컨벤션: `distill-pocket-soft_cs0_s312`, `distill-feat-match_cs0_s312` 등
