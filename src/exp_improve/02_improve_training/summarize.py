"""
실험 결과 요약 스크립트.
ckpts/ 하위 디렉토리의 *_results.json 파일을 읽어 비교 테이블 출력.

Usage:
    python src/exp_improve/summarize.py                     # 전체 요약
    python src/exp_improve/summarize.py 00 01               # 특정 카테고리만
    python src/exp_improve/summarize.py --seed 312 309      # 특정 시드만
    python src/exp_improve/summarize.py 01 --seed 312       # 조합 가능
"""

import argparse
import glob
import json
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPTS_DIR = os.path.join(SCRIPT_DIR, "../", "ckpts")

STUDENT_METRICS_KEYS = [
    ("PCC",      "best_vl_PCC"),
    ("DCC_SR",   "best_vl_DCC_SR"),
    ("aff_loss", "best_vl_aff_loss"),
    ("epoch",    "epochs"),
]

TEACHER_METRICS_KEYS = [
    ("t_PCC",      "best_vl_PCC"),
    ("t_DCC_SR",   "best_vl_DCC_SR"),
    ("t_aff_loss", "best_vl_aff_loss"),
]


def load_results(ckpts_dir, filters=None, seed_filter=None):
    """ckpts 하위 디렉토리에서 results.json 수집 (최대 2단계 깊이)"""
    categories = {}

    pattern = os.path.join(ckpts_dir, "**", "*_results.json")
    for json_path in sorted(glob.glob(pattern, recursive=True)):
        cat_name = os.path.basename(os.path.dirname(json_path))

        # 카테고리 필터
        if filters and not any(f in cat_name for f in filters):
            continue

        with open(json_path, "r") as fp:
            data = json.load(fp)

        # student_metrics 우선, 없으면 하위 호환 metrics 키 사용
        student_metrics = data.get("student_metrics", data.get("metrics", {}))
        teacher_metrics = data.get("teacher_metrics", None)

        train_cfg = data.get("train_config", {})
        exp_name = os.path.basename(json_path).replace("_results.json", "")
        seed = train_cfg.get("seed", "?")

        # 시드 필터
        if seed_filter and str(seed) not in seed_filter:
            continue

        row = {
            "cat": cat_name,
            "exp": exp_name,
            "seed": seed,
        }
        for display_name, key in STUDENT_METRICS_KEYS:
            row[display_name] = student_metrics.get(key, None)

        for display_name, key in TEACHER_METRICS_KEYS:
            row[display_name] = teacher_metrics.get(key, None) if teacher_metrics else None

        categories.setdefault(cat_name, []).append(row)

    return categories


def format_val(val, key):
    if val is None:
        return "-"
    if key == "epoch":
        return str(val)
    return f"{val:.4f}"


def print_table(categories):
    if not categories:
        print("결과 파일이 없습니다.")
        return

    all_display_keys = [k for k, _ in STUDENT_METRICS_KEYS] + [k for k, _ in TEACHER_METRICS_KEYS]
    headers = ["Category", "Exp Name", "Seed"] + all_display_keys

    # 모든 행 데이터를 미리 생성하여 최대 폭 계산
    all_rows = []  # (vals, is_separator)
    for cat_name, rows in categories.items():
        for r in rows:
            vals = [
                str(r["cat"]),
                str(r["exp"]),
                str(r["seed"]),
            ] + [format_val(r[k], k) for k in all_display_keys]
            all_rows.append((vals, False))

        if len(rows) >= 2:
            summary = ["  mean ± std", "", ""]
            for display_name in all_display_keys:
                values = [r[display_name] for r in rows if r[display_name] is not None]
                if not values or display_name == "epoch":
                    summary.append("-")
                    continue
                arr = np.array(values, dtype=float)
                summary.append(f"{arr.mean():.4f} ± {arr.std():.4f}")
            all_rows.append((summary, False))

        all_rows.append((None, True))  # separator marker

    # 컬럼 폭: 헤더와 모든 데이터 중 최대값
    col_widths = [len(h) for h in headers]
    for vals, is_sep in all_rows:
        if is_sep:
            continue
        for i, v in enumerate(vals):
            col_widths[i] = max(col_widths[i], len(str(v)))

    def row_str(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    print(row_str(headers))
    print("-+-".join("-" * w for w in col_widths))

    for vals, is_sep in all_rows:
        if is_sep:
            print("-+-".join("-" * w for w in col_widths))
        else:
            print(row_str(vals))


def main():
    parser = argparse.ArgumentParser(description="실험 결과 요약")
    parser.add_argument("categories", nargs="*", default=None, help="카테고리 필터 (e.g., 00 01)")
    parser.add_argument("--seed", nargs="+", default=None, help="시드 필터 (e.g., --seed 312 309)")
    args = parser.parse_args()

    filters = args.categories or None
    seed_filter = set(args.seed) if args.seed else None
    categories = load_results(CKPTS_DIR, filters, seed_filter)
    print_table(categories)


if __name__ == "__main__":
    main()
