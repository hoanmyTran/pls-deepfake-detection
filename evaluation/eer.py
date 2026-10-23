import argparse
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import compute_eer


def parse_file(filename, subset_list=None):
    df = pd.read_csv(filename, dtype={"ID": str})
    if not {"ID", "Score", "Label"}.issubset(df.columns):
        raise ValueError(f"Expected ID,Score,Label columns: {filename}")
    if df.empty:
        raise ValueError(f"Empty scores: {filename}")
    if df["ID"].isna().any() or df["ID"].duplicated().any():
        raise ValueError(f"Missing/duplicate IDs in {filename}; basename collisions can corrupt results")
    logits = np.asarray([ast.literal_eval(str(score)) for score in df.Score], dtype=float)
    labels = pd.to_numeric(df.Label, errors="raise").to_numpy(dtype=int)
    if logits.shape != (len(df), 2) or not np.isfinite(logits).all():
        raise ValueError("Each Score must contain two finite logits")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (fake) or 1 (real)")
    prefixes = df.ID.map(lambda x: str(x).split("-", 1)[0] if "-" in str(x) else None).to_numpy()
    result = {"Pooled": {"y_true": labels.tolist(), "logits": logits.tolist()}}
    if subset_list:
        for prefix in subset_list:
            mask = prefixes == prefix
            result[prefix] = {"y_true": labels[mask].tolist(), "logits": logits[mask].tolist()}
    return result


def compute_metrics(y_true, logits):
    y = np.asarray(y_true, dtype=int)
    z = np.asarray(logits, dtype=float)
    if not np.isin(y, [0, 1]).all() or z.shape != (len(y), 2):
        raise ValueError("Bad labels/logits shape")
    cm_scores = z[:, 1] - z[:, 0]
    eer, threshold = compute_eer(cm_scores[y == 1], cm_scores[y == 0])
    return {"eer": round(eer * 100, 4), "eer_threshold": round(threshold, 4)}


def main(filename, subset_list=None):
    data = parse_file(filename, subset_list)
    rows = []
    for subset, group in data.items():
        if len(set(group["y_true"])) < 2:
            continue
        rows.append({"subset": subset, **compute_metrics(group["y_true"], group["logits"])})
    if not rows:
        raise ValueError("No subset has both classes")
    output = pd.DataFrame(rows).set_index("subset")
    print("\n===== EER METRICS SUMMARY =====")
    print(output.to_string())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("score_file")
    parser.add_argument("prefixes", nargs="*")
    args = parser.parse_args()
    main(args.score_file, args.prefixes or None)
