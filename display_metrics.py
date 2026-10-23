import argparse
from pathlib import Path
import re
import pandas as pd
from evaluation.eer import parse_file, compute_metrics


def layer_key(path):
    match = re.fullmatch(r"score_(fp|b\d+)\.csv", path.name)
    if not match:
        return None
    stage = match.group(1)
    return -1 if stage == "fp" else int(stage[1:])


def display_ordered_metrics(base_dir, target_dataset=None):
    root = Path(base_dir)
    directories = [root] if any(layer_key(f) is not None for f in root.glob("*.csv")) else sorted(d for d in root.iterdir() if d.is_dir())
    if target_dataset:
        directories = [d for d in directories if d.name.lower() == target_dataset.lower()]
    for directory in directories:
        files = sorted((f for f in directory.glob("*.csv") if layer_key(f) is not None), key=layer_key)
        rows = []
        for score_file in files:
            data = parse_file(score_file)["Pooled"]
            if len(set(data["y_true"])) < 2:
                continue
            rows.append({"Layer": "FP" if layer_key(score_file) == -1 else f"B{layer_key(score_file)}", **compute_metrics(data["y_true"], data["logits"])})
        if rows:
            print(f"\n{'='*80}\n DATASET: {directory.name.upper()} ({directory})\n{'='*80}")
            print(pd.DataFrame(rows).set_index("Layer").rename(columns={"eer": "EER (%)", "eer_threshold": "Threshold"}).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", "-d", required=True)
    parser.add_argument("--dataset", "-ds")
    args = parser.parse_args()
    display_ordered_metrics(args.score_dir, args.dataset)
