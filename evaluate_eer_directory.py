#!/usr/bin/env python3
"""Calculate EER from ID,Score,Label CSVs at FP and transformer block stages."""
import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

STAGE_RE = re.compile(r"^score_(fp|b\d+)\.csv$", re.IGNORECASE)


def stage_key(stage):
    return -1 if stage.lower() == "fp" else int(stage[1:])


def compute_eer(bonafide, spoof):
    """ASVspoof-style closest FRR/FAR EER. Higher scores mean bonafide."""
    bonafide = np.asarray(bonafide, dtype=np.float64)
    spoof = np.asarray(spoof, dtype=np.float64)
    if not len(bonafide) or not len(spoof):
        raise ValueError("EER requires both bonafide (1) and spoof (0) samples")
    scores = np.concatenate((bonafide, spoof))
    labels = np.concatenate((np.ones(len(bonafide)), np.zeros(len(spoof))))
    indices = np.argsort(scores, kind="mergesort")
    sorted_labels = labels[indices]
    tar_cum = np.cumsum(sorted_labels)
    non_cum = len(spoof) - (np.arange(1, len(scores) + 1) - tar_cum)
    frr = np.r_[0.0, tar_cum / len(bonafide)]
    far = np.r_[1.0, non_cum / len(spoof)]
    thresholds = np.r_[scores[indices[0]] - 0.001, scores[indices]]
    idx = int(np.argmin(np.abs(frr - far)))
    return 100 * (frr[idx] + far[idx]) / 2, float(thresholds[idx])


def score_difference(value):
    try:
        pair = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return np.nan
        return float(pair[1]) - float(pair[0])
    except (ValueError, SyntaxError, TypeError, IndexError):
        return np.nan


def read_scores(path, real_label):
    df = pd.read_csv(path, usecols=["Score", "Label"])
    if df.empty:
        raise ValueError("empty CSV")
    scores = df["Score"].map(score_difference).to_numpy(dtype=float)
    labels = pd.to_numeric(df["Label"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    if not valid.all():
        print(f"Warning: {path}: ignoring {len(valid) - int(valid.sum())} invalid rows")
    scores = scores[valid]
    labels = labels[valid]
    bonafide = scores[labels == real_label]
    spoof = scores[labels != real_label]
    return bonafide, spoof


def evaluate_directory(directory, real_label, max_block=None):
    results = []
    raw_scores = {}
    for file in sorted(directory.glob('score_*.csv'), key=lambda p: (stage_key(STAGE_RE.match(p.name).group(1)) if STAGE_RE.match(p.name) else 10**9)):
        match = STAGE_RE.match(file.name)
        if not match:
            continue
        stage = match.group(1).upper()
        if max_block is not None and stage != 'FP' and stage_key(stage) > max_block:
            continue
        try:
            bonafide, spoof = read_scores(file, real_label)
            eer, threshold = compute_eer(bonafide, spoof)
        except (ValueError, KeyError, pd.errors.ParserError) as err:
            print(f"Warning: skipping {file}: {err}")
            continue
        raw_scores[stage] = (bonafide, spoof)
        results.append({"Layer": stage, "EER (%)": eer, "Threshold": threshold,
                        "Bonafide": len(bonafide), "Spoof": len(spoof)})
    return pd.DataFrame(results), raw_scores


def summarize_layers(frames, raw_by_dataset):
    """Macro average of dataset EERs and micro pooled EER using all trials."""
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    rows = []
    for stage in sorted(merged["Layer"].unique(), key=stage_key):
        group = merged[merged["Layer"] == stage]
        bona = [scores[stage][0] for scores in raw_by_dataset.values() if stage in scores]
        spoof = [scores[stage][1] for scores in raw_by_dataset.values() if stage in scores]
        pooled_eer, pooled_threshold = compute_eer(np.concatenate(bona), np.concatenate(spoof))
        rows.append({
            "Layer": stage,
            "Datasets": len(group),
            "Average EER (%)": group["EER (%)"].mean(),
            "Pooled EER (%)": pooled_eer,
            "Pooled Threshold": pooled_threshold,
            "Bonafide": sum(map(len, bona)),
            "Spoof": sum(map(len, spoof)),
        })
    return pd.DataFrame(rows)


# Preferred paper-style ordering; unknown datasets are appended alphabetically.
DATASET_ORDER = [
    "19la", "21la", "21df", "librisevoc", "itw", "for", "sc_sdd",
    "dfadd", "diffssd", "mlaad_en", "decro_en", "25df",
]
DISPLAY_NAMES = {
    "19la": "19LA", "21la": "21LA", "21df": "21DF",
    "librisevoc": "LSV", "itw": "ITW", "for": "FoR",
    "sc_sdd": "SC", "dfadd": "DFADD", "diffssd": "DIFSSD",
    "mlaad_en": "MLAAD-EN", "decro_en": "DECRO-EN", "25df": "25DF",
}


def make_wide_table(frames, summary):
    """Dataset EERs + macro and pooled EER in one row per layer."""
    all_results = pd.concat(frames, ignore_index=True)
    wide = all_results.pivot(index="Layer", columns="Dataset", values="EER (%)")
    dataset_names = sorted(
        wide.columns, key=lambda name: (
            DATASET_ORDER.index(name.lower()) if name.lower() in DATASET_ORDER
            else len(DATASET_ORDER), name.lower()
        )
    )
    wide = wide.reindex(columns=dataset_names)
    wide.columns = [DISPLAY_NAMES.get(name.lower(), name.upper()) for name in dataset_names]
    wide = wide.join(summary.set_index("Layer")[["Average EER (%)", "Pooled EER (%)"]])
    wide = wide.rename(columns={"Average EER (%)": "Avg.", "Pooled EER (%)": "Pool."})
    wide = wide.reindex(sorted(wide.index, key=stage_key))
    wide.index = ["FP" if stage == "FP" else stage[1:] for stage in wide.index]
    wide.index.name = "Layer"
    return wide.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--score_dir', required=True, type=Path,
                        help='Dataset folder or model folder containing dataset subfolders')
    parser.add_argument('--dataset', help='Evaluate only a matching dataset folder name')
    parser.add_argument('--real-label', choices=[0, 1], type=int, default=1,
                        help='Label identifying bonafide; default: 1')
    parser.add_argument('--max-block', type=int, default=None)
    parser.add_argument('--save', type=Path, help='Save per-dataset long-format results to CSV')
    parser.add_argument('--summary-save', type=Path, help='Save macro/pooled EER by layer to CSV')
    parser.add_argument('--table-save', type=Path, help='Save paper-style wide EER table to CSV')
    parser.add_argument('--latex-save', type=Path, help='Save paper-style wide table to LaTeX')
    parser.add_argument('--print-individual', action='store_true',
                        help='Also print individual per-dataset tables')
    parser.add_argument('--decimals', type=int, default=2,
                        help='Printed decimal places (default: 2)')
    args = parser.parse_args()

    root = args.score_dir
    if not root.is_dir():
        parser.error(f'Directory does not exist: {root}')
    dirs = ([root] if any(STAGE_RE.match(p.name) for p in root.glob('*.csv'))
            else [p for p in root.rglob('*') if p.is_dir() and any(STAGE_RE.match(f.name) for f in p.glob('*.csv'))])
    if args.dataset:
        dirs = [p for p in dirs if p.name.lower() == args.dataset.lower()]
    if not dirs:
        parser.error('No matching score_fp.csv / score_bN.csv files found')

    frames = []
    raw_by_dataset = {}
    for directory in sorted(dirs):
        result, raw = evaluate_directory(directory, args.real_label, args.max_block)
        if result.empty:
            continue
        if args.print_individual:
            print('\n' + '=' * 80)
            print(f'DATASET: {directory.name.upper()} ({directory})')
            print('=' * 80)
            print(result.to_string(index=False, formatters={
                "EER (%)": f"{{:.{args.decimals}f}}".format,
                "Threshold": "{:.4f}".format,
            }))
        frames.append(result.assign(Dataset=directory.name))
        raw_by_dataset[str(directory)] = raw

    if not frames:
        parser.error('No valid scores with both bonafide and spoof samples were found')
    summary = summarize_layers(frames, raw_by_dataset)
    wide = make_wide_table(frames, summary)

    print('\n' + '=' * 110)
    print(f'ALL DATASETS: EER (%, lower is better) | {root}')
    print('=' * 110)
    formatters = {column: f"{{:.{args.decimals}f}}".format
                  for column in wide.columns if column != 'Layer'}
    print(wide.to_string(index=False, na_rep='--', formatters=formatters))
    print('\nAvg. = unweighted mean of available dataset EERs at each layer.')
    print('Pool. = EER recomputed from concatenated bonafide and spoof scores at each layer.')
    print('Missing dataset/layer combinations appear as -- and are excluded from the average.')

    def save_csv(df, path):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f'Saved: {path}')

    save_csv(wide, args.table_save)
    save_csv(summary, args.summary_save)
    if args.save:
        merged = pd.concat(frames, ignore_index=True)
        save_csv(merged[['Dataset', 'Layer', 'EER (%)', 'Threshold', 'Bonafide', 'Spoof']], args.save)
    if args.latex_save:
        args.latex_save.parent.mkdir(parents=True, exist_ok=True)
        # The table is intentionally flat so users can add custom multicolumn group headings.
        latex = wide.to_latex(index=False, float_format=f"%.{args.decimals}f",
                              na_rep='--', escape=True)
        args.latex_save.write_text(latex, encoding='utf-8')
        print(f'Saved: {args.latex_save}')


if __name__ == '__main__':
    main()
