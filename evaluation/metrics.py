
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def compute_det_curve(target_scores, nontarget_scores):
    """Calculate FRR, FAR and thresholds; larger score means target/bonafide."""
    target = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    nontarget = np.asarray(nontarget_scores, dtype=np.float64).reshape(-1)
    if not target.size or not nontarget.size:
        raise ValueError("Both target and nontarget scores must be nonempty")
    if not np.isfinite(target).all() or not np.isfinite(nontarget).all():
        raise ValueError("Scores contain NaN or infinity")

    scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones(target.size), np.zeros(nontarget.size)])
    indices = np.argsort(scores, kind="mergesort")
    sorted_labels = labels[indices]
    target_cumsum = np.cumsum(sorted_labels)
    nontarget_above = nontarget.size - (np.arange(1, scores.size + 1) - target_cumsum)
    frr = np.concatenate(([0.0], target_cumsum / target.size))
    far = np.concatenate(([1.0], nontarget_above / nontarget.size))
    thresholds = np.concatenate(([np.nextafter(scores[indices[0]], -np.inf)], scores[indices]))
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Return the minimum-gap (discrete) EER approximation and its threshold."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    index = int(np.argmin(np.abs(frr - far)))
    return float((frr[index] + far[index]) / 2), float(thresholds[index])


def compute_eer_API(score_file: PathLike, protocol_file: PathLike):
    """Join whitespace-delimited score rows with labeled protocol CSV.

    Score format: header optionally present, followed by ``path score`` rows.
    Protocol format: header ``file_name,label`` (extra columns are allowed).
    """
    protocol = pd.read_csv(protocol_file, usecols=["file_name", "label"], dtype=str)
    if protocol["file_name"].duplicated().any():
        raise ValueError("Duplicate file_name values in protocol")

    scores = pd.read_csv(
        score_file,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["file_name", "cm_score"],
        dtype={"file_name": str},
    )
    if not scores.empty and scores.iloc[0]["file_name"] == "file_name":
        scores = scores.iloc[1:].copy()
    scores["cm_score"] = pd.to_numeric(scores["cm_score"], errors="raise")
    if scores["file_name"].duplicated().any():
        raise ValueError("Duplicate file_name values in scores")

    merged = scores.merge(protocol, on="file_name", how="left", validate="one_to_one")
    if merged["label"].isna().any():
        missing = merged.loc[merged["label"].isna(), "file_name"].head(3).tolist()
        raise ValueError(f"Unlabeled scored files: {missing}")
    unknown = set(merged["label"]) - {"bonafide", "spoof"}
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")

    out_path = Path(score_file).parent / "scores_merged.txt"
    merged.to_csv(out_path, index=False)
    bonafide = merged.loc[merged.label == "bonafide", "cm_score"].to_numpy()
    spoof = merged.loc[merged.label == "spoof", "cm_score"].to_numpy()
    return compute_eer(bonafide, spoof)
