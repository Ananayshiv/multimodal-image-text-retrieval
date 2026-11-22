"""Torch-free retrieval metrics, multi-seed aggregation and paired significance tests.

Everything here works on NumPy arrays / pandas frames so it can be unit-tested on a
CPU-only machine and reused by both the training suite and the aggregation script.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

KS = (1, 5, 10)


def ranks_from_similarity(sim: np.ndarray, image_ids: Sequence[int], caption_ids: Sequence[int],
                          image_to_captions: Mapping[int, Iterable[int]],
                          caption_to_image: Mapping[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Exact 1-based ranks of the first correct match for every query.

    sim[i, j] = similarity(image_ids[i], caption_ids[j]).
    Image->text: rank of the BEST-ranked of the image's ground-truth captions.
    Text->image: rank of the caption's single ground-truth image.
    Ties are resolved optimistically (rank = 1 + #strictly-better candidates),
    which matches top-k recall up to tie order.
    """
    sim = np.asarray(sim, dtype=np.float64)
    if sim.shape != (len(image_ids), len(caption_ids)):
        raise ValueError(f"similarity shape {sim.shape} != ({len(image_ids)}, {len(caption_ids)})")
    col = {int(c): j for j, c in enumerate(caption_ids)}
    row = {int(i): k for k, i in enumerate(image_ids)}
    i2t = np.empty(len(image_ids), dtype=np.int64)
    for k, iid in enumerate(image_ids):
        cols = [col[int(c)] for c in image_to_captions[int(iid)] if int(c) in col]
        if not cols:
            raise ValueError(f"image {iid} has no ground-truth caption in the pool")
        best = sim[k, cols].max()
        i2t[k] = 1 + int((sim[k] > best).sum())
    t2i = np.empty(len(caption_ids), dtype=np.int64)
    for j, cid in enumerate(caption_ids):
        true_row = row[int(caption_to_image[int(cid)])]
        t2i[j] = 1 + int((sim[:, j] > sim[true_row, j]).sum())
    return i2t, t2i


def metrics_from_ranks(i2t_ranks: np.ndarray, t2i_ranks: np.ndarray) -> Dict[str, float]:
    """Recall@{1,5,10} per direction, the project's headline score, and MRR."""
    i2t_ranks, t2i_ranks = np.asarray(i2t_ranks), np.asarray(t2i_ranks)
    m: Dict[str, float] = {}
    for k in KS:
        m[f"i2t_R@{k}"] = float(np.mean(i2t_ranks <= k))
    for k in KS:
        m[f"t2i_R@{k}"] = float(np.mean(t2i_ranks <= k))
    m["i2t_score"] = float(np.mean([m[f"i2t_R@{k}"] for k in KS]))
    m["t2i_score"] = float(np.mean([m[f"t2i_R@{k}"] for k in KS]))
    m["mean_score"] = (m["i2t_score"] + m["t2i_score"]) / 2
    m["i2t_MRR"] = float(np.mean(1.0 / i2t_ranks))
    m["t2i_MRR"] = float(np.mean(1.0 / t2i_ranks))
    m["i2t_median_rank"] = float(np.median(i2t_ranks))
    m["t2i_median_rank"] = float(np.median(t2i_ranks))
    return m


def per_query_frame(image_ids, caption_ids, i2t_ranks, t2i_ranks) -> pd.DataFrame:
    a = pd.DataFrame({"direction": "i2t", "query_id": np.asarray(image_ids, dtype=np.int64), "rank": i2t_ranks})
    b = pd.DataFrame({"direction": "t2i", "query_id": np.asarray(caption_ids, dtype=np.int64), "rank": t2i_ranks})
    return pd.concat([a, b], ignore_index=True)


def per_query_score(frame: pd.DataFrame) -> pd.Series:
    """Per-query contribution to the headline score: mean of R@1/5/10 hits.

    Averaging these over queries within a direction reproduces that direction's
    score, so paired tests on this quantity test the headline metric itself.
    """
    hits = sum((frame["rank"] <= k).astype(float) for k in KS) / len(KS)
    return pd.Series(hits.values, index=pd.MultiIndex.from_frame(frame[["direction", "query_id"]]))


def headline_from_query_scores(scores: pd.Series) -> float:
    return float(scores.groupby(level="direction").mean().mean())


def paired_bootstrap_headline(a: pd.Series, b: pd.Series, iterations: int = 5000,
                              seed: int = 0) -> Dict[str, float]:
    """95% CI for headline(b) - headline(a), resampling queries within each direction.

    `a` and `b` are per-query scores (see per_query_score) on identical queries.
    """
    if not a.index.equals(b.index):
        b = b.reindex(a.index)
        if b.isna().any():
            raise ValueError("Paired comparison needs identical query sets")
    rng = np.random.default_rng(seed)
    deltas = []
    by_dir = {d: (a.xs(d, level="direction").to_numpy(), b.xs(d, level="direction").to_numpy())
              for d in a.index.get_level_values("direction").unique()}
    for _ in range(iterations):
        per_dir = []
        for xa, xb in by_dir.values():
            idx = rng.integers(0, len(xa), len(xa))
            per_dir.append(xb[idx].mean() - xa[idx].mean())
        deltas.append(float(np.mean(per_dir)))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    delta = headline_from_query_scores(b) - headline_from_query_scores(a)
    return {"delta": float(delta), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def load_seed_runs(root: str | Path) -> pd.DataFrame:
    """Collect completed.json from <root>/seed_<S>/<experiment>/ into one frame."""
    rows = []
    for done in sorted(Path(root).glob("seed_*/*/completed.json")):
        record = json.loads(done.read_text())
        record["seed"] = int(done.parent.parent.name.split("_", 1)[1])
        record["run_dir"] = str(done.parent)
        rows.append(record)
    if not rows:
        raise FileNotFoundError(f"No completed runs under {root}/seed_*/")
    return pd.DataFrame(rows)


def summarize_seeds(runs: pd.DataFrame, metric_prefix: str = "test_") -> pd.DataFrame:
    """Mean, std and n over seeds for every reported metric, per experiment."""
    cols = [c for c in runs.columns if c.startswith(metric_prefix)] + ["trainable_parameters"]
    grouped = runs.groupby("experiment")[cols]
    table = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    table["n_seeds"] = runs.groupby("experiment").seed.nunique()
    return table.sort_values(f"{metric_prefix}mean_score_mean", ascending=False)


def seed_averaged_query_scores(runs: pd.DataFrame, experiment: str) -> pd.Series:
    """Per-query headline contribution averaged over the experiment's seeds."""
    series = []
    for run_dir in runs.loc[runs.experiment == experiment, "run_dir"]:
        frame = pd.read_csv(Path(run_dir) / "test_per_query.csv")
        series.append(per_query_score(frame))
    if not series:
        raise ValueError(f"No runs for {experiment}")
    out = pd.concat(series, axis=1).mean(axis=1)
    return out.sort_index()


def results_markdown(table: pd.DataFrame, comparisons: List[Dict], prefix: str = "test_") -> str:
    lines = ["| Experiment | Seeds | I→T score | T→I score | **Mean score** | I→T R@1 | T→I R@1 | Trainable params |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda e, m: f"{table.loc[e, f'{prefix}{m}_mean']:.4f} ± {0 if np.isnan(table.loc[e, f'{prefix}{m}_std']) else table.loc[e, f'{prefix}{m}_std']:.4f}"
    for e in table.index:
        lines.append(f"| {e} | {int(table.loc[e, 'n_seeds'])} | {fmt(e, 'i2t_score')} | {fmt(e, 't2i_score')} | "
                     f"**{fmt(e, 'mean_score')}** | {fmt(e, 'i2t_R@1')} | {fmt(e, 't2i_R@1')} | "
                     f"{int(table.loc[e, 'trainable_parameters_mean']):,} |")
    if comparisons:
        lines += ["", "| Comparison (B − A, mean score) | Δ | 95% CI | Significant |", "|---|---:|---:|:---:|"]
        for c in comparisons:
            lines.append(f"| {c['b']} − {c['a']} | {c['delta']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                         f"{'yes' if c['significant'] else 'no'} |")
    lines.append("")
    lines.append("± is the sample standard deviation over seeds. CIs: paired bootstrap over test queries "
                 "(per-query scores averaged over seeds), resampled within each retrieval direction.")
    return "\n".join(lines)
