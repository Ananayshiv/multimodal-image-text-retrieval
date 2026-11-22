#!/usr/bin/env python3
"""Multi-seed, held-out benchmark of the adaptation strategies.

    # GPU (Colab T4/L4/A100). Resumable: re-run the same command after a disconnect.
    python run_seeds.py --seeds 42 43 44 --experiments openclip_lora openclip_full_finetune openclip_gradual_unfreeze
    python run_seeds.py --seeds 42 --experiments clip_zero_shot clip_projection_tuning
    python run_seeds.py --aggregate-only

Each seed writes artifacts/seed_runs/seed_<S>/<experiment>/{completed.json, test_per_query.csv, history.csv}.
Epochs are selected on a dev split carved from train (same split for every seed); the
reported numbers are from data/val.csv, which is never used for any decision.
Aggregation writes artifacts/seed_runs/RESULTS.md and summary.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from retrieval_metrics import (load_seed_runs, paired_bootstrap_headline, results_markdown,
                               seed_averaged_query_scores, summarize_seeds)

ROOT = Path("artifacts/seed_runs")


def train(seeds, experiments, data_dir, epochs, batch_size, selection_images, patience):
    from resumable_experiments import EXPERIMENTS, RetrievalSuite, SuiteConfig  # needs torch/open_clip
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise SystemExit(f"Unknown experiments: {sorted(unknown)}; choose from {sorted(EXPERIMENTS)}")
    for seed in seeds:
        cfg = SuiteConfig(data_dir=data_dir, run_dir=str(ROOT / f"seed_{seed}"), seed=seed,
                          epochs=epochs, batch_size=batch_size, selection_images=selection_images,
                          patience=patience)
        suite = RetrievalSuite(cfg)
        print(f"=== seed {seed}: {len(suite.train)} train pairs, {len(suite.dev)} dev pairs, "
              f"{len(suite.val)} test pairs; selection on {suite.selection_split}")
        suite.run_all({name: EXPERIMENTS[name] for name in experiments})


def aggregate(baseline: str, iterations: int):
    runs = load_seed_runs(ROOT)
    table = summarize_seeds(runs)
    names = list(table.index)
    comparisons = []
    # Every fine-tuned strategy vs the baseline, plus the top-2 head-to-head (the close call).
    pairs = [(baseline, n) for n in names if n != baseline and baseline in names]
    if len(names) >= 2 and (names[1], names[0]) not in pairs and (names[0], names[1]) not in pairs:
        pairs.append((names[1], names[0]))
    for a, b in pairs:
        comp = paired_bootstrap_headline(seed_averaged_query_scores(runs, a),
                                         seed_averaged_query_scores(runs, b), iterations=iterations)
        comparisons.append({"a": a, "b": b, **comp})
    ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(ROOT / "summary.csv")
    (ROOT / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    md = results_markdown(table, comparisons)
    header = ("# Multi-seed held-out results\n\nTest split: data/val.csv (1,000 images, 5,000 captions). "
              "Epoch selection: dev split carved from train. Seeds per experiment shown in the table.\n\n")
    (ROOT / "RESULTS.md").write_text(header + md + "\n")
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--experiments", nargs="+",
                    default=["openclip_lora", "openclip_full_finetune", "openclip_gradual_unfreeze"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=10, help="max epochs (v1 used 10)")
    ap.add_argument("--patience", type=int, default=3, help="early stop on DEV; 0 disables")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--selection-images", type=int, default=500)
    ap.add_argument("--baseline", default="openclip_full_finetune",
                    help="comparisons are reported as <experiment> minus this baseline")
    ap.add_argument("--bootstrap-iterations", type=int, default=5000)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        train(args.seeds, args.experiments, args.data_dir, args.epochs, args.batch_size, args.selection_images,
              args.patience)
    aggregate(args.baseline, args.bootstrap_iterations)


if __name__ == "__main__":
    main()
