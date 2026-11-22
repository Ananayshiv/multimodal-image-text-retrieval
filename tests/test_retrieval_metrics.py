import json

import numpy as np
import pandas as pd
import pytest

from retrieval_metrics import (headline_from_query_scores, load_seed_runs, metrics_from_ranks,
                               paired_bootstrap_headline, per_query_frame, per_query_score,
                               ranks_from_similarity, results_markdown, seed_averaged_query_scores,
                               summarize_seeds)


def toy():
    image_ids, caption_ids = [10, 20], [1, 2, 3, 4]
    i2c = {10: [1, 2], 20: [3, 4]}
    c2i = {1: 10, 2: 10, 3: 20, 4: 20}
    sim = np.array([[0.9, 0.1, 0.8, 0.2],    # image 10: best true caption (1) ranks 1
                    [0.95, 0.3, 0.2, 0.4]])  # image 20: best true caption (4) ranks 2
    return sim, image_ids, caption_ids, i2c, c2i


def test_ranks_exact():
    i2t, t2i = ranks_from_similarity(*toy())
    assert i2t.tolist() == [1, 2]
    # caption 1: img10 .9 vs img20 .95 -> rank 2; caption 2: .1 vs .3 -> 2;
    # caption 3: img20 .2 vs img10 .8 -> 2; caption 4: .4 vs .2 -> 1
    assert t2i.tolist() == [2, 2, 2, 1]


def test_metrics_match_legacy_topk_recall():
    rng = np.random.default_rng(0)
    n_img, per = 30, 5
    image_ids = list(range(n_img))
    caption_ids = list(range(n_img * per))
    i2c = {i: list(range(i * per, (i + 1) * per)) for i in image_ids}
    c2i = {c: c // per for c in caption_ids}
    sim = rng.normal(size=(n_img, n_img * per))
    for i in image_ids:  # make true pairs somewhat more similar
        sim[i, i2c[i]] += 1.5
    i2t, t2i = ranks_from_similarity(sim, image_ids, caption_ids, i2c, c2i)
    m = metrics_from_ranks(i2t, t2i)
    # Legacy implementation from resumable_experiments.evaluate (top-k sets)
    top_i = np.argsort(-sim, axis=1)[:, :10]
    top_t = np.argsort(-sim.T, axis=1)[:, :10]
    for k in (1, 5, 10):
        legacy_i = np.mean([bool(set(top_i[r, :k]) & set(i2c[r])) for r in range(n_img)])
        legacy_t = np.mean([c2i[c] in set(top_t[c, :k]) for c in caption_ids])
        assert m[f"i2t_R@{k}"] == pytest.approx(legacy_i)
        assert m[f"t2i_R@{k}"] == pytest.approx(legacy_t)
    assert m["mean_score"] == pytest.approx((m["i2t_score"] + m["t2i_score"]) / 2)
    assert 0 < m["i2t_MRR"] <= 1


def test_shape_and_missing_ground_truth_errors():
    sim, image_ids, caption_ids, i2c, c2i = toy()
    with pytest.raises(ValueError):
        ranks_from_similarity(sim[:, :3], image_ids, caption_ids, i2c, c2i)
    with pytest.raises(ValueError):
        ranks_from_similarity(sim, image_ids, caption_ids, {10: [99], 20: [3]}, c2i)


def test_per_query_scores_reproduce_headline():
    i2t, t2i = ranks_from_similarity(*toy())
    frame = per_query_frame([10, 20], [1, 2, 3, 4], i2t, t2i)
    assert headline_from_query_scores(per_query_score(frame)) == pytest.approx(
        metrics_from_ranks(i2t, t2i)["mean_score"])


def _scores(i2t_ranks, t2i_ranks):
    return per_query_score(per_query_frame(range(len(i2t_ranks)), range(len(t2i_ranks)), i2t_ranks, t2i_ranks))


def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    base_i, base_t = rng.integers(1, 20, 200), rng.integers(1, 20, 400)
    a = _scores(base_i, base_t)
    better = _scores(np.maximum(1, base_i - 5), np.maximum(1, base_t - 5))
    res = paired_bootstrap_headline(a, better, iterations=500, seed=3)
    assert res["delta"] > 0 and res["ci_low"] > 0 and res["significant"]
    same = paired_bootstrap_headline(a, a, iterations=200)
    assert same["delta"] == 0 and not same["significant"]
    with pytest.raises(ValueError):
        paired_bootstrap_headline(a, _scores(base_i[:10], base_t[:10]), iterations=10)


def test_seed_aggregation(tmp_path):
    for seed, (lora, full) in {42: (0.93, 0.92), 43: (0.925, 0.921), 44: (0.931, 0.918)}.items():
        for name, score, params in (("openclip_lora", lora, 1_228_801), ("openclip_full_finetune", full, 149_620_737)):
            d = tmp_path / f"seed_{seed}" / name
            d.mkdir(parents=True)
            rec = {"experiment": name, "status": "completed", "trainable_parameters": params,
                   "test_mean_score": score, "test_i2t_score": score + .03, "test_t2i_score": score - .03,
                   "test_i2t_R@1": .88, "test_t2i_R@1": .75}
            (d / "completed.json").write_text(json.dumps(rec))
            ranks = np.array([1, 2, 3]) if name == "openclip_lora" else np.array([2, 3, 11])
            per_query_frame([1, 2, 3], [1, 2, 3], ranks, ranks).to_csv(d / "test_per_query.csv", index=False)
    runs = load_seed_runs(tmp_path)
    table = summarize_seeds(runs)
    assert table.index[0] == "openclip_lora"
    assert table.loc["openclip_lora", "n_seeds"] == 3
    assert table.loc["openclip_lora", "test_mean_score_mean"] == pytest.approx(np.mean([.93, .925, .931]))
    a = seed_averaged_query_scores(runs, "openclip_full_finetune")
    b = seed_averaged_query_scores(runs, "openclip_lora")
    comp = {"a": "openclip_full_finetune", "b": "openclip_lora", **paired_bootstrap_headline(a, b, iterations=200)}
    md = results_markdown(table, [comp])
    assert "openclip_lora" in md and "± " in md and "openclip_lora − openclip_full_finetune" in md
    with pytest.raises(FileNotFoundError):
        load_seed_runs(tmp_path / "empty")
