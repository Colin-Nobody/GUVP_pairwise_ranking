#!/usr/bin/env python3
"""Nested image-disjoint training-edge labels and held-out-edge ranking evaluation."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.trueskill_rank import calculate_trueskill

AUDIT = ROOT / "output/vit_vp_1000_v1"
SEEDS = (42, 137, 2026)
RIDGE_ALPHAS = np.logspace(-3, 6, 10)
EN_ALPHAS = np.logspace(-3, 0.5, 12)
EN_L1 = (0.1, 0.4, 0.7, 0.9, 1.0)


def city_roles(frame: pd.DataFrame, repeat: int, outer_fold: int, inner_fold: int | None = None):
    mask = (frame.repeat == repeat) & (frame.outer_fold == outer_fold)
    if inner_fold is not None:
        mask &= frame.inner_fold == inner_fold
    part = frame.loc[mask]
    valid_name = "test" if inner_fold is None else "validation"
    return (set(part.loc[part.partition == "train", "city"].astype(str)),
            set(part.loc[part.partition == valid_name, "city"].astype(str)))


def edges_within(frame: pd.DataFrame, nodes: set[str]) -> pd.DataFrame:
    return frame.loc[frame.image_a.isin(nodes) & frame.image_b.isin(nodes)].copy()


def labels_from_edges(edges: pd.DataFrame) -> dict[str, float]:
    if edges.empty:
        raise ValueError("No valid comparison edges inside training partition")
    votes = list(edges[["image_a", "image_b", "winner"]].itertuples(index=False, name=None))
    ratings = calculate_trueskill(votes)
    mus = {image_id: float(value["trueskill_mu"]) for image_id, value in ratings.items()}
    low, high = min(mus.values()), max(mus.values())
    if high <= low:
        return {image_id: 5.0 for image_id in mus}
    # The comparison winner means cleaner. Reverse normalized preference so larger VP means more polluted.
    return {image_id: 10.0 * (1.0 - (mu - low) / (high - low)) for image_id, mu in mus.items()}


def pair_accuracy(edges: pd.DataFrame, predictions: dict[str, float]):
    if edges.empty:
        return float("nan"), 0, 0
    correct = 0.0
    scored = 0
    for row in edges.itertuples(index=False):
        if row.image_a not in predictions or row.image_b not in predictions:
            continue
        winner = row.winner
        loser = row.image_b if winner == row.image_a else row.image_a
        diff = predictions[winner] - predictions[loser]
        correct += 1.0 if diff < -1e-12 else 0.5 if abs(diff) <= 1e-12 else 0.0
        scored += 1
    return (correct / scored if scored else float("nan")), scored, correct


def fit_predict(family, key, alpha, l1, train_idx, y_map, features, seg_x, predict_idx):
    train_ids = list(y_map)
    row_index = {image_id: i for i, image_id in enumerate(train_ids)}
    fit_idx = np.array([row_index[i] for i in y_map], dtype=int)
    target = np.array([y_map[i] for i in y_map], dtype=np.float64)
    if family == "dinov3_ridge":
        x = features[key]
        estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
    else:
        x = seg_x
        estimator = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000, tol=1e-5, selection="cyclic")
    scaler = StandardScaler()
    xs = scaler.fit_transform(x[fit_idx])
    xp = scaler.transform(x[predict_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(xs, target)
    pred = estimator.predict(xp)
    return {ids_for_index[i]: float(v) for i, v in zip(predict_idx, pred)}


# Keep this mapping global to avoid rebuilding it inside the thousands of candidate fits.
ids_for_index: list[str] = []


def components(edges: pd.DataFrame):
    nodes = set(edges.image_a) | set(edges.image_b)
    parent = {node: node for node in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges[["image_a", "image_b"]].itertuples(index=False, name=None):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return len(nodes), len({find(node) for node in nodes})


def main():
    global ids_for_index
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--output-dir", type=Path, default=AUDIT / "edge_isolated_unseen_city")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= len(SEEDS):
        raise ValueError(f"repeats must be between 1 and {len(SEEDS)}")

    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv")
    ids_for_index = manifest.image_id.astype(str).tolist()
    row_by_id = {image_id: i for i, image_id in enumerate(ids_for_index)}
    city_by_id = dict(zip(ids_for_index, manifest.city.astype(str)))
    features = {}
    feature_keys = []
    for edge in (384, 512):
        path = args.audit_dir / f"features/dinov3_vitb16_lvd1689m_edge{edge}.npz"
        with np.load(path) as payload:
            feature_ids = payload["image_ids"].astype(str).tolist()
            if set(feature_ids) != set(ids_for_index) or len(feature_ids) != len(ids_for_index):
                raise ValueError(f"Feature ID mismatch: {path}")
            order = {image_id: i for i, image_id in enumerate(feature_ids)}
            indices = [order[i] for i in ids_for_index]
            for view in ("cls", "patch_mean", "cls_patch"):
                key = (edge, view)
                features[key] = payload[view][indices]
                feature_keys.append(key)
    seg = pd.read_csv(ROOT / "output/segmentation_expansion_1000/elastic_net_ready.csv")
    seg_ids = seg.image_id.astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    if seg_ids.duplicated().any() or set(seg_ids) != set(ids_for_index):
        raise ValueError("Semantic feature IDs do not match manifest")
    ratio_cols = [c for c in seg.columns if c.startswith("feature_ratio_")]
    seg_order = dict(zip(seg_ids, range(len(seg_ids))))
    seg_x = seg[ratio_cols].to_numpy(dtype=np.float64)[[seg_order[i] for i in ids_for_index]]

    pairs = pd.read_csv(ROOT / "output/vlm_expansion_1000/pair_audit.csv")
    pairs = pairs.loc[(pairs.status == "calibrated") & pairs.winner.notna()].copy()
    for col in ("image_a", "image_b", "winner"):
        pairs[col] = pairs[col].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    if not set(pairs.image_a).issubset(row_by_id) or not set(pairs.image_b).issubset(row_by_id):
        raise ValueError("Comparison edges contain unknown image IDs")
    outer = pd.read_csv(args.audit_dir / "splits/unseen_city_outer.csv")
    inner = pd.read_csv(args.audit_dir / "splits/unseen_city_inner.csv")
    candidates = [("dinov3_ridge", key, float(a), np.nan) for key in feature_keys for a in RIDGE_ALPHAS]
    candidates += [("semantic_elastic_net", None, float(a), float(l1)) for a in EN_ALPHAS for l1 in EN_L1]
    pair_rows, fold_rows, search_rows, graph_rows = [], [], [], []

    for repeat in range(1, args.repeats + 1):
        for outer_fold in range(5):
            train_cities, test_cities = city_roles(outer, repeat, outer_fold)
            outer_train_ids = {image_id for image_id, city in city_by_id.items() if city in train_cities}
            outer_test_ids = {image_id for image_id, city in city_by_id.items() if city in test_cities}
            outer_train_edges = edges_within(pairs, outer_train_ids)
            outer_test_edges = edges_within(pairs, outer_test_ids)
            outer_train_labels = labels_from_edges(outer_train_edges)
            train_idx = np.array([row_by_id[i] for i in outer_train_labels], dtype=int)
            tr_nodes, tr_components = components(outer_train_edges)
            te_nodes, te_components = components(outer_test_edges)
            graph_rows.append({"repeat": repeat, "outer_fold": outer_fold, "partition": "outer_train", "images_assigned": len(outer_train_ids),
                               "images_with_edges": len(outer_train_labels), "edges": len(outer_train_edges), "components": tr_components})
            graph_rows.append({"repeat": repeat, "outer_fold": outer_fold, "partition": "outer_test", "images_assigned": len(outer_test_ids),
                               "images_with_edges": te_nodes, "images_without_edges": len(outer_test_ids) - te_nodes,
                               "edges": len(outer_test_edges), "components": te_components})

            inner_splits = []
            for inner_fold in range(4):
                tr_cities, val_cities = city_roles(inner, repeat, outer_fold, inner_fold)
                inner_train_ids = {image_id for image_id, city in city_by_id.items() if city in tr_cities}
                inner_val_ids = {image_id for image_id, city in city_by_id.items() if city in val_cities}
                tr_edges = edges_within(pairs, inner_train_ids)
                val_edges = edges_within(pairs, inner_val_ids)
                labels = labels_from_edges(tr_edges)
                train_idx_inner = np.array([row_by_id[i] for i in labels], dtype=int)
                inner_splits.append((labels, train_idx_inner, val_edges))
                n_incident, n_components = components(tr_edges)
                graph_rows.append({"repeat": repeat, "outer_fold": outer_fold, "inner_fold": inner_fold, "partition": "inner_train",
                                   "images_assigned": len(inner_train_ids), "images_with_edges": len(labels), "edges": len(tr_edges), "components": n_components})
                val_nodes, val_components = components(val_edges)
                graph_rows.append({"repeat": repeat, "outer_fold": outer_fold, "inner_fold": inner_fold, "partition": "inner_validation",
                                   "images_assigned": len(inner_val_ids), "images_with_edges": val_nodes,
                                   "images_without_edges": len(inner_val_ids) - val_nodes, "edges": len(val_edges), "components": val_components})

            scores = {ci: [] for ci in range(len(candidates))}
            edge_counts = {ci: [] for ci in range(len(candidates))}
            for labels, fit_idx, val_edges in inner_splits:
                eval_ids = sorted(set(val_edges.image_a) | set(val_edges.image_b))
                eval_idx = np.array([row_by_id[i] for i in eval_ids], dtype=int)
                id_by_row = ids_for_index
                for ci, (family, key, alpha, l1) in enumerate(candidates):
                    if family == "dinov3_ridge":
                        x = features[key]; est = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
                    else:
                        x = seg_x; est = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000, tol=1e-5, selection="cyclic")
                    target = np.array([labels[ids_for_index[i]] for i in fit_idx], dtype=float)
                    scaler = StandardScaler()
                    xt = scaler.fit_transform(x[fit_idx]); xv = scaler.transform(x[eval_idx])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        est.fit(xt, target)
                    pred = est.predict(xv)
                    pred_map = {ids_for_index[i]: float(v) for i, v in zip(eval_idx, pred)}
                    acc, n_edges, _ = pair_accuracy(val_edges, pred_map)
                    scores[ci].append(acc)
                    edge_counts[ci].append(n_edges)

            mean_acc = {ci: float(np.average(scores[ci], weights=edge_counts[ci])) for ci in scores}
            winners = {}
            for family in ("dinov3_ridge", "semantic_elastic_net"):
                family_ids = [ci for ci, c in enumerate(candidates) if c[0] == family]
                winners[family] = max(family_ids, key=lambda ci: (mean_acc[ci], -ci))
            for family, ci in winners.items():
                _, key, alpha, l1 = candidates[ci]
                search_rows.append({"repeat": repeat, "seed": SEEDS[repeat - 1], "outer_fold": outer_fold, "model": family,
                                    "inner_pair_accuracy": mean_acc[ci], "inner_edge_count": int(sum(edge_counts[ci])),
                                    "edge": key[0] if key else "", "feature_view": key[1] if key else "",
                                    "alpha": alpha, "l1_ratio": l1, "candidate_count": sum(c[0] == family for c in candidates)})

            heldout_eval_ids = sorted(set(outer_test_edges.image_a) | set(outer_test_edges.image_b))
            heldout_eval_idx = np.array([row_by_id[i] for i in heldout_eval_ids], dtype=int)
            for family, ci in winners.items():
                _, key, alpha, l1 = candidates[ci]
                if family == "dinov3_ridge":
                    x = features[key]; est = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
                else:
                    x = seg_x; est = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000, tol=1e-5, selection="cyclic")
                target = np.array([outer_train_labels[ids_for_index[i]] for i in train_idx], dtype=float)
                scaler = StandardScaler()
                xt = scaler.fit_transform(x[train_idx]); xe = scaler.transform(x[heldout_eval_idx])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    est.fit(xt, target)
                pred = est.predict(xe)
                pred_map = {ids_for_index[i]: float(v) for i, v in zip(heldout_eval_idx, pred)}
                acc, n_edges, correct_points = pair_accuracy(outer_test_edges, pred_map)
                fold_rows.append({"repeat": repeat, "outer_fold": outer_fold, "model": family,
                                  "pair_accuracy": acc, "correct_points": correct_points, "test_edges": n_edges,
                                  "train_edges": len(outer_train_edges), "train_images_with_edges": len(outer_train_labels),
                                  "test_images_with_edges": te_nodes, "test_images_without_edges": len(outer_test_ids) - te_nodes,
                                  "test_edge_components": te_components})
                for edge in outer_test_edges.itertuples(index=False):
                    winner = edge.winner
                    loser = edge.image_b if winner == edge.image_a else edge.image_a
                    diff = pred_map[winner] - pred_map[loser]
                    correctness = 1.0 if diff < -1e-12 else 0.5 if abs(diff) <= 1e-12 else 0.0
                    pair_rows.append({"repeat": repeat, "outer_fold": outer_fold, "model": family,
                                      "comparison_id": edge.comparison_id, "image_a": edge.image_a, "image_b": edge.image_b,
                                      "winner_cleaner": winner, "loser_more_polluted": loser,
                                      "predicted_vp_winner": pred_map[winner], "predicted_vp_loser": pred_map[loser],
                                      "predicted_difference_winner_minus_loser": diff,
                                      "correct": correctness, "city_a": city_by_id[edge.image_a], "city_b": city_by_id[edge.image_b]})
            print(f"repeat {repeat}/{args.repeats} outer {outer_fold + 1}/5: train edges={len(outer_train_edges)}, held-out edges={len(outer_test_edges)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_out = pd.DataFrame(pair_rows)
    pairs_out.to_csv(args.output_dir / "heldout_edge_predictions.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(args.output_dir / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(search_rows).to_csv(args.output_dir / "inner_selected_configs.csv", index=False)
    pd.DataFrame(graph_rows).to_csv(args.output_dir / "comparison_graph_audit.csv", index=False)
    repeat_rows = []
    for (repeat, model), group in pairs_out.groupby(["repeat", "model"]):
        repeat_rows.append({"repeat": int(repeat), "model": model, "heldout_edges": int(len(group)),
                            "pair_accuracy": float(group.correct.mean()),
                            "same_city_accuracy": float(group.loc[group.city_a == group.city_b, "correct"].mean()),
                            "cross_city_accuracy": float(group.loc[group.city_a != group.city_b, "correct"].mean())})
    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    summary = repeat_df.groupby("model").agg({"heldout_edges": ["mean", "std"], "pair_accuracy": ["mean", "std"],
                                              "same_city_accuracy": ["mean", "std"], "cross_city_accuracy": ["mean", "std"]})
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {"protocol": "outer unseen-city, 5-fold, 3 repeats; inner unseen-city, 4-fold",
                "valid_decisive_edges": int(len(pairs)), "ties_and_invalid_edges_excluded": True,
                "training_target": "TrueSkill mu recomputed from edges with both endpoints in current training subset; VP direction reversed and min-maxed within that training graph",
                "heldout_metric": "fraction of held-out decisive edges where predicted VP(winner cleaner) < predicted VP(loser)",
                "hyperparameter_selection": "maximum edge-weighted inner-validation pairwise accuracy",
                "label_scope_warning": "No fixed full-graph score enters this protocol. Uncertainty in the source VLM comparison judgments remains.",
                "repeat_seeds": list(SEEDS[:args.repeats]), "ridge_candidate_count": len(feature_keys)*len(RIDGE_ALPHAS),
                "elastic_net_candidate_count": len(EN_ALPHAS)*len(EN_L1)}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
