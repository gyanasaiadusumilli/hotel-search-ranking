"""End-to-end experiment: baselines vs. pointwise models vs. LambdaMART on Expedia hotel searches.

Usage:
    python -m src.run --data data/train.csv --frac 0.5
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import data, features, metrics  # noqa: E402

NAMES = {
    "random": "Random order", "cheapest_first": "Cheapest first", "highest_stars": "Highest stars first",
    "best_reviews": "Best reviews first", "location_score": "Best location first",
    "hotel_popularity": "Most booked hotels first", "logistic_regression": "Logistic regression (pointwise)",
    "lgbm_pointwise": "LightGBM classifier (pointwise)", "lambdamart": "LambdaMART (LightGBM ranker)",
    "expedia_order": "Expedia's own ranking",
}
LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def group_sizes(df):
    return df.groupby("srch_id", sort=False).size().to_numpy()


def train_models(Xtr, tr, Xva, va, seed):
    models = {}
    # LambdaMART: optimizes NDCG directly, comparing hotels within the same search.
    ranker = lgb.LGBMRanker(objective="lambdarank", random_state=seed, **LGB_PARAMS)
    ranker.fit(Xtr, tr["relevance"], group=group_sizes(tr), eval_set=[(Xva, va["relevance"])],
               eval_group=[group_sizes(va)], eval_at=[5], callbacks=[lgb.early_stopping(100, verbose=False)])
    models["lambdamart"] = ranker
    # Pointwise LightGBM: predicts "will this hotel be clicked or booked?" one row at a time.
    weight = np.where(tr["relevance"] == 5, 5.0, 1.0)
    clf = lgb.LGBMClassifier(objective="binary", random_state=seed, **LGB_PARAMS)
    clf.fit(Xtr, (tr["relevance"] > 0).astype(int), sample_weight=weight,
            eval_set=[(Xva, (va["relevance"] > 0).astype(int))], callbacks=[lgb.early_stopping(100, verbose=False)])
    models["lgbm_pointwise"] = clf
    # Logistic regression on a sample (linear model, missing values filled with training medians).
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(len(Xtr), 300_000), replace=False)
    med = Xtr.median()
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))
    lr.fit(Xtr.iloc[idx].fillna(med).clip(-1e6, 1e6), (tr["relevance"].iloc[idx] > 0).astype(int),
           logisticregression__sample_weight=weight[idx])
    models["logistic_regression"] = (lr, med)
    return models


def score_all(models, Xte, te, seed):
    rng = np.random.default_rng(seed)
    s = {
        "random": rng.random(len(te)),
        "cheapest_first": -Xte["price_per_night"].fillna(1e9).to_numpy(),
        "highest_stars": Xte["prop_starrating"].to_numpy() + 1e-3 * Xte["prop_review_score"].fillna(0).to_numpy(),
        "best_reviews": Xte["prop_review_score"].fillna(0).to_numpy(),
        "location_score": Xte["prop_location_score2"].fillna(Xte["prop_location_score1"] / 10).to_numpy(),
        "hotel_popularity": Xte["hist_booking_rate"].to_numpy(),
    }
    lr, med = models["logistic_regression"]
    s["logistic_regression"] = lr.predict_proba(Xte.fillna(med).clip(-1e6, 1e6))[:, 1]
    s["lgbm_pointwise"] = models["lgbm_pointwise"].predict_proba(Xte)[:, 1]
    s["lambdamart"] = models["lambdamart"].predict(Xte)
    s["expedia_order"] = -te["position"].to_numpy()  # only meaningful for non-random searches
    return s


def evaluate(scores, te, out: Path):
    subsets = {
        "all_test": np.ones(len(te), bool),
        "random_order_searches": te["random_bool"].to_numpy() == 1,
        "expedia_ordered_searches": te["random_bool"].to_numpy() == 0,
    }
    rows = []
    for subset, mask in subsets.items():
        if mask.sum() == 0:
            continue
        for name, sc in scores.items():
            if name == "expedia_order" and subset != "expedia_ordered_searches":
                continue
            m = metrics.summarize(te["srch_id"].to_numpy()[mask], te["relevance"].to_numpy()[mask], sc[mask])
            rows.append({"subset": subset, "model": name, **{k: round(float(v), 4) for k, v in m.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    (out / "comparison.md").write_text(table.assign(model=table["model"].map(NAMES)).to_markdown(index=False))

    # Offline A/B tests on NDCG@5 (paired over the same searches).
    q, rel = te["srch_id"].to_numpy(), te["relevance"].to_numpy()
    nd = {k: metrics.per_search_ndcg(q, rel, v, 5) for k, v in scores.items()}
    ab = {}
    best_baseline = table[(table.subset == "all_test") & table.model.isin(
        ["cheapest_first", "highest_stars", "best_reviews", "location_score", "hotel_popularity"])].sort_values("ndcg@5").iloc[-1]["model"]
    ab[f"lambdamart_vs_{best_baseline}"] = metrics.paired_bootstrap(nd[best_baseline], nd["lambdamart"])
    ab["lambdamart_vs_lgbm_pointwise"] = metrics.paired_bootstrap(nd["lgbm_pointwise"], nd["lambdamart"])
    mask = te["random_bool"].to_numpy() == 0
    if mask.any():
        qe, re_ = q[mask], rel[mask]
        ab["lambdamart_vs_expedia_order"] = metrics.paired_bootstrap(
            metrics.per_search_ndcg(qe, re_, scores["expedia_order"][mask], 5),
            metrics.per_search_ndcg(qe, re_, scores["lambdamart"][mask], 5))
    (out / "ab_tests.json").write_text(json.dumps(ab, indent=2))
    return table, ab


def plot_comparison(table, out: Path):
    d = table[table.subset == "all_test"].sort_values("ndcg@5")
    best = d["ndcg@5"].max()
    learned = {"logistic_regression", "lgbm_pointwise", "lambdamart"}
    colors = ["#0B7C8A" if v == best else ("#6F8A92" if m in learned else "#B9C6C9") for m, v in zip(d.model, d["ndcg@5"])]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(d))
    ax.barh(y, d["ndcg@5"], color=colors, height=0.66)
    for yi, v in zip(y, d["ndcg@5"]):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=9, color="#222", fontweight="bold" if v == best else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES[m] for m in d.model])
    ax.set_xlim(0, best * 1.15)
    ax.set_xlabel("NDCG@5 on held-out future searches (higher is better)")
    ax.set_title("Hotel search ranking: model comparison", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.01, "Light grey: rule-based baselines. Grey: learned models. Teal: best model.", ha="center", fontsize=8.5, color="#555")
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out / "comparison_ndcg.png", dpi=160)
    plt.close(fig)


def plot_importance(ranker, cols, out: Path, top=15):
    imp = pd.Series(ranker.booster_.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(range(len(imp)), imp.values, color="#0B7C8A", height=0.66)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index)
    ax.set_xlabel("Share of total split gain")
    ax.set_title(f"LambdaMART: top {top} features", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=160)
    plt.close(fig)
    imp.sort_values(ascending=False).round(4).to_csv(out / "feature_importance.csv", header=["gain_share"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--frac", type=float, default=0.5, help="fraction of searches to use")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out, mdir = Path(args.out), Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    df = data.load(args.data, args.frac)
    tr, va, te = data.time_split(df)
    del df
    print(f"Searches  train {tr.srch_id.nunique():,}  val {va.srch_id.nunique():,}  test {te.srch_id.nunique():,}  "
          f"(rows {len(tr):,} / {len(va):,} / {len(te):,})")
    Xtr, Xva, Xte = features.build(tr, va, te)
    print(f"{Xtr.shape[1]} features")

    models = train_models(Xtr, tr, Xva, va, args.seed)
    models["lambdamart"].booster_.save_model(str(mdir / "lambdamart.txt"))
    scores = score_all(models, Xte, te, args.seed)
    table, ab = evaluate(scores, te, out)
    plot_comparison(table, out)
    plot_importance(models["lambdamart"], list(Xtr.columns), out)

    info = {"frac": args.frac, "train_searches": int(tr.srch_id.nunique()), "val_searches": int(va.srch_id.nunique()),
            "test_searches": int(te.srch_id.nunique()), "n_features": int(Xtr.shape[1]),
            "lambdamart_trees": int(models["lambdamart"].best_iteration_ or LGB_PARAMS["n_estimators"]),
            "train_period": [str(tr.date_time.min()), str(tr.date_time.max())],
            "test_period": [str(te.date_time.min()), str(te.date_time.max())]}
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(table[table.subset == "all_test"].to_string(index=False))
    print(json.dumps(ab, indent=2))


if __name__ == "__main__":
    main()
