"""Ranking metrics and an offline A/B comparison (paired bootstrap over searches)."""
import numpy as np
import pandas as pd


def per_search_ndcg(srch_id, relevance, score, k: int) -> pd.Series:
    """NDCG@k for every search (gain 2^rel - 1, log2 discount). Searches without any click are skipped."""
    df = pd.DataFrame({"q": np.asarray(srch_id), "rel": np.asarray(relevance), "s": np.asarray(score)})
    df = df[df.groupby("q")["rel"].transform("max") > 0]
    # Ties are broken by row order, so put a random order first to avoid flattering any baseline.
    df = df.sample(frac=1.0, random_state=0)
    df["pos"] = df.groupby("q")["s"].rank(method="first", ascending=False)
    df["ideal"] = df.groupby("q")["rel"].rank(method="first", ascending=False)
    gain = 2.0 ** df["rel"] - 1
    dcg = (gain / np.log2(df["pos"] + 1)).where(df["pos"] <= k, 0).groupby(df["q"]).sum()
    idcg = (gain / np.log2(df["ideal"] + 1)).where(df["ideal"] <= k, 0).groupby(df["q"]).sum()
    return (dcg / idcg).rename(f"ndcg@{k}")


def booked_in_top(srch_id, relevance, score, k: int) -> pd.Series:
    """For searches that ended in a booking: was the booked hotel ranked in the top k?"""
    df = pd.DataFrame({"q": np.asarray(srch_id), "rel": np.asarray(relevance), "s": np.asarray(score)})
    df = df[df.groupby("q")["rel"].transform("max") == 5].sample(frac=1.0, random_state=0)
    df["pos"] = df.groupby("q")["s"].rank(method="first", ascending=False)
    return (df[df["rel"] == 5].groupby("q")["pos"].min() <= k).astype(float)


def summarize(srch_id, relevance, score) -> dict:
    return {
        "ndcg@5": per_search_ndcg(srch_id, relevance, score, 5).mean(),
        "ndcg@10": per_search_ndcg(srch_id, relevance, score, 10).mean(),
        "ndcg@38": per_search_ndcg(srch_id, relevance, score, 38).mean(),
        "booked_in_top5": booked_in_top(srch_id, relevance, score, 5).mean(),
    }


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """Offline A/B test: is ranker B better than ranker A on the same searches?

    a and b are per-search metric values. Searches are resampled with replacement, which
    keeps each search's pair of values together.
    """
    diff = (b - a).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "relative_lift": float(diff.mean() / a.mean()), "p_value": float(max(p, 1 / n_boot)),
            "n_searches": int(len(diff))}
