"""Feature engineering for hotel ranking.

Four groups of features:
1. Raw hotel, search and visitor fields.
2. Within-search features: how a hotel compares with the other hotels in the same results list
   (price rank, price relative to the median, star and review rank, ...). Rankers care about
   relative value, not absolute price.
3. Visitor fit: distance between a hotel and the visitor's booking history (stars and price).
4. Hotel popularity: historical click and booking rates per hotel, computed only from training
   searches with out-of-fold target encoding so a row never sees its own label.
"""
import numpy as np
import pandas as pd

RAW = [
    "prop_starrating", "prop_review_score", "prop_brand_bool", "prop_location_score1",
    "prop_location_score2", "prop_log_historical_price", "price_usd", "promotion_flag",
    "srch_length_of_stay", "srch_booking_window", "srch_adults_count", "srch_children_count",
    "srch_room_count", "srch_saturday_night_bool", "srch_query_affinity_score",
    "orig_destination_distance", "visitor_hist_starrating", "visitor_hist_adr_usd",
]
COMP = range(1, 9)
PRIOR = 50  # smoothing strength for hotel popularity rates


def _within_search(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("srch_id")
    out = pd.DataFrame(index=df.index)
    out["n_results"] = g["prop_id"].transform("size")
    for col, asc in [("price_per_night", True), ("prop_starrating", False), ("prop_review_score", False),
                     ("prop_location_score2", False), ("prop_location_score1", False)]:
        out[f"{col}_rank"] = g[col].rank(pct=True, ascending=asc)
    med = g["price_per_night"].transform("median")
    out["price_vs_median"] = df["price_per_night"] / med.replace(0, np.nan)
    std = g["price_per_night"].transform("std")
    out["price_z"] = (df["price_per_night"] - g["price_per_night"].transform("mean")) / std.replace(0, np.nan)
    out["star_minus_mean"] = df["prop_starrating"] - g["prop_starrating"].transform("mean")
    out["review_minus_mean"] = df["prop_review_score"] - g["prop_review_score"].transform("mean")
    out["value_score"] = df["prop_review_score"] / np.log1p(df["price_per_night"].clip(lower=1))
    out["value_rank"] = out["value_score"].groupby(df["srch_id"]).rank(pct=True, ascending=False)
    return out


def _target_encode(train: pd.DataFrame, others: list, col: str, n_folds: int = 5, seed: int = 0):
    """Smoothed click and booking rates per hotel. Out-of-fold for training rows."""
    rng = np.random.default_rng(seed)
    search_fold = pd.Series(rng.integers(0, n_folds, train["srch_id"].nunique()), index=train["srch_id"].unique())
    fold = train["srch_id"].map(search_fold).to_numpy()
    p_click, p_book = train["click_bool"].mean(), train["booking_bool"].mean()

    def rates(src):
        s = src.groupby(col).agg(n=("click_bool", "size"), c=("click_bool", "sum"), b=("booking_bool", "sum"))
        s["ctr"] = (s["c"] + PRIOR * p_click) / (s["n"] + PRIOR)
        s["br"] = (s["b"] + PRIOR * p_book) / (s["n"] + PRIOR)
        return s

    train_enc = pd.DataFrame(index=train.index, columns=["hist_ctr", "hist_booking_rate", "hist_impressions"], dtype="float64")
    for k in range(n_folds):
        s = rates(train[fold != k])
        idx = train.index[fold == k]
        keys = train.loc[idx, col]
        train_enc.loc[idx, "hist_ctr"] = keys.map(s["ctr"]).fillna(p_click).to_numpy()
        train_enc.loc[idx, "hist_booking_rate"] = keys.map(s["br"]).fillna(p_book).to_numpy()
        train_enc.loc[idx, "hist_impressions"] = keys.map(s["n"]).fillna(0).to_numpy()
    s = rates(train)
    other_enc = []
    for df in others:
        e = pd.DataFrame(index=df.index)
        e["hist_ctr"] = df[col].map(s["ctr"]).fillna(p_click).astype("float32")
        e["hist_booking_rate"] = df[col].map(s["br"]).fillna(p_book).astype("float32")
        e["hist_impressions"] = df[col].map(s["n"]).fillna(0).astype("float32")
        other_enc.append(e)
    return train_enc.astype("float32"), other_enc


def _row_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[RAW].copy()
    out["price_per_night"] = df["price_per_night"]
    out["log_price"] = np.log1p(df["price_per_night"].clip(lower=0))
    out["star_diff_from_history"] = (df["visitor_hist_starrating"] - df["prop_starrating"]).abs()
    out["price_diff_from_history"] = (df["visitor_hist_adr_usd"] - df["price_per_night"]).abs()
    hist_price = np.exp(df["prop_log_historical_price"].where(df["prop_log_historical_price"] > 0))
    out["price_vs_hotel_history"] = df["price_per_night"] - hist_price
    rate = df[[f"comp{i}_rate" for i in COMP]]
    out["comp_cheaper_here"] = (rate == 1).sum(axis=1)   # competitor is more expensive than Expedia
    out["comp_cheaper_there"] = (rate == -1).sum(axis=1)
    out["comp_has_availability"] = (df[[f"comp{i}_inv" for i in COMP]] == 0).sum(axis=1)
    out["comp_mean_pct_diff"] = df[[f"comp{i}_rate_percent_diff" for i in COMP]].clip(upper=200).mean(axis=1)
    return out


def build(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Return feature matrices for train, val and test (same columns in the same order)."""
    for df in (train, val, test):
        # price_usd is per night on some sites and per stay on others; dividing long stays helps.
        df["price_per_night"] = np.where(df["price_usd"] > 1000, df["price_usd"] / df["srch_length_of_stay"].clip(lower=1),
                                         df["price_usd"]).astype("float32")
    tr_enc, (va_enc, te_enc) = _target_encode(train, [val, test], "prop_id")
    mats = []
    for df, enc in [(train, tr_enc), (val, va_enc), (test, te_enc)]:
        X = pd.concat([_row_features(df), _within_search(df), enc], axis=1)
        X["hist_ctr_rank"] = X["hist_ctr"].groupby(df["srch_id"]).rank(pct=True, ascending=False)
        X["hist_booking_rate_rank"] = X["hist_booking_rate"].groupby(df["srch_id"]).rank(pct=True, ascending=False)
        mats.append(X.astype("float32"))
    return mats
