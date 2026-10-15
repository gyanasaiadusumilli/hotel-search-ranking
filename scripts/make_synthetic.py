"""Generate a small synthetic file with the same columns as Expedia's train.csv.

Only for testing the pipeline end to end. Results on this file mean nothing.
Usage: python scripts/make_synthetic.py --out data/synthetic.csv --searches 3000
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.csv")
    ap.add_argument("--searches", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n_props = 4000
    props = pd.DataFrame({
        "prop_id": np.arange(n_props), "prop_country_id": rng.integers(0, 30, n_props),
        "prop_starrating": rng.integers(0, 6, n_props), "prop_review_score": rng.choice(np.arange(0, 5.5, 0.5), n_props),
        "prop_brand_bool": rng.integers(0, 2, n_props), "prop_location_score1": rng.uniform(0, 7, n_props).round(2),
        "prop_location_score2": np.where(rng.random(n_props) < 0.2, np.nan, rng.beta(1, 8, n_props)).round(4),
        "quality": rng.normal(0, 1, n_props),
    })
    props["base_price"] = np.exp(3.8 + 0.3 * props.prop_starrating + rng.normal(0, 0.4, n_props))
    props["prop_log_historical_price"] = np.where(rng.random(n_props) < 0.15, 0, np.log(props.base_price)).round(2)
    rows = []
    start = pd.Timestamp("2012-11-01")
    for q in range(1, a.searches + 1):
        k = rng.integers(10, 39)
        p = props.sample(k, random_state=int(rng.integers(1e9)))
        has_hist = rng.random() < 0.05
        r = pd.DataFrame({
            "srch_id": q, "date_time": start + pd.Timedelta(minutes=int(q * 240000 / a.searches)) + pd.Timedelta(seconds=int(rng.integers(0, 600))),
            "site_id": int(rng.integers(1, 35)), "visitor_location_country_id": int(rng.integers(1, 230)),
            "visitor_hist_starrating": rng.uniform(2, 5) if has_hist else np.nan,
            "visitor_hist_adr_usd": rng.uniform(50, 300) if has_hist else np.nan,
        }, index=range(k))
        for c in ["prop_country_id", "prop_id", "prop_starrating", "prop_review_score", "prop_brand_bool",
                  "prop_location_score1", "prop_location_score2", "prop_log_historical_price"]:
            r[c] = p[c].to_numpy()
        r["price_usd"] = (p.base_price.to_numpy() * rng.uniform(0.7, 1.3, k)).round(2)
        r["promotion_flag"] = rng.integers(0, 2, k)
        r["srch_destination_id"] = rng.integers(1, 20000)
        r["srch_length_of_stay"] = rng.integers(1, 8)
        r["srch_booking_window"] = rng.integers(0, 120)
        r["srch_adults_count"] = rng.integers(1, 5)
        r["srch_children_count"] = rng.integers(0, 3)
        r["srch_room_count"] = 1
        r["srch_saturday_night_bool"] = rng.integers(0, 2)
        r["srch_query_affinity_score"] = np.where(rng.random(k) < 0.9, np.nan, -rng.uniform(5, 60, k))
        r["orig_destination_distance"] = np.where(rng.random(k) < 0.3, np.nan, rng.uniform(1, 5000, k))
        random_bool = int(rng.random() < 0.3)
        r["random_bool"] = random_bool
        for i in range(1, 9):
            r[f"comp{i}_rate"] = rng.choice([np.nan, -1, 0, 1], k, p=[0.6, 0.1, 0.2, 0.1])
            r[f"comp{i}_inv"] = rng.choice([np.nan, 0, 1], k, p=[0.6, 0.35, 0.05])
            r[f"comp{i}_rate_percent_diff"] = np.where(rng.random(k) < 0.8, np.nan, rng.uniform(1, 60, k))
        rel_price = r.price_usd / r.price_usd.median()
        util = (p.quality.to_numpy() + 0.4 * r.prop_review_score + 0.25 * r.prop_starrating
                + 4 * r.prop_location_score2.fillna(0) - 1.2 * np.log(rel_price) + 0.3 * r.promotion_flag).to_numpy()
        order = rng.permutation(k) if random_bool else np.argsort(-(util + rng.normal(0, 1.5, k)))
        r = r.iloc[order].reset_index(drop=True)
        util = util[order]
        r["position"] = np.arange(1, k + 1)
        pos_bias = 1 / np.log2(r.position + 1)
        click_p = 1 / (1 + np.exp(-(util - 4.5))) * pos_bias
        click = rng.random(k) < click_p
        book = np.zeros(k, bool)
        if click.any() and rng.random() < 0.7:
            cand = np.flatnonzero(click)
            book[cand[np.argmax(util[cand])]] = True
        r["click_bool"] = click.astype(int)
        r["booking_bool"] = book.astype(int)
        r["gross_bookings_usd"] = np.where(book, r.price_usd * r.srch_length_of_stay, np.nan)
        rows.append(r)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} rows, {a.searches} searches -> {a.out}")


if __name__ == "__main__":
    main()
