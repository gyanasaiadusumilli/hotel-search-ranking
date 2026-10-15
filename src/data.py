"""Load the Expedia hotel search data, build relevance labels and split by time.

Each row is one hotel (prop_id) shown in one search (srch_id).
Relevance label: 5 if the hotel was booked, 1 if it was clicked but not booked, 0 otherwise.
"""
import numpy as np
import pandas as pd

DROP_AT_SERVING = ["position", "click_bool", "booking_bool", "gross_bookings_usd"]  # only known after the search


def _keep_search(srch_id: pd.Series, frac: float) -> pd.Series:
    """Deterministic subsample of whole searches (no first pass over the file needed)."""
    if frac >= 1:
        return pd.Series(True, index=srch_id.index)
    return (srch_id.astype(np.int64) * 2654435761 % 10_000) < frac * 10_000


def load(path: str, frac: float = 1.0, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read train.csv, keeping a fraction `frac` of searches (all rows of each kept search)."""
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, parse_dates=["date_time"]):
        chunk = chunk[_keep_search(chunk["srch_id"], frac)]
        floats = chunk.select_dtypes("float64").columns
        chunk[floats] = chunk[floats].astype("float32")
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df["relevance"] = np.where(df["booking_bool"] == 1, 5, np.where(df["click_bool"] == 1, 1, 0)).astype("int8")
    return df.sort_values(["srch_id", "position"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.2):
    """Split whole searches by date: oldest for training, then validation, newest for testing.

    A time split mimics production, where a model trained on the past ranks future searches.
    """
    search_time = df.groupby("srch_id")["date_time"].first().sort_values()
    n = len(search_time)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    ids = search_time.index.to_numpy()
    test_ids = set(ids[n - n_test:])
    val_ids = set(ids[n - n_test - n_val:n - n_test])
    split = np.where(df["srch_id"].isin(test_ids), "test", np.where(df["srch_id"].isin(val_ids), "val", "train"))
    return df[split == "train"].copy(), df[split == "val"].copy(), df[split == "test"].copy()
