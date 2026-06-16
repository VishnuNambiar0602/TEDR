"""preprocess_traffic.py
Download and preprocess the 'umairshahpirzada/traffic-net' dataset (or a file within it)
so it's ready for traffic prediction models (time-series forecasting).

Usage (after installing requirements):
python preprocess_traffic.py --file <filename_in_dataset.csv> --freq H --lookback 24 --horizon 1 --output_dir data/processed

If --file isn't provided the script will try common filenames; otherwise supply the exact path inside the Kaggle dataset.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except Exception as e:  # pragma: no cover - helpful message rather than failing hard
    kagglehub = None

from sklearn.preprocessing import StandardScaler, MinMaxScaler


COMMON_FILE_CANDIDATES = [
    "train.csv",
    "train.csv.gz",
    "data.csv",
    "traffic.csv",
    "Traffic.csv",
    "traffic_volume.csv",
    "traffic_volume.csv.gz",
]


def try_load_dataset(dataset: str, file_path: Optional[str] = None) -> pd.DataFrame:
    """Load dataset using kagglehub into a pandas DataFrame. If file_path is None, try common candidates.

    Returns a DataFrame or raises a helpful error.
    """
    if kagglehub is None:
        raise RuntimeError(
            "kagglehub is not installed or importable. Run: pip install kagglehub[pandas-datasets]"
        )

    if file_path:
        print(f"Loading '{file_path}' from dataset {dataset}...")
        return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset, file_path)

    # try common candidates
    for candidate in COMMON_FILE_CANDIDATES:
        try:
            print(f"Trying to load '{candidate}'...")
            df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset, candidate)
            print(f"Loaded using candidate: {candidate}")
            return df
        except Exception:
            continue

    raise FileNotFoundError(
        "Could not find a csv file automatically. Re-run with the --file argument and point to the CSV inside the dataset."
    )


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ("time", "date", "datetime"))]
    for c in candidates:
        try:
            _ = pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    # fallback: try any column with datetime-like dtype
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    numeric = df.select_dtypes(include=["number"]).columns
    # prefer columns that contain traffic-related keywords
    keywords = ["traffic", "volume", "count", "vehicle", "cars", "flow", "speed"]
    for k in keywords:
        for c in numeric:
            if k in c.lower():
                return c
    # fallback: choose first numeric column
    if len(numeric) > 0:
        return numeric[0]
    return None


def preprocess_time_series(
    df: pd.DataFrame,
    datetime_col: Optional[str] = None,
    target_col: Optional[str] = None,
    freq: str = "H",
    lags: int = 24,
    rolling_windows: tuple = (3, 24),
) -> Tuple[pd.DataFrame, str]:
    """Detect important columns, resample, fill gaps, create features and lags.

    Returns (processed_df, target_col)
    """
    df = df.copy()

    if datetime_col is None:
        datetime_col = detect_datetime_column(df)
    if datetime_col is None:
        raise ValueError("No datetime column detected. Please provide --datetime-column.")

    print(f"Detected datetime column: {datetime_col}")
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    if target_col is None:
        target_col = detect_target_column(df)
    if target_col is None:
        raise ValueError("No numeric target column detected. Please provide --target.")

    print(f"Detected target column: {target_col}")

    # Resample to fixed frequency using sum for counts, mean otherwise
    if "count" in target_col.lower() or "volume" in target_col.lower():
        agg = "sum"
    else:
        agg = "mean"

    df_resampled = getattr(df.resample(freq), agg)()

    # Fill small gaps: interpolate for continuous measures, forward-fill for counts
    if agg == "mean":
        df_resampled = df_resampled.interpolate(method="time").ffill().bfill()
    else:
        df_resampled = df_resampled.fillna(0)

    # Time features
    df_resampled["hour"] = df_resampled.index.hour
    df_resampled["dow"] = df_resampled.index.dayofweek
    df_resampled["month"] = df_resampled.index.month
    df_resampled["is_weekend"] = df_resampled["dow"].isin([5, 6]).astype(int)

    # cyclical features
    df_resampled["hour_sin"] = np.sin(2 * np.pi * df_resampled["hour"] / 24)
    df_resampled["hour_cos"] = np.cos(2 * np.pi * df_resampled["hour"] / 24)

    # lags and rolling stats on the target
    for lag in range(1, lags + 1):
        df_resampled[f"lag_{lag}"] = df_resampled[target_col].shift(lag)
    for w in rolling_windows:
        df_resampled[f"rolling_mean_{w}"] = df_resampled[target_col].shift(1).rolling(w).mean()

    # drop rows with NaNs produced by shifting
    df_resampled = df_resampled.dropna()

    return df_resampled, target_col


def train_val_test_split_by_time(df: pd.DataFrame, train_pct=0.7, val_pct=0.15):
    n = len(df)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def scale_dfs(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols=None):
    if feature_cols is None:
        feature_cols = train.columns
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    train_s = pd.DataFrame(scaler.transform(train[feature_cols]), index=train.index, columns=feature_cols)
    val_s = pd.DataFrame(scaler.transform(val[feature_cols]), index=val.index, columns=feature_cols)
    test_s = pd.DataFrame(scaler.transform(test[feature_cols]), index=test.index, columns=feature_cols)
    return scaler, train_s, val_s, test_s


def create_supervised(df: pd.DataFrame, target_col: str, lookback: int = 24, horizon: int = 1):
    """Create X, y for supervised learning where y is horizon steps ahead of the last lookback.

    Returns X (n_samples, lookback, n_features) and y (n_samples, )
    """
    features = df.columns
    Xs = []
    ys = []
    for i in range(lookback, len(df) - horizon + 1):
        Xs.append(df.iloc[i - lookback : i].values)
        ys.append(df.iloc[i + horizon - 1][target_col])
    X = np.array(Xs)
    y = np.array(ys)
    return X, y


def save_processed(train, val, test, output_dir: str = "data/processed", prefix: str = "traffic"):
    os.makedirs(output_dir, exist_ok=True)
    train.to_parquet(os.path.join(output_dir, f"{prefix}_train.parquet"))
    val.to_parquet(os.path.join(output_dir, f"{prefix}_val.parquet"))
    test.to_parquet(os.path.join(output_dir, f"{prefix}_test.parquet"))
    print(f"Saved processed data to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="umairshahpirzada/traffic-net")
    parser.add_argument("--file", default=None, help="CSV file inside the dataset (e.g., 'train.csv')")
    parser.add_argument("--datetime-column", default=None)
    parser.add_argument("--target", default=None, help="Name of the target column to predict")
    parser.add_argument("--freq", default="H", help="Resampling frequency (pandas offset alias). Default: H (hourly)")
    parser.add_argument("--lags", type=int, default=24)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    df = try_load_dataset(args.dataset, args.file)
    print("Raw columns:", df.columns.tolist())
    processed, target_col = preprocess_time_series(
        df, args.datetime_column, args.target, freq=args.freq, lags=args.lags
    )

    train, val, test = train_val_test_split_by_time(processed)
    scaler, train_s, val_s, test_s = scale_dfs(train, val, test)
    save_processed(train_s, val_s, test_s, output_dir=args.output_dir)

    # Create supervised arrays and save small sample for quick training start
    X_train, y_train = create_supervised(train_s, target_col, lookback=args.lookback, horizon=args.horizon)
    X_val, y_val = create_supervised(val_s, target_col, lookback=args.lookback, horizon=args.horizon)
    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(args.output_dir, "y_val.npy"), y_val)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
