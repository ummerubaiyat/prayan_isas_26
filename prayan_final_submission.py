#!/usr/bin/env python3
"""
Prayan - ISAS 2026 final BLE location submission
=================================================

Trains the frozen final class-aware multi-resolution ensemble on ALL labeled
Apr10-Apr13 development data and predicts the organizer-supplied Apr14 test
rows. The original test CSV is preserved row-for-row and one prediction column
is appended.

This script is intentionally deployment-only:
  * it does not read prior probability/prediction caches;
  * it does not use Apr14 labels (they are unavailable);
  * it rebuilds the BLE state/features from the raw training BLE files and the
    organizer test CSV;
  * it uses the frozen architecture/hyperparameters selected before deployment.

Default output:
    /app/output/Prayan_prediction.csv

Expected organizer test template (verified from BLE_Test_predict.csv):
    62,222 rows
    columns: Unnamed: 0, user_id, timestamp, mac address, RSSI, power
    Apr14 2023, user_id=90
    numeric beacon ids (observed 1..23; model space remains PROX_1..PROX_25)
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE
import xgboost as xgb


# =============================================================================
# Frozen challenge/model definitions
# =============================================================================

SEED = 42
MODEL_MISSING_RSSI = -120.0
LABEL_USER_ID = 97
BLE_USER_ID = 90

OFFICIAL_CLASSES = [
    "501", "502", "503", "505", "506", "508", "510", "511", "512", "513",
    "515", "516", "517", "518", "520", "521", "522", "523",
    "cafeteria", "cleaning", "hallway", "kitchen", "nurse station",
]
OFFICIAL_ARRAY = np.asarray(OFFICIAL_CLASSES, dtype=str)
CLASS_TO_IDX = {c: i for i, c in enumerate(OFFICIAL_CLASSES)}
IDX_510 = CLASS_TO_IDX["510"]
IDX_518 = CLASS_TO_IDX["518"]

BEACONS = [f"PROX_{i}" for i in range(1, 26)]
OBS_BEACONS = [f"OBS_{b}" for b in BEACONS]

# Frozen MAC -> beacon mapping used by the development benchmark.
MAC_LIST = [
    "F7:7F:78:76:7E:F3", "C6:CD:5E:3D:2F:BB", "D6:F4:3A:79:74:63", "C9:17:55:E2:3E:0E",
    "CA:60:AB:EE:EC:7F", "D6:51:7F:AB:0E:29", "CC:54:33:F6:A7:90", "EB:20:56:87:04:5A",
    "EE:E7:46:DC:19:6F", "C8:5B:BF:37:07:A0", "D7:26:F6:A3:44:D2", "DD:83:B0:27:FD:36",
    "E5:CD:4A:36:87:06", "DC:22:B8:17:4E:B5", "EA:09:20:80:D6:44", "E6:99:D1:EC:C6:81",
    "F6:DA:97:C7:D5:28", "EA:66:A1:12:2C:F4", "C9:EA:57:8B:0F:80", "D6:7C:1D:2C:2A:0A",
    "DA:E1:70:5F:44:97", "DD:10:10:F6:4F:27", "E6:F3:93:A8:9E:22", "E6:60:05:1F:88:F9",
    "D4:33:FD:F4:C2:A8",
]
MAC_TO_BEACON_ID = {mac: f"PROX_{i + 1}" for i, mac in enumerate(MAC_LIST)}

# Six-position room topology used by the relabeling engine.
RELABEL_TOPOLOGY = {
    "501": ["PROX_None", "PROX_None", "PROX_1", "PROX_13", "PROX_2", "PROX_15"],
    "502": ["PROX_13", "PROX_1", "PROX_2", "PROX_15", "PROX_3", "PROX_16"],
    "503": ["PROX_15", "PROX_2", "PROX_3", "PROX_16", "PROX_5", "PROX_17"],
    "505": ["PROX_16", "PROX_3", "PROX_5", "PROX_17", "PROX_6", "PROX_14"],
    "506": ["PROX_17", "PROX_5", "PROX_6", "PROX_14", "PROX_None", "PROX_None"],
    "513": ["PROX_None", "PROX_None", "PROX_13", "PROX_1", "PROX_15", "PROX_2"],
    "515": ["PROX_1", "PROX_13", "PROX_15", "PROX_2", "PROX_16", "PROX_3"],
    "516": ["PROX_2", "PROX_15", "PROX_16", "PROX_3", "PROX_17", "PROX_5"],
    "517": ["PROX_3", "PROX_16", "PROX_17", "PROX_5", "PROX_14", "PROX_6"],
    "507": ["PROX_None", "PROX_None", "PROX_7", "PROX_18", "PROX_8", "PROX_20"],
    "508": ["PROX_18", "PROX_7", "PROX_8", "PROX_20", "PROX_10", "PROX_21"],
    "510": ["PROX_20", "PROX_8", "PROX_10", "PROX_21", "PROX_11", "PROX_22"],
    "511": ["PROX_21", "PROX_10", "PROX_11", "PROX_22", "PROX_12", "PROX_23"],
    "512": ["PROX_22", "PROX_11", "PROX_12", "PROX_23", "PROX_None", "PROX_None"],
    "518": ["PROX_None", "PROX_None", "PROX_18", "PROX_7", "PROX_20", "PROX_8"],
    "520": ["PROX_7", "PROX_18", "PROX_20", "PROX_8", "PROX_21", "PROX_10"],
    "521": ["PROX_8", "PROX_20", "PROX_21", "PROX_10", "PROX_22", "PROX_11"],
    "522": ["PROX_10", "PROX_21", "PROX_22", "PROX_11", "PROX_23", "PROX_12"],
    "523": ["PROX_11", "PROX_22", "PROX_23", "PROX_12", "PROX_None", "PROX_None"],
}

SYMMETRIC_DONORS = {
    "501": "513", "513": "501", "502": "515", "515": "502",
    "503": "516", "516": "503", "505": "517", "517": "505",
    "507": "518", "518": "507", "508": "520", "520": "508",
    "510": "521", "521": "510", "511": "522", "522": "511",
}

# Only the three frozen final-system constituents are needed for deployment.
FINAL_CONFIGS = {
    "rich_xgb": {
        "model": "xgb", "window": 60, "train_stride": 1,
        "feature_set": "phase5_rich", "augmentation": "symmetric_smote",
    },
    "full_xgb": {
        "model": "xgb", "window": 60, "train_stride": 10,
        "feature_set": "all", "augmentation": "kl_partial",
    },
    "aug_rf": {
        "model": "rf", "window": 10, "train_stride": 1,
        "feature_set": "basic", "augmentation": "symmetric_smote",
    },
}

ENSEMBLE_TEMPERATURES = {"rich_xgb": 1.25, "full_xgb": 1.00, "aug_rf": 1.00}
ENSEMBLE_WEIGHTS = {"rich_xgb": 0.15, "full_xgb": 0.15, "aug_rf": 0.70}
ENSEMBLE_ROLLING_WINDOW = 11
ROOM510_RANK_MAX = 2
ROOM510_PROB_MIN = 0.15
ROOM510_MULTIPLIER = 1.5
ROOM518_VOTE_MULTIPLIER = 3.5


# =============================================================================
# CLI and reproducibility
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Generate Prayan ISAS 2026 BLE test submission.")
    p.add_argument("--label-file", default="/app/data/5f_label_loc_train.csv")
    p.add_argument("--ble-dir", default="/app/data/BLE Data")
    p.add_argument("--test-file", default="/app/data/BLE_Test_predict.csv")
    p.add_argument("--output", default="/app/output/Prayan_prediction.csv")
    p.add_argument(
        "--prediction-column",
        default="prediction",
        help=(
            "Name of the appended location-prediction column. The tutorial transcript "
            "requires one extra prediction column but does not state an exact header."
        ),
    )
    p.add_argument(
        "--strict-test-template",
        action="store_true",
        help="Require the currently supplied organizer test template characteristics (62,222 rows, Apr14).",
    )
    return p.parse_args()


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# =============================================================================
# Generic helpers
# =============================================================================

def parse_beacon_identifier(value):
    text = str(value).strip().upper()
    if text in MAC_TO_BEACON_ID:
        return MAC_TO_BEACON_ID[text]
    try:
        n = int(float(text))
        if 1 <= n <= 25:
            return f"PROX_{n}"
    except (TypeError, ValueError):
        pass
    return None


def ensure_probability_matrix(p, n_rows, label):
    p = np.asarray(p, dtype=np.float64)
    expected = (n_rows, len(OFFICIAL_CLASSES))
    if p.shape != expected:
        raise RuntimeError(f"{label}: expected probability shape {expected}, got {p.shape}")
    if not np.all(np.isfinite(p)):
        raise RuntimeError(f"{label}: non-finite probabilities")
    if np.any(p < -1e-12):
        raise RuntimeError(f"{label}: negative probabilities")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise RuntimeError(f"{label}: probability rows do not sum to 1")


def align_probabilities(local_classes, local_probs):
    local_classes = np.asarray(local_classes, dtype=str)
    local_probs = np.asarray(local_probs, dtype=np.float64)
    out = np.zeros((len(local_probs), len(OFFICIAL_CLASSES)), dtype=np.float64)
    for j, cls in enumerate(local_classes):
        if cls not in CLASS_TO_IDX:
            raise RuntimeError(f"Unexpected model class: {cls}")
        out[:, CLASS_TO_IDX[cls]] = local_probs[:, j]
    ensure_probability_matrix(out, len(local_probs), "Aligned probabilities")
    return out


def apply_temperature(probs, temperature):
    p = np.asarray(probs, dtype=np.float64)
    if float(temperature) == 1.0:
        return p.copy()
    scaled = np.zeros_like(p)
    positive = p > 0
    scaled[positive] = np.power(p[positive], 1.0 / float(temperature))
    denom = scaled.sum(axis=1, keepdims=True)
    if np.any(denom <= 0):
        raise RuntimeError("Temperature adjustment generated a zero-sum row")
    return scaled / denom


def build_augmented_rf():
    # Exact augmented-RF configuration used by the frozen final branch.
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    )


def build_xgb(num_classes):
    return xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.07,
        max_depth=7,
        objective="multi:softprob",
        num_class=num_classes,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


# =============================================================================
# Input loading / BLE reconstruction
# =============================================================================

def clean_labels(label_file):
    df = pd.read_csv(label_file)
    required = {"user_id", "activity", "started_at", "finished_at", "room"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Label file missing columns: {sorted(missing)}")

    df = df[(df["user_id"] == LABEL_USER_ID) & (df["activity"] == "Location")].copy()
    if "deleted_at" in df.columns:
        df = df[df["deleted_at"].isnull()].copy()

    df = df.dropna(subset=["started_at", "finished_at", "room"]).copy()
    df["room"] = df["room"].astype(str).str.strip()
    df = df[df["room"].ne("") & df["room"].ne("nan")].copy()

    for col in ["started_at", "finished_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)

    df = df.dropna(subset=["started_at", "finished_at"]).sort_values("started_at").reset_index(drop=True)

    # Match the fresh benchmark's overlap correction exactly.
    for i in range(len(df) - 1):
        if df.loc[i, "finished_at"] >= df.loc[i + 1, "started_at"]:
            df.loc[i, "finished_at"] = df.loc[i + 1, "started_at"] - pd.Timedelta(seconds=1)

    df["duration_sec"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    df = df[df["duration_sec"] > 0].reset_index(drop=True)

    unexpected = sorted(set(df["room"].astype(str)) - set(OFFICIAL_CLASSES))
    if unexpected:
        raise RuntimeError(f"Unexpected room labels: {unexpected}")
    return df[df["room"].isin(OFFICIAL_CLASSES)].copy()


def load_training_ble(ble_dir):
    files = sorted(Path(ble_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No training BLE CSV files found in {ble_dir}")

    frames = []
    for f in files:
        frame = pd.read_csv(
            f,
            names=["user_id", "timestamp", "name", "mac address", "RSSI", "power"],
            usecols=[0, 1, 3, 4],
            on_bad_lines="skip",
            low_memory=False,
        )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["user_id"] == BLE_USER_ID].copy()
    df["beacon_id"] = df["mac address"].apply(parse_beacon_identifier)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    df["RSSI"] = pd.to_numeric(df["RSSI"], errors="coerce")
    df = df.dropna(subset=["timestamp", "beacon_id", "RSSI"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_test_ble(test_file):
    original = pd.read_csv(test_file)
    required = {"user_id", "timestamp", "mac address", "RSSI"}
    missing = required - set(original.columns)
    if missing:
        raise RuntimeError(f"Test file missing columns: {sorted(missing)}")

    work = original.copy()
    work["_row_order"] = np.arange(len(work), dtype=np.int64)
    work = work[work["user_id"] == BLE_USER_ID].copy()
    if len(work) != len(original):
        raise RuntimeError("Test file contains rows with user_id other than 90; refusing to silently drop them")

    work["beacon_id"] = work["mac address"].apply(parse_beacon_identifier)
    work["timestamp_parsed"] = pd.to_datetime(work["timestamp"], errors="coerce")
    if work["timestamp_parsed"].dt.tz is not None:
        work["timestamp_parsed"] = work["timestamp_parsed"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    work["RSSI_numeric"] = pd.to_numeric(work["RSSI"], errors="coerce")

    bad = work[["timestamp_parsed", "beacon_id", "RSSI_numeric"]].isna().any(axis=1)
    if bad.any():
        raise RuntimeError(f"Test file contains {int(bad.sum())} invalid timestamp/beacon/RSSI rows")

    work["timestamp_sec"] = work["timestamp_parsed"].dt.floor("s")
    return original, work


def build_second_state(df_ble, timestamp_col, rssi_col, beacon_col):
    temp = df_ble[[timestamp_col, beacon_col, rssi_col]].copy()
    temp["timestamp_sec"] = pd.to_datetime(temp[timestamp_col]).dt.floor("s")

    df_rssi = (
        temp.groupby(["timestamp_sec", beacon_col])[rssi_col]
        .mean()
        .unstack()
        .reindex(columns=BEACONS)
    )
    df_obs = (
        temp.groupby(["timestamp_sec", beacon_col])
        .size()
        .unstack()
        .reindex(columns=BEACONS)
    )
    df_rssi.index = pd.to_datetime(df_rssi.index)
    df_obs.index = pd.to_datetime(df_obs.index)

    parts = []
    for _, group_rssi in df_rssi.groupby(df_rssi.index.date, sort=True):
        group_obs = df_obs.loc[group_rssi.index]
        full_index = pd.date_range(group_rssi.index.min(), group_rssi.index.max(), freq="s")

        rss_ffill = group_rssi.reindex(full_index).ffill(limit=3)
        avail = (
            group_obs.reindex(full_index, fill_value=False)
            .astype(float)
            .replace(0, np.nan)
            .ffill(limit=3)
            .notna()
        )
        rss_smooth = rss_ffill.ewm(span=3, adjust=False).mean()
        rss_model = rss_smooth.where(avail, MODEL_MISSING_RSSI)

        combined = rss_model.copy()
        for beacon in BEACONS:
            combined[f"OBS_{beacon}"] = avail[beacon].astype(bool)
        parts.append(combined)

    out = pd.concat(parts).sort_index()
    if out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise RuntimeError("Continuous BLE timeline is not unique/monotonic")
    return out


def assign_training_labels(df_base, labels):
    out = df_base.copy()
    grid_ts = out.index.values
    assigned = np.array(["Transit"] * len(out), dtype=object)
    for _, row in labels.iterrows():
        mask = (
            (grid_ts >= row["started_at"].to_datetime64())
            & (grid_ts <= row["finished_at"].to_datetime64())
        )
        assigned[mask] = row["room"]
    out["assigned_room"] = assigned
    return out


# =============================================================================
# Feature engineering
# =============================================================================

def extract_all_features(df, window):
    X_parts, y_parts = [], []
    for _, grp in df.groupby(df.index.date, sort=True):
        sig = grp[BEACONS].replace(MODEL_MISSING_RSSI, np.nan)
        obs = grp[OBS_BEACONS].astype(float)

        roll_sig = sig.rolling(window=window, min_periods=1)
        roll_obs = obs.rolling(window=window, min_periods=1)

        f_mean = roll_sig.mean().fillna(MODEL_MISSING_RSSI)
        f_std = roll_sig.std().fillna(0.0)
        f_var = roll_sig.var().fillna(0.0)
        f_min = roll_sig.min().fillna(MODEL_MISSING_RSSI)
        f_max = roll_sig.max().fillna(MODEL_MISSING_RSSI)
        f_med = roll_sig.median().fillna(MODEL_MISSING_RSSI)
        f_sum = roll_sig.sum().fillna(MODEL_MISSING_RSSI)
        f_act = roll_obs.sum().fillna(0.0)
        f_diff = f_mean.diff().fillna(0.0)

        max_global = f_mean.max(axis=1)
        f_rel = f_mean.subtract(max_global, axis=0)
        f_global_active = f_act.sum(axis=1).rename("global_active")
        f_top1 = f_mean.max(axis=1).rename("top1_rssi")
        sorted_values = np.sort(f_mean.to_numpy(), axis=1)
        f_top2 = pd.Series(sorted_values[:, -2], index=f_mean.index, name="top2_rssi")

        f_mean.columns = [f"{b}_mean" for b in BEACONS]
        f_std.columns = [f"{b}_std" for b in BEACONS]
        f_var.columns = [f"{b}_var" for b in BEACONS]
        f_min.columns = [f"{b}_min" for b in BEACONS]
        f_max.columns = [f"{b}_max" for b in BEACONS]
        f_med.columns = [f"{b}_med" for b in BEACONS]
        f_sum.columns = [f"{b}_sum" for b in BEACONS]
        f_act.columns = [f"{b}_act" for b in BEACONS]
        f_diff.columns = [f"{b}_diff" for b in BEACONS]
        f_rel.columns = [f"{b}_rel" for b in BEACONS]

        X = pd.concat(
            [
                f_mean, f_std, f_var, f_min, f_max, f_med, f_sum, f_act,
                f_diff, f_rel, f_global_active, f_top1, f_top2,
            ],
            axis=1,
        )
        if not np.all(np.isfinite(X.to_numpy(dtype=float))):
            raise RuntimeError("Non-finite feature values detected")

        X_parts.append(X)
        y_parts.append(grp["assigned_room"].copy())

    return pd.concat(X_parts).sort_index(), pd.concat(y_parts).sort_index()


def get_feature_columns(feature_set):
    if feature_set == "basic":
        suffixes = ["mean", "std", "max", "act"]
    elif feature_set == "phase5_rich":
        suffixes = ["mean", "std", "max", "act", "diff", "rel"]
    elif feature_set == "all":
        suffixes = ["mean", "std", "var", "min", "max", "med", "sum", "act", "diff", "rel"]
    else:
        raise ValueError(feature_set)

    cols = []
    for suffix in suffixes:
        cols.extend([f"{b}_{suffix}" for b in BEACONS])
    if feature_set in {"phase5_rich", "all"}:
        cols.extend(["global_active", "top1_rssi", "top2_rssi"])
    return cols


def take_train_stride(X, y, stride):
    if stride == 1:
        return X, y
    X_parts, y_parts = [], []
    idx = pd.DatetimeIndex(X.index)
    for date in sorted(set(idx.date)):
        date_idx = idx[idx.date == date]
        selected = date_idx[::stride]
        X_parts.append(X.loc[selected])
        y_parts.append(y.loc[selected])
    return pd.concat(X_parts), pd.concat(y_parts)


# =============================================================================
# Fold-local relabeling / SMOTE (deployment uses all development days)
# =============================================================================

def contiguous_runs(df):
    if df.empty:
        return []
    z = df.sort_index()
    gap = z.index.to_series().diff().dt.total_seconds()
    run_id = gap.ne(1).cumsum()
    return [g.copy() for _, g in z.groupby(run_id) if len(g) > 0]


def has_usable_donor_run(df_train, room, window_size):
    room_df = df_train[df_train["assigned_room"].astype(str) == str(room)]
    return (not room_df.empty) and any(len(run) >= window_size for run in contiguous_runs(room_df))


def to_probability_profile(arr):
    arr = np.asarray(arr, dtype=float)
    strength = np.clip(arr - MODEL_MISSING_RSSI + 1e-5, 1e-5, None)
    return strength / strength.sum()


def partial_kl(target_profile, donor_profile, target_topology, donor_topology):
    valid = np.asarray(
        [(tb != "PROX_None") and (db != "PROX_None") for tb, db in zip(target_topology, donor_topology)],
        dtype=bool,
    )
    if valid.sum() < 2:
        return np.inf
    return float(
        entropy(
            to_probability_profile(target_profile[valid]),
            to_probability_profile(donor_profile[valid]),
        )
    )


def topology_fallback_score(target_room, donor_room):
    target = RELABEL_TOPOLOGY[target_room]
    donor = RELABEL_TOPOLOGY[donor_room]
    same_presence = sum((tb != "PROX_None") == (db != "PROX_None") for tb, db in zip(target, donor))
    usable_pairs = sum((tb != "PROX_None") and (db != "PROX_None") for tb, db in zip(target, donor))
    return same_presence, usable_pairs


def choose_symmetric_or_fallback(target, available_rooms, df_train, window_size):
    if target not in RELABEL_TOPOLOGY:
        return None
    preferred = SYMMETRIC_DONORS.get(target)
    if preferred in available_rooms and has_usable_donor_run(df_train, preferred, window_size):
        return preferred

    candidates = sorted(
        r for r in available_rooms
        if r in RELABEL_TOPOLOGY and r != target and has_usable_donor_run(df_train, r, window_size)
    )
    if not candidates:
        return None
    counts = df_train["assigned_room"].astype(str).value_counts()
    return sorted(
        candidates,
        key=lambda donor: (topology_fallback_score(target, donor), counts.get(donor, 0), donor),
        reverse=True,
    )[0]


def get_donor_mapping(method, df_train, y_train, minority_rooms, missing_rooms, window_size):
    mapping = {}
    available_rooms = set(df_train["assigned_room"].astype(str).unique())

    if method == "symmetric":
        for target in sorted(set(missing_rooms) | set(minority_rooms)):
            donor = choose_symmetric_or_fallback(target, available_rooms, df_train, window_size)
            if donor is not None:
                mapping[target] = donor
        return mapping

    if method != "kl_partial":
        raise ValueError(method)

    for target in missing_rooms:
        donor = choose_symmetric_or_fallback(target, available_rooms, df_train, window_size)
        if donor is not None:
            mapping[target] = donor

    room_profiles = {}
    for room in y_train.astype(str).unique():
        if room not in RELABEL_TOPOLOGY:
            continue
        room_data = df_train[df_train["assigned_room"].astype(str) == room]
        profile = []
        for beacon in RELABEL_TOPOLOGY[room]:
            if beacon == "PROX_None":
                profile.append(MODEL_MISSING_RSSI)
            else:
                observed = room_data[f"OBS_{beacon}"].astype(bool)
                values = room_data.loc[observed, beacon]
                profile.append(values.mean() if len(values) else MODEL_MISSING_RSSI)
        room_profiles[room] = np.asarray(profile, dtype=float)

    for target in minority_rooms:
        if target not in room_profiles:
            continue
        best_donor, best_kl = None, np.inf
        for donor, donor_profile in room_profiles.items():
            if (
                donor not in RELABEL_TOPOLOGY
                or donor in minority_rooms
                or donor == target
                or not has_usable_donor_run(df_train, donor, window_size)
            ):
                continue
            score = partial_kl(
                room_profiles[target], donor_profile,
                RELABEL_TOPOLOGY[target], RELABEL_TOPOLOGY[donor],
            )
            if score < best_kl:
                best_kl, best_donor = score, donor
        if best_donor is not None:
            mapping[target] = best_donor
    return mapping


def select_runs_with_budget(runs, budget, rng):
    runs = [r.copy() for r in runs if len(r) > 0]
    if not runs:
        return []
    order = rng.permutation(len(runs))
    selected, remaining = [], int(budget)
    for idx in order:
        if remaining <= 0:
            break
        run = runs[idx]
        if len(run) <= remaining:
            selected.append(run)
            remaining -= len(run)
        else:
            max_start = len(run) - remaining
            start = rng.randint(0, max_start + 1) if max_start > 0 else 0
            selected.append(run.iloc[start:start + remaining].copy())
            remaining = 0
    return selected


def generate_synthetic_blocks(df_fold_base, mapping, window_size, train_stride, required_columns, target_seconds):
    rng = np.random.RandomState(SEED)
    X_parts, y_parts = [], []

    for target_room, donor_room in mapping.items():
        donor_base = df_fold_base[df_fold_base["assigned_room"].astype(str) == str(donor_room)].copy()
        runs = [r for r in contiguous_runs(donor_base) if len(r) >= window_size]
        if not runs:
            continue

        selected_runs = select_runs_with_budget(runs, target_seconds, rng)
        target_beacons = RELABEL_TOPOLOGY.get(target_room, ["PROX_None"] * 6)
        donor_beacons = RELABEL_TOPOLOGY.get(donor_room, ["PROX_None"] * 6)

        for donor in selected_runs:
            for _ in range(3):
                synth = pd.DataFrame(MODEL_MISSING_RSSI, index=donor.index, columns=BEACONS)
                synth_obs = pd.DataFrame(False, index=donor.index, columns=OBS_BEACONS)
                synth["assigned_room"] = target_room

                for target_beacon, donor_beacon in zip(target_beacons, donor_beacons):
                    if target_beacon != "PROX_None" and donor_beacon != "PROX_None" and donor_beacon in donor.columns:
                        synth[target_beacon] = donor[donor_beacon].to_numpy()
                        synth_obs[f"OBS_{target_beacon}"] = donor[f"OBS_{donor_beacon}"].to_numpy()

                noise = rng.normal(0, 1.5, size=synth[BEACONS].shape)
                observed_mask = synth_obs[OBS_BEACONS].to_numpy(dtype=bool)
                synth.loc[:, BEACONS] = np.where(
                    observed_mask,
                    synth[BEACONS].to_numpy(dtype=float) + noise,
                    MODEL_MISSING_RSSI,
                )

                synth_combined = pd.concat([synth, synth_obs], axis=1)
                X_block, y_block = extract_all_features(synth_combined, window=window_size)
                X_block = X_block.loc[:, required_columns]

                if len(X_block) < window_size:
                    continue
                X_block = X_block.iloc[window_size - 1:].copy()
                y_block = y_block.iloc[window_size - 1:].copy()

                if train_stride > 1:
                    X_block = X_block.iloc[::train_stride].copy()
                    y_block = y_block.iloc[::train_stride].copy()

                X_parts.append(X_block)
                y_parts.append(y_block)

    if not X_parts:
        return None, None
    return pd.concat(X_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True)


def apply_smote_pipeline(X, y):
    y = pd.Series(np.asarray(y).astype(str))
    counts = y.value_counts()
    if len(counts) < 2:
        raise RuntimeError("SMOTE requires at least two training classes")

    rare_classes = counts[counts < 6].index
    if len(rare_classes) > 0:
        sampling_strategy = {c: max(6, int(counts[c])) for c in counts.index}
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=SEED)
        X_res, y_res = ros.fit_resample(X, y)
        X = pd.DataFrame(X_res, columns=X.columns)
        y = pd.Series(np.asarray(y_res).astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sampler = SMOTE(k_neighbors=4, random_state=SEED)
    X_res_scaled, y_res = sampler.fit_resample(X_scaled, y)
    X_res = scaler.inverse_transform(X_res_scaled)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(np.asarray(y_res).astype(str))


# =============================================================================
# Final branch fitting
# =============================================================================

def fit_final_branch(config_key, config, X_train_master, y_train_master, X_test_master, train_base):
    required_columns = get_feature_columns(config["feature_set"])
    X_train = X_train_master.loc[:, required_columns].copy()
    y_train = y_train_master.astype(str).copy()
    X_train, y_train = take_train_stride(X_train, y_train, config["train_stride"])

    valid = y_train.isin(OFFICIAL_CLASSES)
    X_train = X_train.loc[valid].copy()
    y_train = y_train.loc[valid].copy()
    if len(X_train) == 0:
        raise RuntimeError(f"{config_key}: no labeled real training rows")

    real_train_n = len(y_train)
    real_classes = sorted(set(y_train))
    missing_rooms = sorted(set(OFFICIAL_CLASSES) - set(real_classes))

    common_areas = {"cafeteria", "kitchen", "nurse station", "hallway", "cleaning"}
    sensor_only = y_train[~y_train.isin(common_areas)]
    sensor_counts = sensor_only.value_counts().sort_values()
    minority_rooms = (
        sensor_counts[sensor_counts <= sensor_counts.quantile(0.35)].index.tolist()
        if len(sensor_counts) else []
    )

    augmentation = config["augmentation"]
    donor_method = "symmetric" if "symmetric" in augmentation else "kl_partial"
    mapping = get_donor_mapping(
        donor_method,
        train_base,
        y_train,
        minority_rooms,
        missing_rooms,
        config["window"],
    )

    synthetic_n = 0
    if mapping:
        X_syn, y_syn = generate_synthetic_blocks(
            train_base,
            mapping,
            window_size=config["window"],
            train_stride=config["train_stride"],
            required_columns=required_columns,
            target_seconds=config["window"] * 15,
        )
        if X_syn is not None:
            synthetic_n = len(y_syn)
            X_train = pd.concat([X_train.reset_index(drop=True), X_syn.reset_index(drop=True)], ignore_index=True)
            y_train = pd.concat([y_train.reset_index(drop=True), y_syn.reset_index(drop=True)], ignore_index=True).astype(str)

    if "smote" in augmentation:
        X_train, y_train = apply_smote_pipeline(X_train, y_train)

    if not np.all(np.isfinite(X_train.to_numpy(dtype=float))):
        raise RuntimeError(f"{config_key}: non-finite post-augmentation training features")

    encoder = LabelEncoder()
    y_local = encoder.fit_transform(y_train.astype(str))

    if config["model"] == "rf":
        model = build_augmented_rf()
        model.fit(X_train, y_local)
    elif config["model"] == "xgb":
        model = build_xgb(len(encoder.classes_))
        sample_weight = compute_sample_weight("balanced", y_local)
        model.fit(X_train, y_local, sample_weight=sample_weight)
    else:
        raise ValueError(config["model"])

    if not np.array_equal(np.asarray(model.classes_), np.arange(len(encoder.classes_))):
        raise RuntimeError(f"{config_key}: classifier class ordering mismatch")

    X_test = X_test_master.loc[:, required_columns]
    local_probs = model.predict_proba(X_test)
    global_probs = align_probabilities(encoder.classes_, local_probs)

    diagnostics = {
        "branch": config_key,
        "window_s": config["window"],
        "train_stride": config["train_stride"],
        "feature_set": config["feature_set"],
        "augmentation": augmentation,
        "real_train_n": int(real_train_n),
        "real_train_classes": real_classes,
        "missing_real_classes": missing_rooms,
        "minority_classes": sorted(minority_rooms),
        "donor_mapping": dict(sorted(mapping.items())),
        "synthetic_n": int(synthetic_n),
        "post_aug_n": int(len(y_train)),
        "final_trained_classes": encoder.classes_.tolist(),
    }
    return global_probs, diagnostics


# =============================================================================
# Final decoder
# =============================================================================

def fused_probabilities(rich_probs, full_probs, rf_probs):
    rich_scaled = apply_temperature(rich_probs, ENSEMBLE_TEMPERATURES["rich_xgb"])
    full_scaled = apply_temperature(full_probs, ENSEMBLE_TEMPERATURES["full_xgb"])
    rf_scaled = apply_temperature(rf_probs, ENSEMBLE_TEMPERATURES["aug_rf"])
    fused = (
        ENSEMBLE_WEIGHTS["rich_xgb"] * rich_scaled
        + ENSEMBLE_WEIGHTS["full_xgb"] * full_scaled
        + ENSEMBLE_WEIGHTS["aug_rf"] * rf_scaled
    )
    ensure_probability_matrix(fused, len(fused), "Final fusion")
    return fused, full_scaled


def centered_majority_decode(raw_indices, timestamps, room518_multiplier=1.0):
    raw_indices = np.asarray(raw_indices, dtype=int)
    timestamps = pd.DatetimeIndex(timestamps)
    gap = pd.Series(timestamps).diff().dt.total_seconds().fillna(0)
    segment_id = gap.gt(1).cumsum().to_numpy()
    final = raw_indices.copy()

    for sid in np.unique(segment_id):
        positions = np.where(segment_id == sid)[0]
        onehot = np.eye(len(OFFICIAL_CLASSES), dtype=np.float64)[raw_indices[positions]]
        counts = (
            pd.DataFrame(onehot)
            .rolling(window=ENSEMBLE_ROLLING_WINDOW, center=True, min_periods=1)
            .sum()
            .to_numpy()
        )
        if room518_multiplier != 1.0:
            counts[:, IDX_518] *= room518_multiplier
        final[positions] = np.argmax(counts, axis=1)
    return final


def decode_final(rich_probs, full_probs, rf_probs, timestamps):
    fused, full_scaled = fused_probabilities(rich_probs, full_probs, rf_probs)
    adjusted = fused.copy()

    p510 = full_scaled[:, IDX_510]
    rank510 = 1 + np.sum(full_scaled > (p510[:, None] + 1e-12), axis=1)
    gate510 = (rank510 <= ROOM510_RANK_MAX) & (p510 > ROOM510_PROB_MIN)
    adjusted[gate510, IDX_510] *= ROOM510_MULTIPLIER

    raw = np.argmax(adjusted, axis=1)
    decoded = centered_majority_decode(
        raw,
        timestamps,
        room518_multiplier=ROOM518_VOTE_MULTIPLIER,
    )
    return OFFICIAL_ARRAY[decoded], gate510


# =============================================================================
# Main deployment
# =============================================================================

def main():
    args = parse_args()
    seed_everything()

    label_file = Path(args.label_file)
    ble_dir = Path(args.ble_dir)
    test_file = Path(args.test_file)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not label_file.is_file():
        raise FileNotFoundError(label_file)
    if not ble_dir.is_dir():
        raise FileNotFoundError(ble_dir)
    if not test_file.is_file():
        raise FileNotFoundError(test_file)

    print("=" * 90)
    print("PRAYAN - ISAS 2026 FINAL BLE SUBMISSION")
    print("=" * 90)
    print("Labels       :", label_file)
    print("Train BLE dir:", ble_dir)
    print("Test template:", test_file)
    print("Output       :", output_file)
    print("Prediction col:", args.prediction_column)
    print()

    # ---------- Load train ----------
    print("[1/8] Loading labels and raw training BLE...")
    labels = clean_labels(label_file)
    train_ble = load_training_ble(ble_dir)

    # Infer the same four labeled development days from labels ∩ BLE.
    label_days = set()
    for _, row in labels.iterrows():
        for ts in pd.date_range(row["started_at"].normalize(), row["finished_at"].normalize(), freq="D"):
            label_days.add(ts.date())
    ble_days = set(pd.DatetimeIndex(train_ble["timestamp"]).date)
    dev_dates = sorted(label_days & ble_days)
    if len(dev_dates) != 4:
        raise RuntimeError(f"Expected exactly four labeled development dates, found {dev_dates}")
    print("Development dates:", dev_dates)

    train_ble = train_ble[train_ble["timestamp"].dt.date.isin(dev_dates)].copy()

    print("[2/8] Reconstructing continuous one-second training state...")
    train_state = build_second_state(train_ble, "timestamp", "RSSI", "beacon_id")
    train_state = assign_training_labels(train_state, labels)

    # ---------- Load exact test template ----------
    print("[3/8] Loading organizer test template...")
    test_original, test_work = load_test_ble(test_file)

    test_dates = sorted(set(pd.DatetimeIndex(test_work["timestamp_parsed"]).date))
    observed_seconds = int(test_work["timestamp_sec"].nunique())
    print(f"Test rows       : {len(test_original):,}")
    print(f"Observed seconds: {observed_seconds:,}")
    print(f"Test dates      : {test_dates}")
    print(f"Beacon ids      : {sorted(test_work['mac address'].astype(int).unique().tolist())}")

    if args.strict_test_template:
        if len(test_original) != 62222:
            raise RuntimeError(f"Strict test check: expected 62,222 rows, got {len(test_original):,}")
        if test_dates != [pd.Timestamp("2023-04-14").date()]:
            raise RuntimeError(f"Strict test check: expected Apr14 only, got {test_dates}")
        if observed_seconds != 5721:
            raise RuntimeError(f"Strict test check: expected 5,721 observed seconds, got {observed_seconds:,}")

    print("[4/8] Reconstructing continuous one-second test state...")
    test_for_state = pd.DataFrame({
        "timestamp": test_work["timestamp_parsed"],
        "beacon_id": test_work["beacon_id"],
        "RSSI": test_work["RSSI_numeric"],
    })
    test_state = build_second_state(test_for_state, "timestamp", "RSSI", "beacon_id")
    test_state["assigned_room"] = "Transit"  # placeholder only; never used as a test label

    print(f"Continuous test grid: {len(test_state):,} seconds")
    if args.strict_test_template and len(test_state) != 25769:
        raise RuntimeError(f"Strict test check: expected 25,769 continuous seconds, got {len(test_state):,}")

    # ---------- Features ----------
    print("[5/8] Building W10/W60 train and test feature matrices...")
    X10_train, y10_train = extract_all_features(train_state, 10)
    X60_train, y60_train = extract_all_features(train_state, 60)
    X10_test, _ = extract_all_features(test_state, 10)
    X60_test, _ = extract_all_features(test_state, 60)

    if not X10_test.index.equals(test_state.index) or not X60_test.index.equals(test_state.index):
        raise RuntimeError("Test feature timeline changed unexpectedly")

    # ---------- Fit the three frozen branches ----------
    print("[6/8] Training the three frozen final-system branches on ALL development days...")
    probs = {}
    diagnostics = []

    for key in ["rich_xgb", "full_xgb", "aug_rf"]:
        cfg = FINAL_CONFIGS[key]
        print(f"  Training {key}: {cfg}")
        if cfg["window"] == 10:
            p, d = fit_final_branch(key, cfg, X10_train, y10_train, X10_test, train_state)
        else:
            p, d = fit_final_branch(key, cfg, X60_train, y60_train, X60_test, train_state)
        probs[key] = p
        diagnostics.append(d)
        print(
            f"    real={d['real_train_n']:,}, post_aug={d['post_aug_n']:,}, "
            f"synthetic={d['synthetic_n']:,}, trained_classes={len(d['final_trained_classes'])}"
        )
        print("    donor mapping:", d["donor_mapping"])

    # ---------- Decode continuous grid ----------
    print("[7/8] Applying final probability fusion and class-aware temporal decoder...")
    grid_pred, gate510 = decode_final(
        probs["rich_xgb"], probs["full_xgb"], probs["aug_rf"], test_state.index
    )

    if len(grid_pred) != len(test_state):
        raise RuntimeError("Grid prediction length mismatch")
    if not set(grid_pred).issubset(set(OFFICIAL_CLASSES)):
        raise RuntimeError("Prediction contains an invalid location class")

    # ---------- Map each continuous-second prediction back to every original row ----------
    print("[8/8] Mapping predictions back to the original 62,222-row test template...")
    second_to_pred = pd.Series(grid_pred, index=test_state.index, name=args.prediction_column)
    row_predictions = test_work["timestamp_sec"].map(second_to_pred)
    if row_predictions.isna().any():
        raise RuntimeError(f"{int(row_predictions.isna().sum())} raw test rows could not be mapped to a prediction")

    submission = test_original.copy()
    if args.prediction_column in submission.columns:
        raise RuntimeError(f"Prediction column already exists: {args.prediction_column}")
    submission[args.prediction_column] = row_predictions.to_numpy(dtype=str)

    # ---------- Final integrity checks ----------
    if len(submission) != len(test_original):
        raise RuntimeError("Submission row count changed")
    for col in test_original.columns:
        # String-safe exact preservation of every organizer-provided column value/order.
        a = test_original[col].astype(str).to_numpy()
        b = submission[col].astype(str).to_numpy()
        if not np.array_equal(a, b):
            raise RuntimeError(f"Original test column changed: {col}")
    if submission[args.prediction_column].isna().any():
        raise RuntimeError("Submission contains missing predictions")
    invalid = sorted(set(submission[args.prediction_column].astype(str)) - set(OFFICIAL_CLASSES))
    if invalid:
        raise RuntimeError(f"Submission contains invalid prediction classes: {invalid}")

    submission.to_csv(output_file, index=False)

    # Save a separate diagnostic file beside the submission; do NOT upload this diagnostic file.
    diag_file = output_file.with_name(output_file.stem + "_diagnostics.json")
    diagnostic_payload = {
        "team": "Prayan",
        "model": "Multi-Resolution RF-XGBoost Ensemble with Minority-Sensitive Temporal Decoding",
        "development_dates": [str(x) for x in dev_dates],
        "test_dates": [str(x) for x in test_dates],
        "test_rows": int(len(test_original)),
        "test_observed_seconds": observed_seconds,
        "test_continuous_seconds": int(len(test_state)),
        "prediction_column": args.prediction_column,
        "prediction_counts": submission[args.prediction_column].value_counts().sort_index().to_dict(),
        "room510_gate_seconds": int(gate510.sum()),
        "branch_diagnostics": diagnostics,
    }
    with open(diag_file, "w", encoding="utf-8") as f:
        json.dump(diagnostic_payload, f, indent=2)

    print()
    print("=" * 90)
    print("SUBMISSION CREATED SUCCESSFULLY")
    print("=" * 90)
    print("CSV             :", output_file)
    print("Rows            :", f"{len(submission):,}")
    print("Columns         :", submission.columns.tolist())
    print("Missing preds   :", int(submission[args.prediction_column].isna().sum()))
    print("Unique locations:", int(submission[args.prediction_column].nunique()))
    print("Prediction counts:")
    print(submission[args.prediction_column].value_counts().sort_index().to_string())
    print("Diagnostics     :", diag_file)
    print()
    print("IMPORTANT: Upload only the prediction CSV, not the diagnostics JSON.")


if __name__ == "__main__":
    main()
