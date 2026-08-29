#!/usr/bin/env python3
"""
ISAS 2026 Challenge - Prayan
Training and validation pipeline.

This script reproduces the development-data experiments reported in:
"Multi-Resolution BLE Indoor Localization for Nursing Care under
Signal Loss, Class Imbalance, and Temporal Shift"

Primary deployment-oriented evaluation:
Strict chronological Forward validation.

Final selected system:
Multi-resolution RF-XGBoost probability ensemble with
minority-sensitive temporal decoding.
"""

# CUBLAS determinism must be configured before importing torch.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import glob
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 0. CLI / FIXED CHALLENGE DEFINITIONS / FROZEN HYPERPARAMETERS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fresh cross-protocol benchmark for the ISAS BLE localization project."
    )
    p.add_argument("--label-file", default="/app/data/5f_label_loc_train.csv")
    p.add_argument("--ble-dir", default="/app/data/BLE Data")
    p.add_argument("--topology-file", default="/app/data/phase21_floorplan_topology_v1.csv")
    p.add_argument("--output-dir", default="/app/output")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument(
        "--protocols",
        default="row_random,block45_random,event_random,lodo,forward",
        help="Comma-separated protocol keys.",
    )
    p.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method keys, or 'all'.",
    )
    p.add_argument(
        "--skip-tcn",
        action="store_true",
        help="Engineering smoke-test only. Do not use for the final paper table.",
    )
    p.add_argument(
        "--skip-ble-hash",
        action="store_true",
        help="Skip SHA256 hashing of raw BLE CSV files (faster preflight).",
    )
    p.add_argument(
        "--prediction-random-seed",
        type=int,
        default=42,
        help="Random-split seed whose prediction rows are retained for later error analysis.",
    )
    return p.parse_args()


ARGS = parse_args()
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_FILE_PATH = Path(ARGS.label_file)
TRAIN_BLE_DIR = Path(ARGS.ble_dir)
TOPOLOGY_FILE = Path(ARGS.topology_file)

SEED = 42
RANDOM_SEEDS = [int(x.strip()) for x in ARGS.seeds.split(",") if x.strip()]
if len(RANDOM_SEEDS) == 0:
    raise ValueError("At least one random seed is required.")

REQUESTED_PROTOCOLS = [x.strip() for x in ARGS.protocols.split(",") if x.strip()]

MODEL_MISSING_RSSI = -120.0
PURGE_SEC = 59
BLOCK45_SEC = 45
TCN_MAX_EPOCHS = 40

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

# Development dates are inferred from the cleaned label intervals and BLE data
# later in the script. No calendar dates are used to fill evaluation results.
DEV_DATES = []
DEV_DATE_SET = set()

# Phase6B-2 fixed six-position room topology used by the relabeling engine.
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

# Frozen configurations selected during the original development search.
# These are exactly the six Phase6B-2 Pareto-shortlisted augmented backbones.
AUGMENTED_CONFIGS = {
    "xgb_rich60_kl_full_smote": {
        "model": "xgb", "window": 60, "train_stride": 1,
        "feature_set": "phase5_rich", "augmentation": "kl_full_smote",
    },
    "xgb_rich60_symmetric_smote": {
        "model": "xgb", "window": 60, "train_stride": 1,
        "feature_set": "phase5_rich", "augmentation": "symmetric_smote",
    },
    "xgb_full60_kl": {
        "model": "xgb", "window": 60, "train_stride": 10,
        "feature_set": "all", "augmentation": "kl_partial",
    },
    "rf_paper60_kl": {
        "model": "rf", "window": 60, "train_stride": 10,
        "feature_set": "paper", "augmentation": "kl_partial",
    },
    "lgbm_basic60_kl_full_smote": {
        "model": "lgbm", "window": 60, "train_stride": 5,
        "feature_set": "basic", "augmentation": "kl_full_smote",
    },
    "rf_basic10_symmetric_smote": {
        "model": "rf", "window": 10, "train_stride": 1,
        "feature_set": "basic", "augmentation": "symmetric_smote",
    },
}

METHOD_LABELS = {
    "rf_basic10": "Random Forest (10 s RSSI summary features)",
    "xgb_rich60_kl_full_smote": "XGBoost (60 s rich features + KL-full relabeling + SMOTE)",
    "xgb_rich60_symmetric_smote": "XGBoost (60 s rich features + symmetric relabeling + SMOTE)",
    "xgb_full60_kl": "XGBoost (60 s full-statistics + KL-partial relabeling)",
    "rf_paper60_kl": "Random Forest (60 s paper-statistics + KL-partial relabeling)",
    "lgbm_basic60_kl_full_smote": "LightGBM (60 s RSSI summary + KL-full relabeling + SMOTE)",
    "rf_basic10_symmetric_smote": "Random Forest (10 s RSSI summary + symmetric relabeling + SMOTE)",
    "rf_lagstack10": "Random Forest (10 s causal lag-stack features)",
    "tcn10": "Causal TCN (10 s RSSI + availability sequence)",
    "rf_topology10": "Topology-Augmented Random Forest (10 s RSSI summary + floor-plan graph features)",
    "ensemble_calibrated": "Calibrated Multi-Resolution RF-XGBoost Ensemble (11 s temporal voting)",
    "ensemble_room510": "Calibrated Multi-Resolution RF-XGBoost Ensemble + class-510 probability recovery",
    "ensemble_class_aware": "Calibrated Multi-Resolution RF-XGBoost Ensemble + class-510 recovery + class-518-weighted temporal voting",
}

METHOD_ORDER = list(METHOD_LABELS.keys())
if ARGS.methods.strip().lower() == "all":
    REQUESTED_METHODS = METHOD_ORDER.copy()
else:
    REQUESTED_METHODS = [x.strip() for x in ARGS.methods.split(",") if x.strip()]
    unknown_methods = sorted(set(REQUESTED_METHODS) - set(METHOD_LABELS))
    if unknown_methods:
        raise ValueError(f"Unknown method keys: {unknown_methods}")

if ARGS.skip_tcn and "tcn10" in REQUESTED_METHODS:
    REQUESTED_METHODS.remove("tcn10")

PROTOCOL_LABELS = {
    "row_random": "Naive stratified row-random 70/30",
    "block45_random": "45 s block-group random 70/30 + 59 s purge",
    "event_random": "Event-group random 70/30 + 59 s purge",
    "lodo": "Leave-one-day-out (Apr11-Apr13)",
    "forward": "Strict expanding-forward",
}
unknown_protocols = sorted(set(REQUESTED_PROTOCOLS) - set(PROTOCOL_LABELS))
if unknown_protocols:
    raise ValueError(f"Unknown protocol keys: {unknown_protocols}")

# Final selected fusion/calibration/post-processing parameters.
ENSEMBLE_TEMPERATURES = {
    "xgb_rich60_symmetric_smote": 1.25,
    "xgb_full60_kl": 1.00,
    "rf_basic10_symmetric_smote": 1.00,
}
ENSEMBLE_WEIGHTS = {
    "xgb_rich60_symmetric_smote": 0.15,
    "xgb_full60_kl": 0.15,
    "rf_basic10_symmetric_smote": 0.70,
}
ENSEMBLE_ROLLING_WINDOW = 11
ROOM510_RANK_MAX = 2
ROOM510_PROB_MIN = 0.15
ROOM510_MULTIPLIER = 1.5
ROOM518_VOTE_MULTIPLIER = 3.5


# =============================================================================
# 1. LOGGING / REPRODUCIBILITY / METADATA HELPERS
# =============================================================================

class DualLogger:
    def __init__(self, path: Path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")

    def write(self, message):
        if self.terminal and not self.terminal.closed:
            self.terminal.write(message)
        if self.log and not self.log.closed:
            self.log.write(message)
        self.flush()

    def flush(self):
        try:
            if self.terminal and not self.terminal.closed:
                self.terminal.flush()
            if self.log and not self.log.closed:
                self.log.flush()
        except Exception:
            pass


LOG_PATH = OUTPUT_DIR / f"phase24_fresh_benchmark_{RUN_ID}.txt"
LOGGER = DualLogger(LOG_PATH)
sys.stdout = LOGGER
sys.stderr = LOGGER


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


set_seed(SEED)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def fixed23_metrics(y_true, y_pred) -> Dict[str, float]:
    yy = np.asarray(y_true, dtype=str)
    pp = np.asarray(y_pred, dtype=str)
    if len(yy) != len(pp):
        raise RuntimeError("Metric input length mismatch.")
    return {
        "Macro_F1": float(f1_score(yy, pp, labels=OFFICIAL_CLASSES, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(yy, pp, labels=OFFICIAL_CLASSES, average="weighted", zero_division=0)),
        "Accuracy": float(accuracy_score(yy, pp)),
        "Eval_N": int(len(yy)),
        "True_Classes": int(len(set(yy))),
        "Pred_Classes": int(len(set(pp))),
    }


def sample_sd(values: Sequence[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def ensure_probability_matrix(probs, n_rows: int, label: str):
    p = np.asarray(probs, dtype=np.float64)
    if p.shape != (n_rows, len(OFFICIAL_CLASSES)):
        raise RuntimeError(f"{label}: expected probability shape {(n_rows, len(OFFICIAL_CLASSES))}, got {p.shape}")
    if not np.all(np.isfinite(p)):
        raise RuntimeError(f"{label}: non-finite probability values.")
    if np.any(p < -1e-12):
        raise RuntimeError(f"{label}: negative probability values.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise RuntimeError(f"{label}: probability rows do not sum to 1.")


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


def apply_temperature(probs, temperature: float):
    p = np.asarray(probs, dtype=np.float64)
    if temperature == 1.0:
        return p.copy()
    scaled = np.zeros_like(p)
    positive = p > 0
    scaled[positive] = np.power(p[positive], 1.0 / temperature)
    row_sum = scaled.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise RuntimeError("Temperature scaling generated a zero-sum row.")
    scaled /= row_sum
    return scaled


def build_basic_rf():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        max_features=10,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    )


def build_augmented_rf():
    # Exact Phase6B-2 / Phase20B augmented-RF configuration: max_features is left
    # at sklearn's selected/default behavior rather than forcing the Basic RF value.
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    )


def build_xgb(num_classes: int):
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


def build_lgbm(num_classes: int):
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=150,
        learning_rate=0.07,
        max_depth=8,
        num_leaves=64,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
        num_class=num_classes,
    )


# =============================================================================
# 2. INPUT PREFLIGHT / RAW DATA RECONSTRUCTION
# =============================================================================

print("=" * 110)
print("PHASE 24: FRESH CROSS-PROTOCOL BENCHMARK")
print("=" * 110)
print("Run ID       :", RUN_ID)
print("Protocols    :", REQUESTED_PROTOCOLS)
print("Methods      :", REQUESTED_METHODS)
print("Random seeds :", RANDOM_SEEDS)
print("PyTorch      :", torch.__version__)
print("Torch CUDA   :", torch.version.cuda)
print("CUDA device  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print()

if not LABEL_FILE_PATH.is_file():
    raise FileNotFoundError(f"Label file not found: {LABEL_FILE_PATH}")
if not TRAIN_BLE_DIR.is_dir():
    raise FileNotFoundError(f"BLE directory not found: {TRAIN_BLE_DIR}")
if not TOPOLOGY_FILE.is_file():
    raise FileNotFoundError(f"Topology file not found: {TOPOLOGY_FILE}")

BLE_FILES = sorted(Path(p) for p in glob.glob(str(TRAIN_BLE_DIR / "*.csv")))
if not BLE_FILES:
    raise FileNotFoundError(f"No BLE CSV files found under: {TRAIN_BLE_DIR}")

print("Input files found. Calculating input metadata...")
input_records = [
    {
        "Type": "labels",
        "Path": str(LABEL_FILE_PATH),
        "Bytes": LABEL_FILE_PATH.stat().st_size,
        "SHA256": file_sha256(LABEL_FILE_PATH),
    },
    {
        "Type": "topology",
        "Path": str(TOPOLOGY_FILE),
        "Bytes": TOPOLOGY_FILE.stat().st_size,
        "SHA256": file_sha256(TOPOLOGY_FILE),
    },
]
for f in BLE_FILES:
    input_records.append(
        {
            "Type": "ble",
            "Path": str(f),
            "Bytes": f.stat().st_size,
            "SHA256": "SKIPPED" if ARGS.skip_ble_hash else file_sha256(f),
        }
    )
pd.DataFrame(input_records).to_csv(OUTPUT_DIR / f"phase24_input_manifest_{RUN_ID}.csv", index=False)

print("Loading and cleaning location labels...")
df_labels_raw = pd.read_csv(LABEL_FILE_PATH)
required_label_columns = {"user_id", "activity", "started_at", "finished_at", "room"}
missing_label_columns = required_label_columns - set(df_labels_raw.columns)
if missing_label_columns:
    raise RuntimeError(f"Label file missing columns: {sorted(missing_label_columns)}")

df_labels = df_labels_raw[
    (df_labels_raw["user_id"] == LABEL_USER_ID)
    & (df_labels_raw["activity"] == "Location")
].copy()
if "deleted_at" in df_labels.columns:
    df_labels = df_labels[df_labels["deleted_at"].isnull()].copy()

df_labels = df_labels.dropna(subset=["started_at", "finished_at", "room"]).copy()
df_labels["room"] = df_labels["room"].astype(str).str.strip()
df_labels = df_labels[df_labels["room"].ne("") & df_labels["room"].ne("nan")].copy()

for col in ["started_at", "finished_at"]:
    df_labels[col] = pd.to_datetime(df_labels[col], errors="coerce")
    if df_labels[col].dt.tz is not None:
        df_labels[col] = df_labels[col].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)

df_labels = df_labels.dropna(subset=["started_at", "finished_at"]).sort_values("started_at").reset_index(drop=True)
for i in range(len(df_labels) - 1):
    if df_labels.loc[i, "finished_at"] >= df_labels.loc[i + 1, "started_at"]:
        df_labels.loc[i, "finished_at"] = df_labels.loc[i + 1, "started_at"] - pd.Timedelta(seconds=1)

df_labels["duration_sec"] = (
    df_labels["finished_at"] - df_labels["started_at"]
).dt.total_seconds()
df_labels = df_labels[df_labels["duration_sec"] > 0].reset_index(drop=True)

unexpected_rooms = sorted(set(df_labels["room"].astype(str)) - set(OFFICIAL_CLASSES))
if unexpected_rooms:
    raise RuntimeError(f"Unexpected room labels in location annotations: {unexpected_rooms}")
df_labels = df_labels[df_labels["room"].isin(OFFICIAL_CLASSES)].copy()

print("Loading raw BLE CSV files...")
ble_frames = []
for f in BLE_FILES:
    frame = pd.read_csv(
        f,
        names=["user_id", "timestamp", "name", "mac address", "RSSI", "power"],
        usecols=[0, 1, 3, 4],
        on_bad_lines="skip",
        low_memory=False,
    )
    ble_frames.append(frame)

df_ble_raw = pd.concat(ble_frames, ignore_index=True)
df_ble = df_ble_raw[df_ble_raw["user_id"] == BLE_USER_ID].copy()
df_ble["beacon_id"] = df_ble["mac address"].apply(parse_beacon_identifier)
df_ble["timestamp"] = pd.to_datetime(df_ble["timestamp"], errors="coerce")
if df_ble["timestamp"].dt.tz is not None:
    df_ble["timestamp"] = df_ble["timestamp"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
df_ble["RSSI"] = pd.to_numeric(df_ble["RSSI"], errors="coerce")
df_ble = df_ble.dropna(subset=["timestamp", "beacon_id", "RSSI"]).sort_values("timestamp").reset_index(drop=True)
df_ble["timestamp_sec"] = df_ble["timestamp"].dt.floor("s")

df_rssi = (
    df_ble.groupby(["timestamp_sec", "beacon_id"])["RSSI"]
    .mean()
    .unstack()
    .reindex(columns=BEACONS)
)
df_obs = (
    df_ble.groupby(["timestamp_sec", "beacon_id"])
    .size()
    .unstack()
    .reindex(columns=BEACONS)
)
df_rssi.index = pd.to_datetime(df_rssi.index)
df_obs.index = pd.to_datetime(df_obs.index)

# Infer the labeled development calendar from the actual inputs. Label intervals
# can theoretically cross midnight, so expand each interval to all calendar days
# it touches, then require corresponding BLE observations.
label_date_set = set()
for _, row in df_labels.iterrows():
    day_range = pd.date_range(
        pd.Timestamp(row["started_at"]).normalize(),
        pd.Timestamp(row["finished_at"]).normalize(),
        freq="D",
    )
    label_date_set.update(ts.date() for ts in day_range)
ble_date_set = set(pd.DatetimeIndex(df_rssi.index).date)
DEV_DATES = sorted(label_date_set & ble_date_set)
DEV_DATE_SET = set(DEV_DATES)
if len(DEV_DATES) != 4:
    raise RuntimeError(
        "Expected four labeled development dates from the intersection of cleaned "
        f"location annotations and BLE observations, found {len(DEV_DATES)}: {DEV_DATES}"
    )
print("Development dates inferred from inputs:", DEV_DATES)

print("Reconstructing the labeled development-day continuous 1-second BLE state...")
daily_base = []
seen_dev_dates = set()
for date, group_rssi in df_rssi.groupby(df_rssi.index.date):
    if date not in DEV_DATE_SET:
        continue
    seen_dev_dates.add(date)
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
    daily_base.append(combined)

if seen_dev_dates != DEV_DATE_SET:
    raise RuntimeError(
        f"Development-day mismatch. Expected {sorted(DEV_DATE_SET)}, found {sorted(seen_dev_dates)}"
    )

df_base = pd.concat(daily_base).sort_index()
if df_base.index.has_duplicates or not df_base.index.is_monotonic_increasing:
    raise RuntimeError("Continuous BLE timeline is not unique and monotonic.")

grid_ts = df_base.index.values
assigned = np.array(["Transit"] * len(df_base), dtype=object)
for _, row in df_labels.iterrows():
    mask = (
        (grid_ts >= row["started_at"].to_datetime64())
        & (grid_ts <= row["finished_at"].to_datetime64())
    )
    assigned[mask] = row["room"]
df_base["assigned_room"] = assigned


# =============================================================================
# 3. PHASE6B NATIVE FEATURE MATRICES
# =============================================================================

def extract_all_features(df: pd.DataFrame, window: int):
    X_parts, y_parts = [], []
    for _, grp in df.groupby(df.index.date, sort=True):
        if len(grp) == 0:
            continue
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
        if X.columns.duplicated().any():
            raise RuntimeError("Duplicate native feature columns detected.")
        if not np.all(np.isfinite(X.to_numpy(dtype=float))):
            raise RuntimeError("Non-finite native feature values detected.")

        X_parts.append(X)
        y_parts.append(grp["assigned_room"].copy())

    return pd.concat(X_parts).sort_index(), pd.concat(y_parts).sort_index()


def get_feature_columns(feature_set: str):
    if feature_set == "paper":
        suffixes = ["mean", "var", "std", "min", "max", "sum", "med", "act"]
    elif feature_set == "basic":
        suffixes = ["mean", "std", "max", "act"]
    elif feature_set == "phase5_rich":
        suffixes = ["mean", "std", "max", "act", "diff", "rel"]
    elif feature_set == "all":
        suffixes = ["mean", "std", "var", "min", "max", "med", "sum", "act", "diff", "rel"]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    cols = []
    for suffix in suffixes:
        cols.extend([f"{b}_{suffix}" for b in BEACONS])
    if feature_set in {"phase5_rich", "all"}:
        cols.extend(["global_active", "top1_rssi", "top2_rssi"])
    return cols


print("Building native W10 and W60 feature matrices...")
X_W10, y_W10 = extract_all_features(df_base.copy(), window=10)
X_W60, y_W60 = extract_all_features(df_base.copy(), window=60)


def take_train_stride(X: pd.DataFrame, y: pd.Series, stride: int):
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


# Full-day stride grids are needed when a random protocol retains only a subset
# of timestamps but must preserve the original model's day-anchored training stride.
def full_day_stride_set(index: pd.DatetimeIndex, stride: int):
    idx = pd.DatetimeIndex(index)
    out = []
    for date in sorted(set(idx.date)):
        d = idx[idx.date == date]
        out.extend(d[::stride].tolist())
    return set(pd.DatetimeIndex(out))


W60_STRIDE10_SET = full_day_stride_set(X_W60.index, 10)


# =============================================================================
# 4. PHASE19/20 HISTORY-INTEGRITY REPRESENTATIONS + EVENTS + 45-S BLOCKS
# =============================================================================

print("Building common 10-second-history evaluation population...")
history_parts = []
tensor_parts = []

for date, grp in df_base.groupby(df_base.index.date, sort=True):
    rss_model = grp[BEACONS].astype(float)
    obs_model = grp[OBS_BEACONS].astype(float)
    sig = rss_model.replace(MODEL_MISSING_RSSI, np.nan)

    f_mean = sig.rolling(10, min_periods=1).mean().fillna(MODEL_MISSING_RSSI)
    f_std = sig.rolling(10, min_periods=1).std().fillna(0.0)
    f_max = sig.rolling(10, min_periods=1).max().fillna(MODEL_MISSING_RSSI)
    f_act = obs_model.rolling(10, min_periods=1).sum().fillna(0.0)

    f_mean.columns = [f"{b}_mean" for b in BEACONS]
    f_std.columns = [f"{b}_std" for b in BEACONS]
    f_max.columns = [f"{b}_max" for b in BEACONS]
    f_act.columns = [f"{b}_act" for b in BEACONS]
    basic = pd.concat([f_mean, f_std, f_max, f_act], axis=1)

    raw_state = pd.concat(
        [
            rss_model.rename(columns={b: f"{b}_rssi" for b in BEACONS}),
            obs_model.rename(columns={b: f"{b}_avail" for b in BEACONS}),
        ],
        axis=1,
    )
    lag_parts = []
    for lag in range(10):
        shifted = raw_state.shift(lag)
        shifted.columns = [f"{c}_t-{lag}" for c in raw_state.columns]
        lag_parts.append(shifted)
    lagstack = pd.concat(lag_parts, axis=1)

    rss_norm = np.clip(
        (rss_model.to_numpy(dtype=np.float32) + 120.0) / 100.0,
        0.0,
        1.0,
    )
    state = np.concatenate([rss_norm, obs_model.to_numpy(dtype=np.float32)], axis=1)
    seq_view = np.lib.stride_tricks.sliding_window_view(state, window_shape=10, axis=0)
    seq = np.ascontiguousarray(seq_view.transpose(0, 2, 1))

    frame = pd.concat([basic, lagstack], axis=1).iloc[9:].copy()
    frame["timestamp"] = frame.index
    frame["day"] = f"apr{date.day}"
    frame["room"] = grp["assigned_room"].iloc[9:].to_numpy()

    if len(frame) != len(seq):
        raise RuntimeError(f"History/tensor alignment mismatch on {date}.")

    history_parts.append(frame)
    tensor_parts.append(seq)

hist = pd.concat(history_parts).sort_values(["timestamp"]).reset_index(drop=True)
tcn_tensor_all = np.concatenate(tensor_parts, axis=0)
if len(hist) != len(tcn_tensor_all):
    raise RuntimeError("History dataframe and TCN tensor have different lengths.")

valid_hist = hist["room"].isin(OFFICIAL_CLASSES).to_numpy()
hist = hist.loc[valid_hist].reset_index(drop=True)
tcn_tensor = tcn_tensor_all[valid_hist]

BASIC_COLS = [c for c in hist.columns if any(c.endswith(s) for s in ["_mean", "_std", "_max", "_act"])]
LAG_COLS = [c for c in hist.columns if "_t-" in c]
if len(BASIC_COLS) != 4 * len(BEACONS):
    raise RuntimeError(f"Basic feature dimension mismatch: {len(BASIC_COLS)}")
if len(LAG_COLS) != 10 * 2 * len(BEACONS):
    raise RuntimeError(f"Lag-stack feature dimension mismatch: {len(LAG_COLS)}")
if tcn_tensor.shape[1:] != (10, 2 * len(BEACONS)):
    raise RuntimeError(f"TCN tensor dimension mismatch: {tcn_tensor.shape}")
if not np.all(np.isfinite(hist[BASIC_COLS + LAG_COLS].to_numpy(dtype=float))):
    raise RuntimeError("Non-finite history features detected.")
if not np.all(np.isfinite(tcn_tensor)):
    raise RuntimeError("Non-finite TCN values detected.")

# Dynamic equivalence guard: Phase19 Basic and Phase6B W10 Basic must represent
# the same values at the common history timestamps.
native_basic = X_W10.loc[pd.DatetimeIndex(hist["timestamp"]), get_feature_columns("basic")]
if not np.allclose(native_basic.to_numpy(dtype=float), hist[BASIC_COLS].to_numpy(dtype=float), atol=1e-12, rtol=0.0):
    raise RuntimeError("W10 Basic representations disagree between native/history builders.")

# Robust event IDs after removing Transit, matching the Phase20A concept.
day_change = hist["day"].ne(hist["day"].shift())
time_gap = hist["timestamp"].diff().dt.total_seconds().ne(1)
room_change = hist["room"].ne(hist["room"].shift())
hist["event_id"] = (day_change | time_gap | room_change).cumsum().astype(int)

# Non-overlapping 45-second blocks contained entirely inside one labeled event.
hist["block45_id"] = -1
block_id = 0
for _, g in hist.groupby("event_id", sort=True):
    rows = g.index.to_numpy()
    n_blocks = len(rows) // BLOCK45_SEC
    for j in range(n_blocks):
        selected = rows[j * BLOCK45_SEC:(j + 1) * BLOCK45_SEC]
        if len(selected) != BLOCK45_SEC:
            continue
        t0 = hist.loc[selected[0], "timestamp"]
        t1 = hist.loc[selected[-1], "timestamp"]
        if t1 - t0 != pd.Timedelta(seconds=BLOCK45_SEC - 1):
            raise RuntimeError("A proposed 45-second block is not contiguous.")
        hist.loc[selected, "block45_id"] = block_id
        block_id += 1

META_TS = pd.DatetimeIndex(hist["timestamp"])
if META_TS.has_duplicates or not META_TS.is_monotonic_increasing:
    raise RuntimeError("Common evaluation timestamps are not unique and ordered.")
TS_TO_HIST_POS = {pd.Timestamp(t): i for i, t in enumerate(META_TS)}
TS_TO_LABEL = dict(zip(META_TS, hist["room"].astype(str)))

X_BASIC_HIST = hist[BASIC_COLS].copy()
X_LAG_HIST = hist[LAG_COLS].copy()


# =============================================================================
# 5. FLOOR-PLAN TOPOLOGY REPRESENTATION
# =============================================================================

def load_floorplan_graph(topology_file: Path):
    topo = pd.read_csv(topology_file)
    required = {"beacon", "immediate_neighbors", "verified"}
    missing = required - set(topo.columns)
    if missing:
        raise RuntimeError(f"Topology file missing columns: {sorted(missing)}")
    if len(topo) != len(BEACONS) or topo["beacon"].duplicated().any():
        raise RuntimeError("Topology must contain one unique row for each of the 25 beacons.")

    verified = topo["verified"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    if not verified.all():
        raise RuntimeError("Topology contains unverified rows.")

    adjacency = {}
    for _, row in topo.iterrows():
        raw = row["immediate_neighbors"]
        if pd.isna(raw) or str(raw).strip() == "":
            neighbors = []
        else:
            neighbors = [x.strip() for x in str(raw).split(",") if x.strip()]
        adjacency[str(row["beacon"])] = neighbors

    if set(adjacency) != set(BEACONS):
        raise RuntimeError("Topology beacon set does not match the 25 BLE beacons.")
    for a, neighbors in adjacency.items():
        for b in neighbors:
            if b not in adjacency:
                raise RuntimeError(f"Topology contains unknown neighbor: {a} -> {b}")
            if a == b:
                raise RuntimeError(f"Topology contains self-loop: {a}")
            if a not in adjacency[b]:
                raise RuntimeError(f"Topology edge is not symmetric: {a} <-> {b}")
    return adjacency, topo


def construct_spatial_representation(X_basic: pd.DataFrame, adjacency: Dict[str, List[str]]):
    neighbor_mean = pd.DataFrame(index=X_basic.index)
    for beacon in BEACONS:
        neighbors = adjacency[beacon]
        if not neighbors:
            neighbor_mean[f"{beacon}_nbr_mean"] = MODEL_MISSING_RSSI
        else:
            vals = X_basic[[f"{n}_mean" for n in neighbors]].where(
                lambda z: z > MODEL_MISSING_RSSI
            )
            neighbor_mean[f"{beacon}_nbr_mean"] = vals.mean(axis=1).fillna(MODEL_MISSING_RSSI)

    contrast = pd.DataFrame(index=X_basic.index)
    for beacon in BEACONS:
        own = X_basic[f"{beacon}_mean"]
        nbr = neighbor_mean[f"{beacon}_nbr_mean"]
        valid = (own > MODEL_MISSING_RSSI) & (nbr > MODEL_MISSING_RSSI)
        values = pd.Series(0.0, index=X_basic.index)
        values.loc[valid] = own.loc[valid] - nbr.loc[valid]
        contrast[f"{beacon}_contrast"] = values

    neighbor_activity = pd.DataFrame(index=X_basic.index)
    for beacon in BEACONS:
        neighbors = adjacency[beacon]
        if not neighbors:
            neighbor_activity[f"{beacon}_nbr_act"] = 0.0
        else:
            neighbor_activity[f"{beacon}_nbr_act"] = X_basic[
                [f"{n}_act" for n in neighbors]
            ].sum(axis=1)

    spatial = pd.concat([X_basic, neighbor_mean, contrast, neighbor_activity], axis=1)
    expected_dim = X_basic.shape[1] + 3 * len(BEACONS)
    if spatial.shape[1] != expected_dim:
        raise RuntimeError(f"Spatial feature dimension mismatch: {spatial.shape[1]} vs {expected_dim}")
    if not np.all(np.isfinite(spatial.to_numpy(dtype=float))):
        raise RuntimeError("Non-finite spatial features detected.")
    return spatial


ADJACENCY, TOPOLOGY_DF = load_floorplan_graph(TOPOLOGY_FILE)
X_SPATIAL_HIST = construct_spatial_representation(X_BASIC_HIST, ADJACENCY)


# =============================================================================
# 6. DATASET / METHOD / PROTOCOL REGISTRIES
# =============================================================================

# Class support and event counts are calculated from the fresh common population.
day_support = pd.crosstab(hist["room"], hist["day"]).reindex(OFFICIAL_CLASSES, fill_value=0)
day_support["Total_Seconds"] = day_support.sum(axis=1)
event_counts = hist.groupby("room")["event_id"].nunique().reindex(OFFICIAL_CLASSES, fill_value=0)
day_support["Total_Events"] = event_counts.astype(int)
day_support.reset_index().rename(columns={"room": "Class"}).to_csv(
    OUTPUT_DIR / f"phase24_class_support_{RUN_ID}.csv", index=False
)

method_registry = [
    {
        "MethodKey": "rf_basic10",
        "PaperName": METHOD_LABELS["rf_basic10"],
        "Representation": "10 s rolling mean/std/max/activity for 25 beacons",
        "Dimension": len(BASIC_COLS),
        "Training": "real data only",
        "Estimator": "RandomForestClassifier",
        "Hyperparameters": json.dumps({"n_estimators":100,"max_depth":12,"max_features":10,"class_weight":"balanced","random_state":42}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase19A / Phase20A",
    },
    {
        "MethodKey": "xgb_rich60_kl_full_smote",
        "PaperName": METHOD_LABELS["xgb_rich60_kl_full_smote"],
        "Representation": "60 s rich mean/std/max/activity/difference/relative/global features",
        "Dimension": len(get_feature_columns("phase5_rich")),
        "Training": "KL-full donor relabeling + SMOTE + balanced sample weights",
        "Estimator": "XGBClassifier",
        "Hyperparameters": json.dumps({"n_estimators":150,"learning_rate":0.07,"max_depth":7,"objective":"multi:softprob","random_state":42}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase6B-2 Pareto shortlist",
    },
    {
        "MethodKey": "rf_paper60_kl",
        "PaperName": METHOD_LABELS["rf_paper60_kl"],
        "Representation": "60 s paper-statistics: mean/var/std/min/max/sum/median/activity",
        "Dimension": len(get_feature_columns("paper")),
        "Training": "KL-partial donor relabeling",
        "Estimator": "RandomForestClassifier",
        "Hyperparameters": json.dumps({"n_estimators":100,"max_depth":12,"class_weight":"balanced","random_state":42}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase6B-2",
    },
    {
        "MethodKey": "xgb_full60_kl",
        "PaperName": METHOD_LABELS["xgb_full60_kl"],
        "Representation": "60 s full statistical + relative/difference/global features",
        "Dimension": len(get_feature_columns("all")),
        "Training": "KL-partial donor relabeling + balanced sample weights",
        "Estimator": "XGBClassifier",
        "Hyperparameters": json.dumps({"n_estimators":150,"learning_rate":0.07,"max_depth":7,"objective":"multi:softprob","random_state":42}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase6B-2",
    },
    {
        "MethodKey": "xgb_rich60_symmetric_smote",
        "PaperName": METHOD_LABELS["xgb_rich60_symmetric_smote"],
        "Representation": "60 s rich mean/std/max/activity/difference/relative/global features",
        "Dimension": len(get_feature_columns("phase5_rich")),
        "Training": "symmetric donor relabeling + SMOTE + balanced sample weights",
        "Estimator": "XGBClassifier",
        "Hyperparameters": json.dumps({"n_estimators":150,"learning_rate":0.07,"max_depth":7,"objective":"multi:softprob","random_state":42}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase6B-2",
    },
    {
        "MethodKey": "lgbm_basic60_kl_full_smote",
        "PaperName": METHOD_LABELS["lgbm_basic60_kl_full_smote"],
        "Representation": "60 s rolling mean/std/max/activity for 25 beacons",
        "Dimension": len(get_feature_columns("basic")),
        "Training": "KL-full donor relabeling + SMOTE",
        "Estimator": "LGBMClassifier",
        "Hyperparameters": json.dumps({"n_estimators":150,"learning_rate":0.07,"max_depth":8,"num_leaves":64,"class_weight":"balanced","random_state":42}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase6B-2 Pareto shortlist",
    },
    {
        "MethodKey": "rf_basic10_symmetric_smote",
        "PaperName": METHOD_LABELS["rf_basic10_symmetric_smote"],
        "Representation": "10 s rolling mean/std/max/activity for 25 beacons",
        "Dimension": len(get_feature_columns("basic")),
        "Training": "symmetric donor relabeling + SMOTE",
        "Estimator": "RandomForestClassifier",
        "Hyperparameters": json.dumps({"n_estimators":100,"max_depth":12,"class_weight":"balanced","random_state":42}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase6B-2",
    },
    {
        "MethodKey": "rf_lagstack10",
        "PaperName": METHOD_LABELS["rf_lagstack10"],
        "Representation": "10 x 50 causal RSSI/availability lag stack flattened to 500D",
        "Dimension": len(LAG_COLS),
        "Training": "real data only",
        "Estimator": "RandomForestClassifier",
        "Hyperparameters": json.dumps({"n_estimators":100,"max_depth":12,"max_features":10,"class_weight":"balanced","random_state":42}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase19A",
    },
    {
        "MethodKey": "tcn10",
        "PaperName": METHOD_LABELS["tcn10"],
        "Representation": "10 x 50 causal normalized RSSI + availability sequence",
        "Dimension": "10x50",
        "Training": "weighted cross-entropy; fold-local epoch selection",
        "Estimator": "Causal TCN",
        "Hyperparameters": json.dumps({"channels":64,"kernel_size":3,"dilations":[1,2,4],"dropout":0.10,"optimizer":"AdamW","lr":0.001,"weight_decay":0.0001,"max_epochs":40}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase19C",
    },
    {
        "MethodKey": "rf_topology10",
        "PaperName": METHOD_LABELS["rf_topology10"],
        "Representation": "Basic100 + neighbor mean + neighbor contrast + neighbor activity",
        "Dimension": X_SPATIAL_HIST.shape[1],
        "Training": "real data only",
        "Estimator": "RandomForestClassifier",
        "Hyperparameters": json.dumps({"n_estimators":100,"max_depth":12,"max_features":10,"class_weight":"balanced","random_state":42}),
        "Postprocessing": "none",
        "DevelopmentSource": "Phase21A",
    },
    {
        "MethodKey": "ensemble_calibrated",
        "PaperName": METHOD_LABELS["ensemble_calibrated"],
        "Representation": "Three selected native constituent representations",
        "Dimension": "153 + 253 + 100 constituent feature spaces",
        "Training": "fresh independent constituent fits per split",
        "Estimator": "Probability-level RF-XGBoost ensemble",
        "Hyperparameters": json.dumps({"temperatures":ENSEMBLE_TEMPERATURES,"weights":ENSEMBLE_WEIGHTS}),
        "Postprocessing": f"centered {ENSEMBLE_ROLLING_WINDOW} s majority voting",
        "DevelopmentSource": "Phase8B-1 / Phase8B-2",
    },
    {
        "MethodKey": "ensemble_room510",
        "PaperName": METHOD_LABELS["ensemble_room510"],
        "Representation": "Three selected native constituent representations",
        "Dimension": "153 + 253 + 100 constituent feature spaces",
        "Training": "fresh independent constituent fits per split",
        "Estimator": "Probability-level RF-XGBoost ensemble",
        "Hyperparameters": json.dumps({"temperatures":ENSEMBLE_TEMPERATURES,"weights":ENSEMBLE_WEIGHTS,"room510_rank_max":ROOM510_RANK_MAX,"room510_prob_min":ROOM510_PROB_MIN,"room510_multiplier":ROOM510_MULTIPLIER}),
        "Postprocessing": f"510 probability recovery + centered {ENSEMBLE_ROLLING_WINDOW} s majority voting",
        "DevelopmentSource": "Phase8B-2 / Phase11B-1",
    },
    {
        "MethodKey": "ensemble_class_aware",
        "PaperName": METHOD_LABELS["ensemble_class_aware"],
        "Representation": "Three selected native constituent representations",
        "Dimension": "153 + 253 + 100 constituent feature spaces",
        "Training": "fresh independent constituent fits per split",
        "Estimator": "Probability-level RF-XGBoost ensemble",
        "Hyperparameters": json.dumps({"temperatures":ENSEMBLE_TEMPERATURES,"weights":ENSEMBLE_WEIGHTS,"room510_rank_max":ROOM510_RANK_MAX,"room510_prob_min":ROOM510_PROB_MIN,"room510_multiplier":ROOM510_MULTIPLIER,"room518_vote_multiplier":ROOM518_VOTE_MULTIPLIER}),
        "Postprocessing": f"510 probability adjustment + centered {ENSEMBLE_ROLLING_WINDOW} s class-aware temporal voting",
        "DevelopmentSource": "Phase8B-2 / Phase11B-1 / Phase11B-2",
    },
]
pd.DataFrame(method_registry).to_csv(
    OUTPUT_DIR / f"phase24_method_registry_{RUN_ID}.csv", index=False
)

protocol_registry = [
    {
        "ProtocolKey":"row_random",
        "Protocol":PROTOCOL_LABELS["row_random"],
        "SplitUnit":"individual common-population seconds",
        "TrainTest":"70/30 stratified by class",
        "LeakageControl":"none by design at the feature/model split; diagnostic of optimistic same-distribution evaluation",
        "DecoderContext":"held-out test timestamps only; training-row predictions are not allowed to enter temporal decoding",
        "Aggregation":"mean and sample SD across requested seeds",
    },
    {
        "ProtocolKey":"block45_random",
        "Protocol":PROTOCOL_LABELS["block45_random"],
        "SplitUnit":"complete non-overlapping 45 s blocks inside labeled events",
        "TrainTest":"approximately 70/30 blocks within each class",
        "LeakageControl":"training rows within 59 s of any held-out row are purged",
        "DecoderContext":"held-out block timestamps only",
        "Aggregation":"mean and sample SD across requested seeds",
    },
    {
        "ProtocolKey":"event_random",
        "Protocol":PROTOCOL_LABELS["event_random"],
        "SplitUnit":"whole contiguous location events",
        "TrainTest":"approximately 70/30 events within each class; singleton-event classes remain training-only",
        "LeakageControl":"training rows within 59 s of any held-out row are purged",
        "DecoderContext":"held-out event timestamps only",
        "Aggregation":"mean and sample SD across requested seeds",
    },
    {
        "ProtocolKey":"lodo",
        "Protocol":PROTOCOL_LABELS["lodo"],
        "SplitUnit":"whole day",
        "TrainTest":"each development day after the first is held out; all other development days train",
        "LeakageControl":"held-out day absent from training and augmentation donors",
        "DecoderContext":"complete held-out day",
        "Aggregation":"pooled predictions across held-out development days after the first",
    },
    {
        "ProtocolKey":"forward",
        "Protocol":PROTOCOL_LABELS["forward"],
        "SplitUnit":"whole future day",
        "TrainTest":"strict earlier-day -> later-day expanding chain over inferred development dates",
        "LeakageControl":"only earlier dates can train a future-date fold",
        "DecoderContext":"complete held-out future day",
        "Aggregation":"pooled predictions across development days after the first",
    },
]
pd.DataFrame(protocol_registry).to_csv(
    OUTPUT_DIR / f"phase24_protocol_registry_{RUN_ID}.csv", index=False
)


# =============================================================================
# 7. SPLIT DEFINITIONS
# =============================================================================

@dataclass
class SplitSpec:
    protocol: str
    split_id: str
    seed: Optional[int]
    train_ts: pd.DatetimeIndex
    test_ts: pd.DatetimeIndex
    context_ts: pd.DatetimeIndex
    train_dates: Tuple
    val_date: Optional[object]


def timestamps_for_dates(dates: Sequence):
    date_set = set(dates)
    mask = np.isin(META_TS.date, list(date_set))
    return META_TS[mask]


def purge_training_timestamps(candidate_train_ts, test_ts, seconds=PURGE_SEC):
    candidate = pd.DatetimeIndex(candidate_train_ts)
    test = pd.DatetimeIndex(test_ts)
    kept = []
    for date in DEV_DATES:
        tr = candidate[candidate.date == date]
        te = test[test.date == date]
        if len(tr) == 0:
            continue
        if len(te) == 0:
            kept.append(tr)
            continue

        tr_s = tr.asi8 // 10**9
        te_s = np.sort(te.asi8 // 10**9)
        pos = np.searchsorted(te_s, tr_s)
        left = np.full(len(tr_s), np.inf)
        right = np.full(len(tr_s), np.inf)
        m = pos > 0
        left[m] = np.abs(tr_s[m] - te_s[pos[m] - 1])
        m = pos < len(te_s)
        right[m] = np.abs(tr_s[m] - te_s[pos[m]])
        keep = np.minimum(left, right) > seconds
        kept.append(tr[keep])

    if not kept:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate([x.values for x in kept]))


def assert_minimum_temporal_distance(train_ts, test_ts, seconds):
    train = pd.DatetimeIndex(train_ts)
    test = pd.DatetimeIndex(test_ts)
    for date in DEV_DATES:
        tr = train[train.date == date]
        te = test[test.date == date]
        if len(tr) == 0 or len(te) == 0:
            continue
        tr_s = tr.asi8 // 10**9
        te_s = np.sort(te.asi8 // 10**9)
        pos = np.searchsorted(te_s, tr_s)
        left = np.full(len(tr_s), np.inf)
        right = np.full(len(tr_s), np.inf)
        m = pos > 0
        left[m] = np.abs(tr_s[m] - te_s[pos[m] - 1])
        m = pos < len(te_s)
        right[m] = np.abs(tr_s[m] - te_s[pos[m]])
        if np.any(np.minimum(left, right) <= seconds):
            raise RuntimeError(f"Temporal purge violation on {date}.")


def choose_groups_within_class(
    group_df, group_col, seed, test_fraction=0.30, rng_kind="default_rng"
):
    # Phase20A event-group selection used NumPy's Generator/default_rng.
    # The project-defined 45 s block sensitivity protocol used legacy RandomState.
    if rng_kind == "default_rng":
        rng = np.random.default_rng(seed)
    elif rng_kind == "random_state":
        rng = np.random.RandomState(seed)
    else:
        raise ValueError(f"Unknown RNG kind: {rng_kind}")

    selected = []
    for room, g in group_df.groupby("room", sort=True):
        ids = g[group_col].to_numpy()
        n = len(ids)
        if n <= 1:
            continue
        n_test = int(round(test_fraction * n))
        n_test = max(1, min(n - 1, n_test))
        selected.extend(rng.choice(ids, size=n_test, replace=False).tolist())
    return set(selected)


def validate_split(spec: SplitSpec):
    train = pd.DatetimeIndex(spec.train_ts)
    test = pd.DatetimeIndex(spec.test_ts)
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError(f"Empty train/test population: {spec.protocol}/{spec.split_id}")
    if len(set(train) & set(test)):
        raise RuntimeError(f"Train/test timestamp overlap: {spec.protocol}/{spec.split_id}")

    if spec.protocol in {"event_random", "block45_random"}:
        assert_minimum_temporal_distance(train, test, PURGE_SEC)

    if spec.protocol == "forward":
        if spec.val_date is None or not spec.train_dates:
            raise RuntimeError("Malformed forward split.")
        if any(d >= spec.val_date for d in spec.train_dates):
            raise RuntimeError(f"Forward chronology violation: {spec.split_id}")

    if spec.protocol == "lodo":
        if spec.val_date in set(spec.train_dates):
            raise RuntimeError(f"LODO held-out day leaked into train dates: {spec.split_id}")


def build_splits(protocol: str):
    specs = []

    if protocol == "forward":
        for i in range(1, len(DEV_DATES)):
            train_dates = tuple(DEV_DATES[:i])
            val_date = DEV_DATES[i]
            test_ts = timestamps_for_dates([val_date])
            context_ts = pd.DatetimeIndex(df_base.index[df_base.index.date == val_date])
            specs.append(
                SplitSpec(
                    protocol=protocol,
                    split_id=f"apr{val_date.day}",
                    seed=None,
                    train_ts=timestamps_for_dates(train_dates),
                    test_ts=test_ts,
                    context_ts=context_ts,
                    train_dates=train_dates,
                    val_date=val_date,
                )
            )

    elif protocol == "lodo":
        # Apr10 is retained as the first training day so the evaluated population
        # stays identical to Forward (Apr11-Apr13).
        for val_date in DEV_DATES[1:]:
            train_dates = tuple(d for d in DEV_DATES if d != val_date)
            test_ts = timestamps_for_dates([val_date])
            context_ts = pd.DatetimeIndex(df_base.index[df_base.index.date == val_date])
            specs.append(
                SplitSpec(
                    protocol=protocol,
                    split_id=f"apr{val_date.day}",
                    seed=None,
                    train_ts=timestamps_for_dates(train_dates),
                    test_ts=test_ts,
                    context_ts=context_ts,
                    train_dates=train_dates,
                    val_date=val_date,
                )
            )

    elif protocol == "row_random":
        all_pos = np.arange(len(hist))
        labels = hist["room"].astype(str).to_numpy()
        for seed in RANDOM_SEEDS:
            train_pos, test_pos = train_test_split(
                all_pos,
                test_size=0.30,
                random_state=seed,
                stratify=labels,
            )
            train_ts = pd.DatetimeIndex(hist.loc[train_pos, "timestamp"])
            test_ts = pd.DatetimeIndex(hist.loc[test_pos, "timestamp"])
            specs.append(
                SplitSpec(
                    protocol=protocol,
                    split_id=f"seed{seed}",
                    seed=seed,
                    train_ts=train_ts,
                    test_ts=test_ts,
                    context_ts=pd.DatetimeIndex(sorted(test_ts)),
                    train_dates=tuple(),
                    val_date=None,
                )
            )

    elif protocol == "event_random":
        events = hist.groupby("event_id", as_index=False).agg(
            room=("room", "first"), n_seconds=("timestamp", "count")
        )
        for seed in RANDOM_SEEDS:
            test_events = choose_groups_within_class(
                events, "event_id", seed, 0.30, rng_kind="default_rng"
            )
            is_test = hist["event_id"].isin(test_events).to_numpy()
            test_ts = pd.DatetimeIndex(hist.loc[is_test, "timestamp"])
            train_ts = purge_training_timestamps(META_TS, test_ts, PURGE_SEC)
            specs.append(
                SplitSpec(
                    protocol=protocol,
                    split_id=f"seed{seed}",
                    seed=seed,
                    train_ts=train_ts,
                    test_ts=test_ts,
                    context_ts=pd.DatetimeIndex(sorted(test_ts)),
                    train_dates=tuple(),
                    val_date=None,
                )
            )

    elif protocol == "block45_random":
        eligible = hist[hist["block45_id"] >= 0].copy()
        if eligible.empty:
            raise RuntimeError("No complete 45-second labeled blocks are available.")
        blocks = eligible.groupby("block45_id", as_index=False).agg(
            room=("room", "first"), n_seconds=("timestamp", "count")
        )
        if not (blocks["n_seconds"] == BLOCK45_SEC).all():
            raise RuntimeError("45-second block table contains incomplete blocks.")

        for seed in RANDOM_SEEDS:
            test_blocks = choose_groups_within_class(
                blocks, "block45_id", seed, 0.30, rng_kind="random_state"
            )
            test_mask = eligible["block45_id"].isin(test_blocks).to_numpy()
            test_ts = pd.DatetimeIndex(eligible.loc[test_mask, "timestamp"])
            candidate_train = pd.DatetimeIndex(eligible.loc[~test_mask, "timestamp"])
            train_ts = purge_training_timestamps(candidate_train, test_ts, PURGE_SEC)
            specs.append(
                SplitSpec(
                    protocol=protocol,
                    split_id=f"seed{seed}",
                    seed=seed,
                    train_ts=train_ts,
                    test_ts=test_ts,
                    context_ts=pd.DatetimeIndex(sorted(test_ts)),
                    train_dates=tuple(),
                    val_date=None,
                )
            )
    else:
        raise ValueError(protocol)

    for spec in specs:
        validate_split(spec)
    return specs


# =============================================================================
# 8. PHASE6B-2 RELABELING / SMOTE ENGINE
# =============================================================================

def contiguous_runs(df: pd.DataFrame):
    if df.empty:
        return []
    z = df.sort_index()
    gap = z.index.to_series().diff().dt.total_seconds()
    run_id = gap.ne(1).cumsum()
    return [g.copy() for _, g in z.groupby(run_id) if len(g) > 0]


def has_usable_donor_run(df_train, room, window_size):
    room_df = df_train[df_train["assigned_room"].astype(str) == str(room)]
    if room_df.empty:
        return False
    return any(len(run) >= window_size for run in contiguous_runs(room_df))


def to_probability_profile(arr):
    arr = np.asarray(arr, dtype=float)
    strength = np.clip(arr - MODEL_MISSING_RSSI + 1e-5, 1e-5, None)
    return strength / strength.sum()


def partial_kl(target_profile, donor_profile, target_topology, donor_topology):
    valid = np.asarray(
        [
            (tb != "PROX_None") and (db != "PROX_None")
            for tb, db in zip(target_topology, donor_topology)
        ],
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
    same_presence = sum(
        (tb != "PROX_None") == (db != "PROX_None")
        for tb, db in zip(target, donor)
    )
    usable_pairs = sum(
        (tb != "PROX_None") and (db != "PROX_None")
        for tb, db in zip(target, donor)
    )
    return same_presence, usable_pairs


def choose_symmetric_or_fallback(target, available_rooms, df_train, window_size):
    if target not in RELABEL_TOPOLOGY:
        return None
    preferred = SYMMETRIC_DONORS.get(target)
    if (
        preferred in available_rooms
        and has_usable_donor_run(df_train, preferred, window_size)
    ):
        return preferred

    candidates = sorted(
        [
            r
            for r in available_rooms
            if r in RELABEL_TOPOLOGY
            and r != target
            and has_usable_donor_run(df_train, r, window_size)
        ]
    )
    if not candidates:
        return None

    counts = df_train["assigned_room"].astype(str).value_counts()
    ranked = sorted(
        candidates,
        key=lambda donor: (
            topology_fallback_score(target, donor),
            counts.get(donor, 0),
            donor,
        ),
        reverse=True,
    )
    return ranked[0]


def get_donor_mapping(method, df_train, y_train, minority_rooms, missing_rooms, window_size):
    mapping = {}
    available_rooms = set(df_train["assigned_room"].astype(str).unique())

    if method == "symmetric":
        for target in sorted(set(missing_rooms) | set(minority_rooms)):
            donor = choose_symmetric_or_fallback(
                target, available_rooms, df_train, window_size
            )
            if donor is not None:
                mapping[target] = donor
        return mapping

    if method not in {"kl_partial", "kl_full"}:
        raise ValueError(f"Unknown donor method: {method}")

    # Missing classes first receive the frozen symmetric/fallback treatment.
    for target in missing_rooms:
        donor = choose_symmetric_or_fallback(
            target, available_rooms, df_train, window_size
        )
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
                profile.append(
                    values.mean() if len(values) else MODEL_MISSING_RSSI
                )
        room_profiles[room] = np.asarray(profile, dtype=float)

    for target in minority_rooms:
        if target not in room_profiles:
            continue
        best_donor = None
        best_kl = np.inf
        for donor, donor_profile in room_profiles.items():
            if (
                donor not in RELABEL_TOPOLOGY
                or donor in minority_rooms
                or donor == target
                or not has_usable_donor_run(df_train, donor, window_size)
            ):
                continue
            if method == "kl_full":
                # Exact Phase6B-2 rule: full KL donors must have all six
                # topology positions defined. The target profile is still the
                # six-position profile used in the original search.
                if "PROX_None" in RELABEL_TOPOLOGY.get(donor, []):
                    continue
                score = float(
                    entropy(
                        to_probability_profile(room_profiles[target]),
                        to_probability_profile(donor_profile),
                    )
                )
            else:
                score = partial_kl(
                    room_profiles[target],
                    donor_profile,
                    RELABEL_TOPOLOGY[target],
                    RELABEL_TOPOLOGY[donor],
                )
            if score < best_kl:
                best_kl = score
                best_donor = donor
        if best_donor is not None:
            mapping[target] = best_donor

    return mapping


def select_runs_with_budget(runs, budget, rng):
    runs = [r.copy() for r in runs if len(r) > 0]
    if not runs:
        return []
    order = rng.permutation(len(runs))
    selected = []
    remaining = int(budget)

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


def generate_synthetic_blocks(
    df_fold_base,
    mapping,
    window_size,
    train_stride,
    required_columns,
    target_seconds,
):
    rng = np.random.RandomState(SEED)
    X_parts, y_parts = [], []

    for target_room, donor_room in mapping.items():
        donor_base = df_fold_base[
            df_fold_base["assigned_room"].astype(str) == str(donor_room)
        ].copy()
        runs = [r for r in contiguous_runs(donor_base) if len(r) >= window_size]
        if not runs:
            continue

        selected_runs = select_runs_with_budget(runs, target_seconds, rng)
        target_beacons = RELABEL_TOPOLOGY.get(target_room, ["PROX_None"] * 6)
        donor_beacons = RELABEL_TOPOLOGY.get(donor_room, ["PROX_None"] * 6)

        for donor in selected_runs:
            for _ in range(3):
                synth = pd.DataFrame(
                    MODEL_MISSING_RSSI, index=donor.index, columns=BEACONS
                )
                synth_obs = pd.DataFrame(
                    False, index=donor.index, columns=OBS_BEACONS
                )
                synth["assigned_room"] = target_room

                for target_beacon, donor_beacon in zip(target_beacons, donor_beacons):
                    if (
                        target_beacon != "PROX_None"
                        and donor_beacon != "PROX_None"
                        and donor_beacon in donor.columns
                    ):
                        synth[target_beacon] = donor[donor_beacon].to_numpy()
                        synth_obs[f"OBS_{target_beacon}"] = donor[
                            f"OBS_{donor_beacon}"
                        ].to_numpy()

                noise = rng.normal(0, 1.5, size=synth[BEACONS].shape)
                observed_mask = synth_obs[OBS_BEACONS].to_numpy(dtype=bool)
                synth.loc[:, BEACONS] = np.where(
                    observed_mask,
                    synth[BEACONS].to_numpy(dtype=float) + noise,
                    MODEL_MISSING_RSSI,
                )

                synth_combined = pd.concat([synth, synth_obs], axis=1)
                X_block, y_block = extract_all_features(
                    synth_combined, window=window_size
                )
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
    return (
        pd.concat(X_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
    )


def apply_smote_pipeline(X: pd.DataFrame, y: pd.Series):
    y = pd.Series(np.asarray(y).astype(str), name=getattr(y, "name", None))
    counts = y.value_counts()
    if len(counts) < 2:
        raise RuntimeError("SMOTE requires at least two training classes.")

    rare_classes = counts[counts < 6].index
    if len(rare_classes) > 0:
        sampling_strategy = {c: max(6, int(counts[c])) for c in counts.index}
        ros = RandomOverSampler(
            sampling_strategy=sampling_strategy, random_state=SEED
        )
        X, y = ros.fit_resample(X, y)
        X = pd.DataFrame(X, columns=X.columns)
        y = pd.Series(np.asarray(y).astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sampler = SMOTE(k_neighbors=4, random_state=SEED)
    X_res_scaled, y_res = sampler.fit_resample(X_scaled, y)
    X_res = scaler.inverse_transform(X_res_scaled)
    return (
        pd.DataFrame(X_res, columns=X.columns),
        pd.Series(np.asarray(y_res).astype(str)),
    )


def random_protocol_fold_base(train_ts):
    """
    Create a fold-local donor base for random protocols.

    Sensor observations remain present, but every labeled second that is not an
    allowed outer-training timestamp is converted to Transit. Therefore a held-
    out label can never be selected as an augmentation donor.
    """
    allowed = set(pd.DatetimeIndex(train_ts))
    out = df_base.copy()
    official = out["assigned_room"].isin(OFFICIAL_CLASSES)
    allowed_mask = pd.Series(out.index.isin(allowed), index=out.index)
    out.loc[official & ~allowed_mask, "assigned_room"] = "Transit"

    remaining_official = set(out.index[out["assigned_room"].isin(OFFICIAL_CLASSES)])
    if not remaining_official.issubset(allowed):
        raise RuntimeError("Random-protocol donor base contains held-out labels.")
    return out


@dataclass
class BranchResult:
    timestamps: pd.DatetimeIndex
    probabilities: np.ndarray
    diagnostics: Dict


def fit_augmented_branch(
    config_key: str,
    split: SplitSpec,
    predict_ts: pd.DatetimeIndex,
):
    cfg = AUGMENTED_CONFIGS[config_key]
    window = cfg["window"]
    train_stride = cfg["train_stride"]
    feature_set = cfg["feature_set"]
    augmentation = cfg["augmentation"]
    model_kind = cfg["model"]

    X_master, y_master = (X_W10, y_W10) if window == 10 else (X_W60, y_W60)
    required_columns = get_feature_columns(feature_set)
    X_full = X_master.loc[:, required_columns]

    if split.protocol in {"forward", "lodo"}:
        # Exact original native population: all feature rows from the selected
        # training days, then apply the architecture's day-local stride.
        train_mask = np.isin(X_full.index.date, split.train_dates)
        X_train = X_full.loc[train_mask].copy()
        y_train = y_master.loc[train_mask].astype(str).copy()
        X_train, y_train = take_train_stride(X_train, y_train, train_stride)

        donor_base = df_base[np.isin(df_base.index.date, split.train_dates)].copy()
        if split.val_date in set(donor_base.index.date):
            raise RuntimeError("Held-out day leaked into chronological donor base.")
    else:
        allowed_train = set(pd.DatetimeIndex(split.train_ts))
        selected_index = [t for t in X_full.index if t in allowed_train]
        if train_stride > 1:
            if window != 60 or train_stride != 10:
                # General day-anchored stride handling for future extensions.
                stride_allowed = full_day_stride_set(X_full.index, train_stride)
            else:
                stride_allowed = W60_STRIDE10_SET
            selected_index = [t for t in selected_index if t in stride_allowed]
        selected_index = pd.DatetimeIndex(selected_index)
        X_train = X_full.loc[selected_index].copy()
        y_train = y_master.loc[selected_index].astype(str).copy()
        donor_base = random_protocol_fold_base(split.train_ts)

    valid = y_train.isin(OFFICIAL_CLASSES)
    X_train = X_train.loc[valid].copy()
    y_train = y_train.loc[valid].astype(str).copy()
    if len(X_train) == 0:
        raise RuntimeError(f"{config_key}: no real training rows.")

    real_train_n = len(y_train)
    real_classes = sorted(set(y_train))
    missing_rooms = sorted(set(OFFICIAL_CLASSES) - set(real_classes))

    common_areas = {"cafeteria", "kitchen", "nurse station", "hallway", "cleaning"}
    sensor_only = y_train[~y_train.isin(common_areas)]
    sensor_counts = sensor_only.value_counts().sort_values()
    minority_rooms = (
        sensor_counts[
            sensor_counts <= sensor_counts.quantile(0.35)
        ].index.tolist()
        if len(sensor_counts)
        else []
    )

    if "symmetric" in augmentation:
        donor_method = "symmetric"
    elif "kl_full" in augmentation:
        donor_method = "kl_full"
    elif "kl_partial" in augmentation:
        donor_method = "kl_partial"
    else:
        raise RuntimeError(f"Unsupported relabeling mode in {config_key}: {augmentation}")
    mapping = get_donor_mapping(
        donor_method,
        donor_base,
        y_train,
        minority_rooms,
        missing_rooms,
        window,
    )

    synthetic_n = 0
    if mapping:
        X_syn, y_syn = generate_synthetic_blocks(
            donor_base,
            mapping,
            window_size=window,
            train_stride=train_stride,
            required_columns=required_columns,
            target_seconds=window * 15,
        )
        if X_syn is not None:
            synthetic_n = len(y_syn)
            X_train = pd.concat(
                [X_train.reset_index(drop=True), X_syn.reset_index(drop=True)],
                ignore_index=True,
            )
            y_train = pd.concat(
                [y_train.reset_index(drop=True), y_syn.reset_index(drop=True)],
                ignore_index=True,
            ).astype(str)

    if "smote" in augmentation:
        X_train, y_train = apply_smote_pipeline(X_train, y_train)

    if not np.all(np.isfinite(X_train.to_numpy(dtype=float))):
        raise RuntimeError(f"{config_key}: non-finite training features after augmentation.")

    encoder = LabelEncoder()
    y_local = encoder.fit_transform(y_train.astype(str))

    if model_kind == "rf":
        model = build_augmented_rf()
        model.fit(X_train, y_local)
    elif model_kind == "xgb":
        model = build_xgb(len(encoder.classes_))
        sample_weight = compute_sample_weight("balanced", y_local)
        model.fit(X_train, y_local, sample_weight=sample_weight)
    elif model_kind == "lgbm":
        model = build_lgbm(len(encoder.classes_))
        model.fit(X_train, y_local)
    else:
        raise ValueError(model_kind)

    expected_local = np.arange(len(encoder.classes_))
    if not np.array_equal(np.asarray(model.classes_), expected_local):
        raise RuntimeError(f"{config_key}: classifier class ordering mismatch.")

    predict_ts = pd.DatetimeIndex(predict_ts)
    missing_prediction_ts = predict_ts.difference(X_full.index)
    if len(missing_prediction_ts):
        raise RuntimeError(
            f"{config_key}: {len(missing_prediction_ts)} requested prediction timestamps are absent from its feature matrix."
        )
    X_predict = X_full.loc[predict_ts]
    local_probs = model.predict_proba(X_predict)
    global_probs = align_probabilities(encoder.classes_, local_probs)
    ensure_probability_matrix(global_probs, len(predict_ts), config_key)

    diagnostics = {
        "MethodKey": config_key,
        "ProtocolKey": split.protocol,
        "Split": split.split_id,
        "Seed": split.seed,
        "Window_s": window,
        "Train_Stride": train_stride,
        "Feature_Set": feature_set,
        "Augmentation": augmentation,
        "Real_Train_N": real_train_n,
        "Real_Train_Classes": len(real_classes),
        "Real_Train_Class_List": "|".join(real_classes),
        "Missing_Real_Classes": len(missing_rooms),
        "Missing_Real_Class_List": "|".join(missing_rooms),
        "Minority_Class_List": "|".join(sorted(minority_rooms)),
        "Donor_Mapping": json.dumps(dict(sorted(mapping.items()))),
        "Synthetic_N": synthetic_n,
        "Post_Aug_N": len(y_train),
        "Final_Trained_Classes": len(encoder.classes_),
    }
    return BranchResult(predict_ts, global_probs, diagnostics)


# =============================================================================
# 9. TCN MODEL AND FOLD-LOCAL EPOCH SELECTION
# =============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, 3, dilation)
        self.dropout = nn.Dropout(0.10)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        residual = self.residual(x)
        z = F.relu(self.conv(x))
        z = self.dropout(z)
        return F.relu(z + residual)


class W10CausalTCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = CausalBlock(50, 64, dilation=1)
        self.block2 = CausalBlock(64, 64, dilation=2)
        self.block3 = CausalBlock(64, 64, dilation=4)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        return self.classifier(z[:, :, -1])


def hist_positions_for_timestamps(timestamps):
    positions = []
    for ts in pd.DatetimeIndex(timestamps):
        key = pd.Timestamp(ts)
        if key not in TS_TO_HIST_POS:
            raise RuntimeError(f"Timestamp absent from common history population: {key}")
        positions.append(TS_TO_HIST_POS[key])
    return np.asarray(positions, dtype=int)


def choose_inner_groups(
    sub, group_col, seed, fraction=0.20, rng_kind="default_rng"
):
    groups = sub.groupby(group_col, as_index=False).agg(room=("room", "first"))
    selected_groups = choose_groups_within_class(
        groups, group_col, seed, fraction, rng_kind=rng_kind
    )
    return set(sub.loc[sub[group_col].isin(selected_groups), "orig_pos"].astype(int))


def inner_split_for_tcn(split: SplitSpec):
    outer_pos = hist_positions_for_timestamps(split.train_ts)
    sub = hist.loc[outer_pos, ["timestamp", "room", "event_id", "block45_id"]].copy()
    sub["orig_pos"] = outer_pos

    if split.protocol in {"forward", "lodo"}:
        latest_date = max(pd.DatetimeIndex(sub["timestamp"]).date)
        latest = sub[pd.DatetimeIndex(sub["timestamp"]).date == latest_date].sort_values("timestamp")
        cut = int(len(latest) * 0.80)
        if cut <= 0 or cut >= len(latest):
            raise RuntimeError(f"Invalid chronological TCN inner split: {split.protocol}/{split.split_id}")
        val_pos = set(latest.iloc[cut:]["orig_pos"].astype(int))
        train_pos = set(outer_pos.tolist()) - val_pos

    elif split.protocol == "row_random":
        rng = np.random.RandomState((split.seed if split.seed is not None else SEED) + 10000)
        val_pos = set()
        for _, g in sub.groupby("room", sort=True):
            ids = g["orig_pos"].to_numpy()
            if len(ids) <= 1:
                continue
            n_val = int(round(0.20 * len(ids)))
            n_val = max(1, min(len(ids) - 1, n_val))
            val_pos.update(rng.choice(ids, size=n_val, replace=False).tolist())
        train_pos = set(outer_pos.tolist()) - val_pos

    elif split.protocol == "event_random":
        inner_seed = (split.seed if split.seed is not None else SEED) + 10000
        val_pos = choose_inner_groups(
            sub, "event_id", inner_seed, 0.20, rng_kind="default_rng"
        )
        raw_train = pd.DatetimeIndex(
            hist.loc[sorted(set(outer_pos.tolist()) - val_pos), "timestamp"]
        )
        val_ts = pd.DatetimeIndex(hist.loc[sorted(val_pos), "timestamp"])
        # TCN history is 10 s, so a 9 s inner purge is sufficient to prevent
        # input-window overlap during epoch selection.
        purged = purge_training_timestamps(raw_train, val_ts, seconds=9)
        train_pos = set(hist_positions_for_timestamps(purged).tolist())

    elif split.protocol == "block45_random":
        eligible = sub[sub["block45_id"] >= 0].copy()
        if eligible.empty:
            raise RuntimeError("No 45-second blocks available for TCN inner split.")
        inner_seed = (split.seed if split.seed is not None else SEED) + 10000
        val_pos = choose_inner_groups(
            eligible, "block45_id", inner_seed, 0.20, rng_kind="random_state"
        )
        raw_train = pd.DatetimeIndex(
            hist.loc[sorted(set(outer_pos.tolist()) - val_pos), "timestamp"]
        )
        val_ts = pd.DatetimeIndex(hist.loc[sorted(val_pos), "timestamp"])
        purged = purge_training_timestamps(raw_train, val_ts, seconds=9)
        train_pos = set(hist_positions_for_timestamps(purged).tolist())

    else:
        raise ValueError(split.protocol)

    train_pos = np.asarray(sorted(train_pos), dtype=int)
    val_pos = np.asarray(sorted(val_pos), dtype=int)
    if len(train_pos) == 0 or len(val_pos) == 0:
        raise RuntimeError(f"Empty TCN inner split: {split.protocol}/{split.split_id}")
    if len(set(train_pos) & set(val_pos)):
        raise RuntimeError("TCN inner train/validation overlap.")
    return train_pos, val_pos


def class_weights_tensor(labels, classes, device):
    mapping = {c: i for i, c in enumerate(classes)}
    encoded = np.asarray([mapping[c] for c in labels], dtype=int)
    counts = np.bincount(encoded, minlength=len(classes))
    if np.any(counts == 0):
        raise RuntimeError("Unexpected zero-count class in TCN class list.")
    weights = len(encoded) / (len(classes) * counts)
    return encoded, torch.tensor(weights, dtype=torch.float32, device=device)


def select_tcn_epoch(split: SplitSpec, device):
    train_pos, val_pos = inner_split_for_tcn(split)
    train_labels = hist.loc[train_pos, "room"].astype(str).to_numpy()
    classes = [c for c in OFFICIAL_CLASSES if c in set(train_labels)]
    encoded, class_weights = class_weights_tensor(train_labels, classes, device)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(tcn_tensor[train_pos], dtype=torch.float32),
            torch.tensor(encoded, dtype=torch.long),
        ),
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(tcn_tensor[val_pos], dtype=torch.float32)),
        batch_size=512,
        shuffle=False,
    )
    val_true = hist.loc[val_pos, "room"].astype(str).to_numpy()
    class_array = np.asarray(classes, dtype=str)

    set_seed(SEED)
    model = W10CausalTCN(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_macro = -1.0
    best_epoch = 1
    for epoch in range(1, TCN_MAX_EPOCHS + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        pred_parts = []
        model.eval()
        with torch.no_grad():
            for (bx,) in val_loader:
                local_pred = torch.argmax(model(bx.to(device)), dim=1).cpu().numpy()
                pred_parts.append(class_array[local_pred])
        val_pred = np.concatenate(pred_parts)
        macro = f1_score(
            val_true,
            val_pred,
            labels=OFFICIAL_CLASSES,
            average="macro",
            zero_division=0,
        )
        if macro > best_macro:
            best_macro = macro
            best_epoch = epoch

    del model, optimizer, criterion, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_epoch


def fit_predict_tcn(split: SplitSpec, device):
    selected_epoch = select_tcn_epoch(split, device)
    train_pos = hist_positions_for_timestamps(split.train_ts)
    test_pos = hist_positions_for_timestamps(split.test_ts)
    train_labels = hist.loc[train_pos, "room"].astype(str).to_numpy()
    classes = [c for c in OFFICIAL_CLASSES if c in set(train_labels)]
    encoded, class_weights = class_weights_tensor(train_labels, classes, device)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(tcn_tensor[train_pos], dtype=torch.float32),
            torch.tensor(encoded, dtype=torch.long),
        ),
        batch_size=256,
        shuffle=True,
    )

    set_seed(SEED)
    model = W10CausalTCN(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for _ in range(selected_epoch):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    test_loader = DataLoader(
        TensorDataset(torch.tensor(tcn_tensor[test_pos], dtype=torch.float32)),
        batch_size=512,
        shuffle=False,
    )
    prob_parts = []
    model.eval()
    with torch.no_grad():
        for (bx,) in test_loader:
            prob_parts.append(torch.softmax(model(bx.to(device)), dim=1).cpu().numpy())

    local_probs = np.vstack(prob_parts)
    probs = align_probabilities(classes, local_probs)
    pred = OFFICIAL_ARRAY[np.argmax(probs, axis=1)]

    del model, optimizer, criterion, train_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, selected_epoch


# =============================================================================
# 10. ENSEMBLE DECODERS
# =============================================================================

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
            .rolling(
                window=ENSEMBLE_ROLLING_WINDOW,
                center=True,
                min_periods=1,
            )
            .sum()
            .to_numpy()
        )
        if room518_multiplier != 1.0:
            counts[:, IDX_518] *= room518_multiplier
        final[positions] = np.argmax(counts, axis=1)
    return final


def fused_probabilities(rich_probs, full_probs, rf_probs):
    rich_scaled = apply_temperature(
        rich_probs, ENSEMBLE_TEMPERATURES["xgb_rich60_symmetric_smote"]
    )
    full_scaled = apply_temperature(
        full_probs, ENSEMBLE_TEMPERATURES["xgb_full60_kl"]
    )
    rf_scaled = apply_temperature(
        rf_probs, ENSEMBLE_TEMPERATURES["rf_basic10_symmetric_smote"]
    )
    fused = (
        ENSEMBLE_WEIGHTS["xgb_rich60_symmetric_smote"] * rich_scaled
        + ENSEMBLE_WEIGHTS["xgb_full60_kl"] * full_scaled
        + ENSEMBLE_WEIGHTS["rf_basic10_symmetric_smote"] * rf_scaled
    )
    # A convex combination of valid probability vectors must still sum to 1.
    ensure_probability_matrix(fused, len(fused), "Calibrated ensemble fusion")
    return fused, full_scaled


def decode_calibrated_ensemble(rich_probs, full_probs, rf_probs, timestamps):
    fused, _ = fused_probabilities(rich_probs, full_probs, rf_probs)
    raw = np.argmax(fused, axis=1)
    decoded = centered_majority_decode(raw, timestamps, room518_multiplier=1.0)
    return OFFICIAL_ARRAY[decoded]


def apply_room510_probability_recovery(rich_probs, full_probs, rf_probs):
    fused, full_scaled = fused_probabilities(rich_probs, full_probs, rf_probs)
    adjusted = fused.copy()

    p510 = full_scaled[:, IDX_510]
    rank510 = 1 + np.sum(full_scaled > (p510[:, None] + 1e-12), axis=1)
    room510_gate = (rank510 <= ROOM510_RANK_MAX) & (p510 > ROOM510_PROB_MIN)
    adjusted[room510_gate, IDX_510] *= ROOM510_MULTIPLIER
    return adjusted


def decode_room510_ensemble(rich_probs, full_probs, rf_probs, timestamps):
    adjusted = apply_room510_probability_recovery(rich_probs, full_probs, rf_probs)
    raw = np.argmax(adjusted, axis=1)
    decoded = centered_majority_decode(raw, timestamps, room518_multiplier=1.0)
    return OFFICIAL_ARRAY[decoded]


def decode_class_aware_ensemble(rich_probs, full_probs, rf_probs, timestamps):
    adjusted = apply_room510_probability_recovery(rich_probs, full_probs, rf_probs)
    raw = np.argmax(adjusted, axis=1)
    decoded = centered_majority_decode(
        raw,
        timestamps,
        room518_multiplier=ROOM518_VOTE_MULTIPLIER,
    )
    return OFFICIAL_ARRAY[decoded]


# =============================================================================
# 11. PAPER-READY RESULT HELPERS
# =============================================================================

def true_labels_for_timestamps(timestamps):
    labels = []
    for ts in pd.DatetimeIndex(timestamps):
        key = pd.Timestamp(ts)
        if key not in TS_TO_LABEL:
            raise RuntimeError(f"Evaluation timestamp lacks an official label: {key}")
        labels.append(TS_TO_LABEL[key])
    return np.asarray(labels, dtype=str)


def append_perclass_rows(store, y_true, y_pred, method_key, protocol_key, split_id, seed, scope):
    yy = np.asarray(y_true, dtype=str)
    pp = np.asarray(y_pred, dtype=str)
    precision, recall, f1_values, support = precision_recall_fscore_support(
        yy,
        pp,
        labels=OFFICIAL_CLASSES,
        average=None,
        zero_division=0,
    )
    predicted_support = np.asarray([(pp == c).sum() for c in OFFICIAL_CLASSES], dtype=int)
    for cls, pr, rc, f1v, supp, psupp in zip(
        OFFICIAL_CLASSES,
        precision,
        recall,
        f1_values,
        support,
        predicted_support,
    ):
        store.append(
            {
                "MethodKey": method_key,
                "Method": METHOD_LABELS[method_key],
                "ProtocolKey": protocol_key,
                "Protocol": PROTOCOL_LABELS[protocol_key],
                "Split": split_id,
                "Seed": seed,
                "Scope": scope,
                "Class": cls,
                "Precision": float(pr),
                "Recall": float(rc),
                "F1": float(f1v),
                "Support": int(supp),
                "Predicted_Support": int(psupp),
            }
        )


def append_confusion_rows(store, y_true, y_pred, method_key, protocol_key, split_id, seed, scope):
    yy = np.asarray(y_true, dtype=str)
    pp = np.asarray(y_pred, dtype=str)
    cm = confusion_matrix(yy, pp, labels=OFFICIAL_CLASSES)
    row_sum = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm,
        row_sum,
        out=np.zeros_like(cm, dtype=float),
        where=row_sum > 0,
    )
    for i, true_cls in enumerate(OFFICIAL_CLASSES):
        for j, pred_cls in enumerate(OFFICIAL_CLASSES):
            store.append(
                {
                    "MethodKey": method_key,
                    "Method": METHOD_LABELS[method_key],
                    "ProtocolKey": protocol_key,
                    "Protocol": PROTOCOL_LABELS[protocol_key],
                    "Split": split_id,
                    "Seed": seed,
                    "Scope": scope,
                    "True_Class": true_cls,
                    "Pred_Class": pred_cls,
                    "Count": int(cm[i, j]),
                    "True_Normalized": float(normalized[i, j]),
                }
            )


def append_prediction_csv(path: Path, method_key, split: SplitSpec, y_true, y_pred):
    if split.protocol not in {"forward", "lodo"} and split.seed != ARGS.prediction_random_seed:
        return
    frame = pd.DataFrame(
        {
            "MethodKey": method_key,
            "Method": METHOD_LABELS[method_key],
            "ProtocolKey": split.protocol,
            "Protocol": PROTOCOL_LABELS[split.protocol],
            "Split": split.split_id,
            "Seed": split.seed,
            "timestamp": pd.DatetimeIndex(split.test_ts),
            "y_true": np.asarray(y_true, dtype=str),
            "y_pred": np.asarray(y_pred, dtype=str),
        }
    )
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        compression="gzip",
    )


def method_split_diagnostics(split: SplitSpec):
    train_labels = true_labels_for_timestamps(split.train_ts)
    test_labels = true_labels_for_timestamps(split.test_ts)
    train_classes = set(train_labels)
    test_classes = set(test_labels)
    unseen = sorted(test_classes - train_classes)
    unseen_n = int(np.isin(test_labels, unseen).sum()) if unseen else 0
    return {
        "Real_Train_N_Common": int(len(train_labels)),
        "Real_Train_Classes_Common": int(len(train_classes)),
        "Test_Classes": int(len(test_classes)),
        "Unseen_Test_Classes": int(len(unseen)),
        "Unseen_Test_Class_List": "|".join(unseen),
        "Unseen_Test_Seconds": unseen_n,
    }


# =============================================================================
# 12. MAIN CROSS-PROTOCOL BENCHMARK
# =============================================================================

DETAIL_ROWS = []
PERCLASS_ROWS = []
CONFUSION_ROWS = []
AUGMENTATION_ROWS = []
PROTOCOL_SUMMARY = {}
TEMPORAL_POOLED = {}
FORWARD_LODO_PRED_CHECKS = {}

PREDICTION_PATH = OUTPUT_DIR / f"phase24_selected_predictions_{RUN_ID}.csv.gz"
if PREDICTION_PATH.exists():
    PREDICTION_PATH.unlink()

TCN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_standard_rf(method_key: str, split: SplitSpec):
    train_pos = hist_positions_for_timestamps(split.train_ts)
    test_pos = hist_positions_for_timestamps(split.test_ts)
    y_train = hist.loc[train_pos, "room"].astype(str).to_numpy()

    if method_key == "rf_basic10":
        X_train = X_BASIC_HIST.iloc[train_pos].to_numpy(dtype=float)
        X_test = X_BASIC_HIST.iloc[test_pos].to_numpy(dtype=float)
    elif method_key == "rf_lagstack10":
        X_train = X_LAG_HIST.iloc[train_pos].to_numpy(dtype=float)
        X_test = X_LAG_HIST.iloc[test_pos].to_numpy(dtype=float)
    elif method_key == "rf_topology10":
        X_train = X_SPATIAL_HIST.iloc[train_pos].to_numpy(dtype=float)
        X_test = X_SPATIAL_HIST.iloc[test_pos].to_numpy(dtype=float)
    else:
        raise ValueError(method_key)

    model = build_basic_rf()
    model.fit(X_train, y_train)
    return model.predict(X_test).astype(str)


def probabilities_at_timestamps(result: BranchResult, timestamps):
    requested = pd.DatetimeIndex(timestamps)
    if not result.timestamps.equals(requested):
        # Build an explicit map only when a strict equal-order relationship is absent.
        lookup = {pd.Timestamp(t): i for i, t in enumerate(result.timestamps)}
        missing = [pd.Timestamp(t) for t in requested if pd.Timestamp(t) not in lookup]
        if missing:
            raise RuntimeError(f"Branch result misses {len(missing)} requested timestamps.")
        rows = [lookup[pd.Timestamp(t)] for t in requested]
        return result.probabilities[np.asarray(rows, dtype=int)]
    return result.probabilities


def persist_incremental_outputs():
    pd.DataFrame(DETAIL_ROWS).to_csv(
        OUTPUT_DIR / f"phase24_protocol_detail_{RUN_ID}.csv", index=False
    )
    pd.DataFrame(PERCLASS_ROWS).to_csv(
        OUTPUT_DIR / f"phase24_perclass_detail_{RUN_ID}.csv", index=False
    )
    pd.DataFrame(CONFUSION_ROWS).to_csv(
        OUTPUT_DIR / f"phase24_confusion_detail_{RUN_ID}.csv", index=False
    )
    pd.DataFrame(AUGMENTATION_ROWS).to_csv(
        OUTPUT_DIR / f"phase24_augmentation_diagnostics_{RUN_ID}.csv", index=False
    )


for protocol_key in REQUESTED_PROTOCOLS:
    print("\n" + "=" * 110)
    print("PROTOCOL:", PROTOCOL_LABELS[protocol_key])
    print("=" * 110)
    splits = build_splits(protocol_key)

    protocol_method_split_metrics = {m: [] for m in REQUESTED_METHODS}
    pooled_y = {m: [] for m in REQUESTED_METHODS}
    pooled_pred = {m: [] for m in REQUESTED_METHODS}

    for split in splits:
        print(
            f"\n[{protocol_key}/{split.split_id}] "
            f"train={len(split.train_ts):,} test={len(split.test_ts):,}"
        )
        y_true = true_labels_for_timestamps(split.test_ts)
        common_diag = method_split_diagnostics(split)

        # ------------------------------------------------------------------
        # Freshly train the selected augmented branches once for this split.
        # The in-memory probabilities are then used both for the branch's
        # standalone row and, where applicable, for the two ensemble rows.
        # Nothing is loaded from disk.
        # ------------------------------------------------------------------
        branch_results = {}
        needs_ensemble = any(
            m in REQUESTED_METHODS
            for m in ["ensemble_calibrated", "ensemble_room510", "ensemble_class_aware"]
        )
        branch_required = set()
        for key in AUGMENTED_CONFIGS:
            if key in REQUESTED_METHODS:
                branch_required.add(key)
        if needs_ensemble:
            branch_required.update(
                {
                    "xgb_rich60_symmetric_smote",
                    "xgb_full60_kl",
                    "rf_basic10_symmetric_smote",
                }
            )

        for branch_key in [
            "xgb_rich60_kl_full_smote",
            "xgb_rich60_symmetric_smote",
            "xgb_full60_kl",
            "rf_paper60_kl",
            "lgbm_basic60_kl_full_smote",
            "rf_basic10_symmetric_smote",
        ]:
            if branch_key not in branch_required:
                continue
            prediction_context = (
                split.context_ts
                if branch_key
                in {
                    "xgb_rich60_symmetric_smote",
                    "xgb_full60_kl",
                    "rf_basic10_symmetric_smote",
                }
                and needs_ensemble
                else split.test_ts
            )
            t0 = time.time()
            print(f"   Fitting fresh branch: {METHOD_LABELS[branch_key]}")
            branch_results[branch_key] = fit_augmented_branch(
                branch_key, split, prediction_context
            )
            AUGMENTATION_ROWS.append(
                {
                    **branch_results[branch_key].diagnostics,
                    "PaperName": METHOD_LABELS[branch_key],
                    "Fit_Runtime_Sec": time.time() - t0,
                }
            )

        split_predictions = {}
        split_epochs = {}
        split_runtimes = {}

        # Basic / lag-stack / topology RFs.
        for method_key in ["rf_basic10", "rf_lagstack10", "rf_topology10"]:
            if method_key not in REQUESTED_METHODS:
                continue
            t0 = time.time()
            print("   ->", METHOD_LABELS[method_key])
            split_predictions[method_key] = evaluate_standard_rf(method_key, split)
            split_runtimes[method_key] = time.time() - t0

        # Standalone augmented architectures.
        for method_key in [
            "xgb_rich60_kl_full_smote",
            "xgb_rich60_symmetric_smote",
            "xgb_full60_kl",
            "rf_paper60_kl",
            "lgbm_basic60_kl_full_smote",
            "rf_basic10_symmetric_smote",
        ]:
            if method_key not in REQUESTED_METHODS:
                continue
            t0 = time.time()
            probs = probabilities_at_timestamps(
                branch_results[method_key], split.test_ts
            )
            split_predictions[method_key] = OFFICIAL_ARRAY[np.argmax(probs, axis=1)]
            split_runtimes[method_key] = time.time() - t0
            print("   ->", METHOD_LABELS[method_key], "[fresh fit above]")

        # TCN.
        if "tcn10" in REQUESTED_METHODS:
            t0 = time.time()
            print("   ->", METHOD_LABELS["tcn10"])
            pred_tcn, selected_epoch = fit_predict_tcn(split, TCN_DEVICE)
            split_predictions["tcn10"] = pred_tcn
            split_epochs["tcn10"] = selected_epoch
            split_runtimes["tcn10"] = time.time() - t0

        # Calibrated and class-aware ensembles, using only freshly computed
        # probabilities from this same outer split.
        if needs_ensemble:
            rich_result = branch_results["xgb_rich60_symmetric_smote"]
            full_result = branch_results["xgb_full60_kl"]
            rf_result = branch_results["rf_basic10_symmetric_smote"]

            if not (
                rich_result.timestamps.equals(full_result.timestamps)
                and rich_result.timestamps.equals(rf_result.timestamps)
            ):
                raise RuntimeError(
                    f"Constituent prediction timelines do not match: {protocol_key}/{split.split_id}"
                )
            context_ts = rich_result.timestamps

            if "ensemble_calibrated" in REQUESTED_METHODS:
                t0 = time.time()
                context_pred = decode_calibrated_ensemble(
                    rich_result.probabilities,
                    full_result.probabilities,
                    rf_result.probabilities,
                    context_ts,
                )
                pred_lookup = dict(zip(context_ts, context_pred))
                split_predictions["ensemble_calibrated"] = np.asarray(
                    [pred_lookup[pd.Timestamp(t)] for t in split.test_ts], dtype=str
                )
                split_runtimes["ensemble_calibrated"] = time.time() - t0
                print("   ->", METHOD_LABELS["ensemble_calibrated"])

            if "ensemble_room510" in REQUESTED_METHODS:
                t0 = time.time()
                context_pred = decode_room510_ensemble(
                    rich_result.probabilities,
                    full_result.probabilities,
                    rf_result.probabilities,
                    context_ts,
                )
                pred_lookup = dict(zip(context_ts, context_pred))
                split_predictions["ensemble_room510"] = np.asarray(
                    [pred_lookup[pd.Timestamp(t)] for t in split.test_ts], dtype=str
                )
                split_runtimes["ensemble_room510"] = time.time() - t0
                print("   ->", METHOD_LABELS["ensemble_room510"])

            if "ensemble_class_aware" in REQUESTED_METHODS:
                t0 = time.time()
                context_pred = decode_class_aware_ensemble(
                    rich_result.probabilities,
                    full_result.probabilities,
                    rf_result.probabilities,
                    context_ts,
                )
                pred_lookup = dict(zip(context_ts, context_pred))
                split_predictions["ensemble_class_aware"] = np.asarray(
                    [pred_lookup[pd.Timestamp(t)] for t in split.test_ts], dtype=str
                )
                split_runtimes["ensemble_class_aware"] = time.time() - t0
                print("   ->", METHOD_LABELS["ensemble_class_aware"])

        if set(split_predictions) != set(REQUESTED_METHODS):
            missing = sorted(set(REQUESTED_METHODS) - set(split_predictions))
            extra = sorted(set(split_predictions) - set(REQUESTED_METHODS))
            raise RuntimeError(f"Method execution mismatch. Missing={missing}, extra={extra}")

        for method_key in REQUESTED_METHODS:
            pred = np.asarray(split_predictions[method_key], dtype=str)
            if len(pred) != len(y_true):
                raise RuntimeError(
                    f"{method_key}/{protocol_key}/{split.split_id}: prediction length mismatch"
                )
            unexpected_pred = sorted(set(pred) - set(OFFICIAL_CLASSES))
            if unexpected_pred:
                raise RuntimeError(
                    f"{method_key}/{protocol_key}/{split.split_id}: unexpected predictions {unexpected_pred}"
                )

            metrics = fixed23_metrics(y_true, pred)
            print(
                f"      {method_key:34s} "
                f"Macro={metrics['Macro_F1']:.6f} "
                f"Weighted={metrics['Weighted_F1']:.6f} "
                f"Acc={metrics['Accuracy']:.6f} N={metrics['Eval_N']:,}"
            )

            row = {
                "MethodKey": method_key,
                "Method": METHOD_LABELS[method_key],
                "ProtocolKey": protocol_key,
                "Protocol": PROTOCOL_LABELS[protocol_key],
                "Split": split.split_id,
                "Seed": split.seed,
                **metrics,
                **common_diag,
                "TCN_Selected_Epoch": split_epochs.get(method_key, np.nan),
                "Runtime_Sec": split_runtimes.get(method_key, np.nan),
            }
            DETAIL_ROWS.append(row)
            protocol_method_split_metrics[method_key].append(metrics)

            append_perclass_rows(
                PERCLASS_ROWS,
                y_true,
                pred,
                method_key,
                protocol_key,
                split.split_id,
                split.seed,
                "split",
            )
            append_confusion_rows(
                CONFUSION_ROWS,
                y_true,
                pred,
                method_key,
                protocol_key,
                split.split_id,
                split.seed,
                "split",
            )
            append_prediction_csv(
                PREDICTION_PATH,
                method_key,
                split,
                y_true,
                pred,
            )

            if protocol_key in {"forward", "lodo"}:
                pooled_y[method_key].append(y_true)
                pooled_pred[method_key].append(pred)

            # Dynamic Apr13 consistency check can be performed after both protocols.
            if protocol_key in {"forward", "lodo"} and split.split_id == "apr13":
                FORWARD_LODO_PRED_CHECKS[(method_key, protocol_key)] = pred.copy()

        persist_incremental_outputs()
        del branch_results, split_predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Protocol-level aggregation.
    for method_key in REQUESTED_METHODS:
        split_metrics = protocol_method_split_metrics[method_key]
        if protocol_key in {"forward", "lodo"}:
            yy = np.concatenate(pooled_y[method_key])
            pp = np.concatenate(pooled_pred[method_key])
            pooled_metrics = fixed23_metrics(yy, pp)
            PROTOCOL_SUMMARY[(method_key, protocol_key)] = {
                **pooled_metrics,
                "Macro_F1_SD": 0.0,
                "Weighted_F1_SD": 0.0,
                "Accuracy_SD": 0.0,
                "Split_Count": len(split_metrics),
            }
            TEMPORAL_POOLED[(method_key, protocol_key)] = (yy, pp)
            append_perclass_rows(
                PERCLASS_ROWS,
                yy,
                pp,
                method_key,
                protocol_key,
                "POOLED",
                None,
                "pooled",
            )
            append_confusion_rows(
                CONFUSION_ROWS,
                yy,
                pp,
                method_key,
                protocol_key,
                "POOLED",
                None,
                "pooled",
            )
        else:
            PROTOCOL_SUMMARY[(method_key, protocol_key)] = {
                "Macro_F1": float(np.mean([m["Macro_F1"] for m in split_metrics])),
                "Macro_F1_SD": sample_sd([m["Macro_F1"] for m in split_metrics]),
                "Weighted_F1": float(np.mean([m["Weighted_F1"] for m in split_metrics])),
                "Weighted_F1_SD": sample_sd([m["Weighted_F1"] for m in split_metrics]),
                "Accuracy": float(np.mean([m["Accuracy"] for m in split_metrics])),
                "Accuracy_SD": sample_sd([m["Accuracy"] for m in split_metrics]),
                "Eval_N": float(np.mean([m["Eval_N"] for m in split_metrics])),
                "True_Classes": float(np.mean([m["True_Classes"] for m in split_metrics])),
                "Pred_Classes": float(np.mean([m["Pred_Classes"] for m in split_metrics])),
                "Split_Count": len(split_metrics),
            }

    persist_incremental_outputs()


# Dynamic integrity check: Forward-Apr13 and LODO-Apr13 have the same training
# dates (Apr10-Apr12), so every deterministic architecture should make identical
# predictions there. This checks implementation consistency without using any
# historical metric value.
if "forward" in REQUESTED_PROTOCOLS and "lodo" in REQUESTED_PROTOCOLS:
    for method_key in REQUESTED_METHODS:
        fwd_key = (method_key, "forward")
        lodo_key = (method_key, "lodo")
        if fwd_key in FORWARD_LODO_PRED_CHECKS and lodo_key in FORWARD_LODO_PRED_CHECKS:
            if not np.array_equal(
                FORWARD_LODO_PRED_CHECKS[fwd_key],
                FORWARD_LODO_PRED_CHECKS[lodo_key],
            ):
                raise RuntimeError(
                    f"Dynamic Apr13 Forward/LODO prediction parity failed for {METHOD_LABELS[method_key]}."
                )
print("\nDynamic Apr13 Forward/LODO parity checks: PASS")


# =============================================================================
# 13. MASTER TABLE / PER-CLASS SUMMARY
# =============================================================================

summary_rows = []
for method_key in REQUESTED_METHODS:
    for protocol_key in REQUESTED_PROTOCOLS:
        metrics = PROTOCOL_SUMMARY.get((method_key, protocol_key))
        if metrics is None:
            continue
        summary_rows.append(
            {
                "MethodKey": method_key,
                "Method": METHOD_LABELS[method_key],
                "ProtocolKey": protocol_key,
                "Protocol": PROTOCOL_LABELS[protocol_key],
                **metrics,
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    OUTPUT_DIR / f"phase24_protocol_summary_{RUN_ID}.csv", index=False
)

master_rows = []
for method_key in METHOD_ORDER:
    if method_key not in REQUESTED_METHODS:
        continue
    row = {"Method": METHOD_LABELS[method_key]}
    for protocol_key in [
        "row_random",
        "block45_random",
        "event_random",
        "lodo",
        "forward",
    ]:
        metrics = PROTOCOL_SUMMARY.get((method_key, protocol_key))
        row[PROTOCOL_LABELS[protocol_key]] = (
            np.nan if metrics is None else metrics["Macro_F1"]
        )
    row["Exact 3.5-day/1.5-day"] = "N/A: Apr14 labels hidden"
    master_rows.append(row)

master_df = pd.DataFrame(master_rows)
master_df.to_csv(
    OUTPUT_DIR / f"phase24_master_macro_table_{RUN_ID}.csv", index=False
)

# Random-protocol per-class summaries = mean/sample-SD across seeds.
# Temporal protocols = pooled class metrics.
perclass_detail_df = pd.DataFrame(PERCLASS_ROWS)
perclass_summary_parts = []
if not perclass_detail_df.empty:
    temporal_pc = perclass_detail_df[
        (perclass_detail_df["Scope"] == "pooled")
        & perclass_detail_df["ProtocolKey"].isin(["forward", "lodo"])
    ].copy()
    if not temporal_pc.empty:
        temporal_pc["Precision_SD"] = 0.0
        temporal_pc["Recall_SD"] = 0.0
        temporal_pc["F1_SD"] = 0.0
        temporal_pc["Support_Mean"] = temporal_pc["Support"].astype(float)
        temporal_pc["Predicted_Support_Mean"] = temporal_pc[
            "Predicted_Support"
        ].astype(float)
        perclass_summary_parts.append(
            temporal_pc[
                [
                    "MethodKey", "Method", "ProtocolKey", "Protocol", "Class",
                    "Precision", "Precision_SD", "Recall", "Recall_SD",
                    "F1", "F1_SD", "Support_Mean", "Predicted_Support_Mean",
                ]
            ]
        )

    random_pc = perclass_detail_df[
        (perclass_detail_df["Scope"] == "split")
        & perclass_detail_df["ProtocolKey"].isin(
            ["row_random", "block45_random", "event_random"]
        )
    ].copy()
    if not random_pc.empty:
        grouped_rows = []
        group_cols = ["MethodKey", "Method", "ProtocolKey", "Protocol", "Class"]
        for keys, g in random_pc.groupby(group_cols, sort=False):
            grouped_rows.append(
                {
                    **dict(zip(group_cols, keys)),
                    "Precision": float(g["Precision"].mean()),
                    "Precision_SD": sample_sd(g["Precision"]),
                    "Recall": float(g["Recall"].mean()),
                    "Recall_SD": sample_sd(g["Recall"]),
                    "F1": float(g["F1"].mean()),
                    "F1_SD": sample_sd(g["F1"]),
                    "Support_Mean": float(g["Support"].mean()),
                    "Predicted_Support_Mean": float(g["Predicted_Support"].mean()),
                }
            )
        perclass_summary_parts.append(pd.DataFrame(grouped_rows))

perclass_summary_df = (
    pd.concat(perclass_summary_parts, ignore_index=True)
    if perclass_summary_parts
    else pd.DataFrame()
)
perclass_summary_df.to_csv(
    OUTPUT_DIR / f"phase24_perclass_summary_{RUN_ID}.csv", index=False
)

persist_incremental_outputs()


# =============================================================================
# 14. EXACT PROJECT PAPER-STYLE 45-S RANDOM FOREST (SEPARATE POPULATION)
# =============================================================================

def extract_exact_paper45_features(df):
    X_parts, y_parts = [], []
    for _, grp in df.groupby(df.index.date, sort=True):
        sig = grp[BEACONS].replace(MODEL_MISSING_RSSI, np.nan)
        obs = grp[OBS_BEACONS].astype(float)
        roll_sig = sig.rolling(window=45, min_periods=1)
        roll_obs = obs.rolling(window=45, min_periods=1)

        f_mean = roll_sig.mean().fillna(MODEL_MISSING_RSSI).iloc[::45]
        f_std = roll_sig.std().fillna(0.0).iloc[::45]
        f_max = roll_sig.max().fillna(MODEL_MISSING_RSSI).iloc[::45]
        f_min = roll_sig.min().fillna(MODEL_MISSING_RSSI).iloc[::45]
        f_var = roll_sig.var().fillna(0.0).iloc[::45]
        f_med = roll_sig.median().fillna(MODEL_MISSING_RSSI).iloc[::45]
        f_sum = roll_sig.sum().fillna(MODEL_MISSING_RSSI).iloc[::45]
        f_act = roll_obs.sum().fillna(0.0).iloc[::45]

        f_mean.columns = [f"{b}_mean" for b in BEACONS]
        f_std.columns = [f"{b}_std" for b in BEACONS]
        f_max.columns = [f"{b}_max" for b in BEACONS]
        f_min.columns = [f"{b}_min" for b in BEACONS]
        f_var.columns = [f"{b}_var" for b in BEACONS]
        f_med.columns = [f"{b}_med" for b in BEACONS]
        f_sum.columns = [f"{b}_sum" for b in BEACONS]
        f_act.columns = [f"{b}_act" for b in BEACONS]

        time_features = pd.DataFrame(index=f_mean.index)
        minutes = f_mean.index.hour * 60 + f_mean.index.minute
        time_features["minute_of_day"] = minutes
        time_features["hour"] = f_mean.index.hour
        time_features["sin_time"] = np.sin(2 * np.pi * minutes / 1440)
        time_features["cos_time"] = np.cos(2 * np.pi * minutes / 1440)

        X = pd.concat(
            [
                f_mean, f_var, f_std, f_min, f_max, f_sum, f_med, f_act,
                time_features,
            ],
            axis=1,
        )
        y = grp["assigned_room"].iloc[::45].copy()
        if X.columns.duplicated().any():
            raise RuntimeError("Duplicate exact paper-style features.")
        X_parts.append(X)
        y_parts.append(y)

    return pd.concat(X_parts).sort_index(), pd.concat(y_parts).sort_index()


def fit_predict_paper45_rf(X_train, y_train, X_test):
    encoder = LabelEncoder()
    y_local = encoder.fit_transform(y_train.astype(str))
    model = build_augmented_rf()  # same RF hyperparameters/default max_features as Phase6 paper baseline
    model.fit(X_train, y_local)
    pred_local = model.predict(X_test).astype(int)
    return encoder.inverse_transform(pred_local).astype(str)


print("\n" + "=" * 110)
print("SEPARATE LITERATURE-COMPARISON BASELINE: PAPER-STYLE RF 45 s")
print("=" * 110)
X_PAPER45, y_PAPER45 = extract_exact_paper45_features(df_base.copy())
valid45 = y_PAPER45.astype(str).isin(OFFICIAL_CLASSES)
X45 = X_PAPER45.loc[valid45].copy()
y45 = y_PAPER45.loc[valid45].astype(str).copy()

paper45_rows = []
paper45_perclass = []

# Random 70/30 is only computed if a stratified split is mathematically valid.
class_counts45 = y45.value_counts().reindex(OFFICIAL_CLASSES, fill_value=0)
if (class_counts45[class_counts45 > 0] < 2).any():
    paper45_rows.append(
        {
            "Protocol": "Stratified random 70/30",
            "Status": "N/A",
            "Reason": "At least one class present in the exact 45-second sample population has fewer than two samples, so an exact stratified split is not valid.",
        }
    )
else:
    random45_metrics = []
    for seed in RANDOM_SEEDS:
        train_idx, test_idx = train_test_split(
            np.arange(len(X45)),
            test_size=0.30,
            random_state=seed,
            stratify=y45.to_numpy(),
        )
        pred = fit_predict_paper45_rf(
            X45.iloc[train_idx], y45.iloc[train_idx], X45.iloc[test_idx]
        )
        true = y45.iloc[test_idx].to_numpy(dtype=str)
        m = fixed23_metrics(true, pred)
        random45_metrics.append(m)
        precision, recall, f1s, support = precision_recall_fscore_support(
            true, pred, labels=OFFICIAL_CLASSES, average=None, zero_division=0
        )
        for cls, pr, rc, f1v, supp in zip(
            OFFICIAL_CLASSES, precision, recall, f1s, support
        ):
            paper45_perclass.append(
                {
                    "Protocol":"Stratified random 70/30",
                    "Seed":seed,
                    "Class":cls,
                    "Precision":pr,
                    "Recall":rc,
                    "F1":f1v,
                    "Support":int(supp),
                }
            )
    paper45_rows.append(
        {
            "Protocol":"Stratified random 70/30",
            "Status":"OK",
            "Macro_F1":float(np.mean([m["Macro_F1"] for m in random45_metrics])),
            "Macro_F1_SD":sample_sd([m["Macro_F1"] for m in random45_metrics]),
            "Weighted_F1":float(np.mean([m["Weighted_F1"] for m in random45_metrics])),
            "Weighted_F1_SD":sample_sd([m["Weighted_F1"] for m in random45_metrics]),
            "Accuracy":float(np.mean([m["Accuracy"] for m in random45_metrics])),
            "Accuracy_SD":sample_sd([m["Accuracy"] for m in random45_metrics]),
            "Eval_N_Mean":float(np.mean([m["Eval_N"] for m in random45_metrics])),
        }
    )

# Forward and LODO are scored on this sparse native 45-second population and are
# therefore reported separately from the common-population master table.
for protocol_name in ["Forward", "LODO"]:
    all_true, all_pred = [], []
    for val_date in DEV_DATES[1:]:
        if protocol_name == "Forward":
            train_dates = DEV_DATES[:DEV_DATES.index(val_date)]
        else:
            train_dates = [d for d in DEV_DATES if d != val_date]

        train_mask = np.isin(X45.index.date, train_dates)
        test_mask = X45.index.date == val_date
        X_train, y_train = X45.loc[train_mask], y45.loc[train_mask]
        X_test, y_test = X45.loc[test_mask], y45.loc[test_mask]
        if len(X_train) == 0 or len(X_test) == 0:
            raise RuntimeError(f"Paper45 {protocol_name}: empty fold for {val_date}")

        pred = fit_predict_paper45_rf(X_train, y_train, X_test)
        all_true.append(y_test.to_numpy(dtype=str))
        all_pred.append(pred)

    yy = np.concatenate(all_true)
    pp = np.concatenate(all_pred)
    m = fixed23_metrics(yy, pp)
    paper45_rows.append(
        {
            "Protocol":protocol_name,
            "Status":"OK",
            **m,
            "Macro_F1_SD":0.0,
            "Weighted_F1_SD":0.0,
            "Accuracy_SD":0.0,
        }
    )

paper45_df = pd.DataFrame(paper45_rows)
paper45_df.to_csv(
    OUTPUT_DIR / f"phase24_exact_paper45_baseline_{RUN_ID}.csv", index=False
)
pd.DataFrame(
    {"Class":OFFICIAL_CLASSES, "Exact45s_Samples":class_counts45.reindex(OFFICIAL_CLASSES).to_numpy(dtype=int)}
).to_csv(
    OUTPUT_DIR / f"phase24_exact_paper45_class_counts_{RUN_ID}.csv", index=False
)
pd.DataFrame(paper45_perclass).to_csv(
    OUTPUT_DIR / f"phase24_exact_paper45_perclass_{RUN_ID}.csv", index=False
)

# Explicitly document why the 3.5d/1.5d score is unavailable.
pd.DataFrame(
    [
        {
            "Protocol":"Exact 3.5-day training / 1.5-day testing",
            "Status":"NOT COMPUTABLE FROM CHALLENGE RELEASE",
            "Reason":"The exact literature protocol requires labeled Apr14 observations, but Apr14 location labels are hidden competition ground truth.",
        }
    ]
).to_csv(
    OUTPUT_DIR / f"phase24_3p5d_1p5d_status_{RUN_ID}.csv", index=False
)


# =============================================================================
# 15. RUN METADATA / FINAL REPORT
# =============================================================================

run_metadata = {
    "run_id": RUN_ID,
    "fresh_run": True,
    "reads_previous_result_files": False,
    "reads_saved_probability_files": False,
    "reads_saved_prediction_files": False,
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "imbalanced_learn": imblearn.__version__,
    "xgboost": xgb.__version__,
    "lightgbm": lgb.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "label_file": str(LABEL_FILE_PATH),
    "ble_dir": str(TRAIN_BLE_DIR),
    "topology_file": str(TOPOLOGY_FILE),
    "topology_sha256_calculated": file_sha256(TOPOLOGY_FILE),
    "protocols": REQUESTED_PROTOCOLS,
    "methods": REQUESTED_METHODS,
    "random_seeds": RANDOM_SEEDS,
    "development_dates_inferred": [str(d) for d in DEV_DATES],
    "common_population_n": int(len(hist)),
    "official_classes": OFFICIAL_CLASSES,
    "fusion_temperatures": ENSEMBLE_TEMPERATURES,
    "fusion_weights": ENSEMBLE_WEIGHTS,
    "rolling_window": ENSEMBLE_ROLLING_WINDOW,
    "room510_rule": {
        "rank_max": ROOM510_RANK_MAX,
        "probability_min": ROOM510_PROB_MIN,
        "multiplier": ROOM510_MULTIPLIER,
    },
    "room518_vote_multiplier": ROOM518_VOTE_MULTIPLIER,
}
with open(OUTPUT_DIR / f"phase24_run_metadata_{RUN_ID}.json", "w", encoding="utf-8") as f:
    json.dump(run_metadata, f, indent=2, default=str)

print("\n" + "=" * 110)
print("MASTER FIXED-23 MACRO-F1 TABLE")
print("=" * 110)
print(master_df.to_string(index=False, na_rep="N/A", float_format=lambda x: f"{x:.6f}"))

print("\nSeparate exact paper-style 45 s baseline:")
print(paper45_df.to_string(index=False, na_rep="N/A", float_format=lambda x: f"{x:.6f}"))

print("\nAll results above were calculated in this run. No saved prediction/probability/result file was used as input.")
print("Random-protocol SD values use sample SD (ddof=1).")
print("The row-random protocol is intentionally optimistic and should be labeled diagnostic in the paper.")
print("The exact 45 s paper-style baseline is kept separate because it has a different native evaluation population.")
print("The exact 3.5-day/1.5-day literature protocol remains N/A because Apr14 labels are hidden.")

print("\nSaved outputs:")
for path in sorted(OUTPUT_DIR.glob(f"phase24_*_{RUN_ID}.*")):
    print(" ", path)
print("=" * 110)
