"""Data: load a small patient cohort, turn it into features, split it by patient.

Source: the public MSK-CHORD study on cBioPortal (dataset page: OncoLens-data, wiki-llm/datasets/msk-chord.md;
license CC BY-NC-ND, so rows are cached locally in a git-ignored folder and never committed, printed or re-published).

    cohort = load_cohort(cfg.data)      # patient-level table
    describe(cohort)                    # aggregates only, safe to log

THE LABEL IS A PLUMBING LABEL. `OS_STATUS` (deceased at last follow-up) is predicted from attributes that are
not anchored to any prediction time (current age, sites of disease ever recorded, highest stage). That is not
leak-free and it is not the OncoLens task, which is undefined (lab gap G1). It exists so the pipeline has
something to fit. Do not report its numbers as results.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oncolens.config import DataConfig

API = "https://www.cbioportal.org/api"
CACHE_DIR = Path(".cache/data")
LABEL = "OS_STATUS"
NUMERIC = ["CURRENT_AGE_DEID", "NUM_ICDO_DX"]
CATEGORICAL = [
    "GENDER", "STAGE_HIGHEST_RECORDED", "SMOKING_PREDICTIONS_3_CLASSES", "PRIOR_MED_TO_MSK", "HER2", "HR",
    "HISTORY_OF_PDL1", "ADRENAL_GLANDS", "BONE", "CNS_BRAIN", "INTRA_ABDOMINAL", "LIVER", "LUNG", "LYMPH_NODES",
    "PLEURA", "REPRODUCTIVE_ORGANS", "OTHER",
]


@dataclass
class Cohort:
    """One row per patient. `train` and `val` index disjoint patients."""

    patient_ids: list[str]
    x: np.ndarray           # float32 [n_patients, n_features]
    y: np.ndarray           # float32 [n_patients], 1 = positive plumbing label
    feature_names: list[str]
    train: np.ndarray       # int64 row indices
    val: np.ndarray


def _get_json(url: str, body: dict | None = None) -> list:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def fetch_table(study: str, n_patients: int) -> dict[str, dict[str, str]]:
    """{patient_id: {attribute: value}} for the first `n_patients` of the study, cached on disk."""
    cache = CACHE_DIR / f"{study}-{n_patients}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    patients = _get_json(f"{API}/studies/{study}/patients?projection=ID&pageSize={n_patients}&pageNumber=0")
    ids = [p["patientId"] for p in patients]
    rows = _get_json(
        f"{API}/studies/{study}/clinical-data/fetch?clinicalDataType=PATIENT&projection=SUMMARY",
        {"attributeIds": [LABEL, *NUMERIC, *CATEGORICAL], "ids": ids},
    )
    table: dict[str, dict[str, str]] = {pid: {} for pid in ids}
    for row in rows:
        table[row["patientId"]][row["clinicalAttributeId"]] = row["value"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(table), encoding="utf-8")
    return table


def split_by_patient(patient_ids: list[str], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic split on a hash of the patient id: a patient is always on the same side, on any machine."""
    def bucket(pid: str) -> float:
        return int(hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    is_val = np.array([bucket(pid) < val_fraction for pid in patient_ids])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def load_cohort(cfg: DataConfig) -> Cohort:
    table = fetch_table(cfg.study, cfg.n_patients)
    ids = sorted(pid for pid, attrs in table.items() if attrs.get(LABEL))  # patients with a known label
    y = np.array([table[pid][LABEL].startswith("1") for pid in ids], dtype=np.float32)
    train, val = split_by_patient(ids, cfg.val_fraction, cfg.seed)

    columns, names = [], []
    for attr in NUMERIC:  # standardized with train statistics; missing -> 0 plus an indicator column
        raw = np.array([_to_float(table[pid].get(attr)) for pid in ids])
        missing = np.isnan(raw)
        mean, std = np.nanmean(raw[train]), np.nanstd(raw[train]) or 1.0
        columns += [np.where(missing, 0.0, (raw - mean) / std), missing.astype(float)]
        names += [attr, f"{attr}__missing"]
    for attr in CATEGORICAL:  # one-hot over the values seen in the train split
        values = [table[pid].get(attr, "__missing") for pid in ids]
        for level in sorted({values[i] for i in train}):
            columns.append(np.array([v == level for v in values], dtype=float))
            names.append(f"{attr}={level}")
    return Cohort(ids, np.stack(columns, axis=1).astype(np.float32), y, names, train, val)


def _to_float(value: str | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def describe(cohort: Cohort) -> dict:
    """Aggregates only: safe for logs, the run folder and W&B."""
    return {
        "n_patients": len(cohort.patient_ids),
        "n_features": len(cohort.feature_names),
        "n_train": int(len(cohort.train)),
        "n_val": int(len(cohort.val)),
        "positive_rate_train": round(float(cohort.y[cohort.train].mean()), 4),
        "positive_rate_val": round(float(cohort.y[cohort.val].mean()), 4),
    }
