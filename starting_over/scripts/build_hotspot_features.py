# scripts/build_hotspot_features.py
import pandas as pd
from pathlib import Path
from collections import deque

SOURCE = Path("scripts/BPD_Arrests.csv")
TARGET = Path("data/hotspot_training_features.csv")
MAX_ROWS = 80000  # downsample to keep model training manageable

raw = pd.read_csv(SOURCE, parse_dates=["ArrestDateTime"])
if len(raw) > MAX_ROWS:
    raw = raw.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

violent_keywords = ("ASSAULT", "SHOOTING", "HOMICIDE", "ROBBERY", "CARJACKING")

def infer_crime_type(text):
    if pd.isna(text):
        return "other"
    text = text.lower()
    for kw in violent_keywords:
        if kw.lower() in text:
            return kw.lower()
    if "drug" in text:
        return "drug"
    if "theft" in text or "burglary" in text:
        return "property"
    return "other"

def severity_score(row):
    desc = (row.get("ChargeDescription") or "").lower()
    score = 0.0
    if any(word in desc for word in ("felony", "weapon", "gun")):
        score += 0.5
    if any(word in desc for word in ("violence", "assault", "homicide")):
        score += 0.4
    return min(score, 1.0)

# rough hotspot label: 1 if offense matches violent keywords, else 0
def label_hotspot(offense):
    if pd.isna(offense):
        return 0
    offense = offense.upper()
    return int(any(kw in offense for kw in violent_keywords))

def normalize_location(row):
    if isinstance(row, str) and row.strip():
        return row.strip().lower()
    return "unknown"

def compute_recent_counts(df, location_col, timestamp_col, window_hours=24):
    window_seconds = window_hours * 3600
    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
    numeric_ts = timestamps.astype("int64") // 1_000_000_000
    valid_mask = timestamps.notna()
    temp = pd.DataFrame(
        {
            "idx": df.index[valid_mask],
            "location": df.loc[valid_mask, location_col].fillna("unknown"),
            "ts": numeric_ts[valid_mask],
        }
    )
    temp = temp.sort_values(["location", "ts"])
    counts = pd.Series(0, index=df.index, dtype=int)
    window = deque()
    current_location = None

    for row in temp.itertuples(index=False):
        loc = row.location
        ts = int(row.ts)
        idx = row.idx
        if loc != current_location:
            window.clear()
            current_location = loc
        while window and ts - window[0][0] > window_seconds:
            window.popleft()
        counts.at[idx] = len(window)
        window.append((ts, idx))
    return counts

raw = raw.assign(
    location=raw["Neighborhood"].apply(normalize_location).fillna(
        raw["District"].apply(normalize_location)
    ),
    hour_of_day=pd.to_datetime(raw["ArrestDateTime"], errors="coerce").dt.hour.fillna(0).astype(int),
    day_of_week=pd.to_datetime(raw["ArrestDateTime"], errors="coerce").dt.dayofweek.fillna(0).astype(int),
    crime_type=raw["IncidentOffence"].apply(infer_crime_type),
    severity_score=raw.apply(severity_score, axis=1),
    is_hotspot=raw["IncidentOffence"].apply(label_hotspot),
)
raw["incident_count_last_24h"] = compute_recent_counts(
    raw,
    location_col="location",
    timestamp_col="ArrestDateTime",
    window_hours=24,
)

feature_cols = [
    "location",
    "hour_of_day",
    "day_of_week",
    "crime_type",
    "incident_count_last_24h",
    "severity_score",
    "is_hotspot",
]

TARGET.parent.mkdir(parents=True, exist_ok=True)
raw[feature_cols].to_csv(TARGET, index=False)
print(f"Saved {len(raw)} rows to {TARGET}")
