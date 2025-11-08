import os
import sys
import json
import argparse
import traceback
from datetime import datetime, timedelta, timezone
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction import DictVectorizer
import google.generativeai as genai


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json"

with open(CONFIG_PATH, "r") as _f:
    config = json.load(_f)


# Configuration defaults
config.setdefault("incident_dataset_path", "scripts/BPD_Arrests.csv")
config.setdefault("summary_recent_days", 30)
config.setdefault("summary_max_records", 50)
config.setdefault("max_context_chars", 3000)
config.setdefault("enable_hotspot_prediction", True)
config.setdefault("hotspot_model_path", "crime_hotspot_model.pkl")
config.setdefault("feature_vectorizer_path", "crime_feature_vectorizer.pkl")
config.setdefault("incident_recency_hours", 24)
config.setdefault("gemini_model", "models/gemini-2.5-flash")
config.setdefault("gemini_stream", True)
config.setdefault("gemini_temperature", 0.2)
config.setdefault("gemini_top_p", 0.9)
config.setdefault("gemini_top_k", 40)
config.setdefault("summary_include_samples", 5)
config.setdefault("homeless_dataset_path", "scripts/Homeless_Shelter_1167958174770612354.csv")
config.setdefault("university_dataset_path", "scripts/Universities_and_Colleges_-3774145757448629151.csv")


# Gemini setup
gemini_api_key = config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Missing Gemini API key. Set config['gemini_api_key'] or GEMINI_API_KEY env var.")

genai.configure(api_key=gemini_api_key)
gemini_model_name = config["gemini_model"]
gemini_generation_config = {
    "temperature": config.get("gemini_temperature", 0.2),
    "top_p": config.get("gemini_top_p", 0.9),
    "top_k": config.get("gemini_top_k", 40),
}
gemini_model = genai.GenerativeModel(gemini_model_name, generation_config=gemini_generation_config)
print(f"Using Gemini model: {gemini_model_name}", file=sys.stderr)


# Hotspot model setup
hotspot_model = None
feature_vectorizer = None
if config.get("enable_hotspot_prediction", True):
    hotspot_model_path = Path(config.get("hotspot_model_path", "crime_hotspot_model.pkl"))
    if not hotspot_model_path.is_absolute():
        hotspot_model_path = ROOT_DIR / hotspot_model_path
    vectorizer_path = Path(config.get("feature_vectorizer_path", "crime_feature_vectorizer.pkl"))
    if not vectorizer_path.is_absolute():
        vectorizer_path = ROOT_DIR / vectorizer_path

    try:
        hotspot_model = joblib.load(hotspot_model_path)
        print(f"Loaded hotspot model from {hotspot_model_path}", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: failed to load hotspot model ({exc}); predictions will be skipped.", file=sys.stderr)
        hotspot_model = None

    try:
        feature_vectorizer = joblib.load(vectorizer_path)
        print(f"Loaded feature vectorizer from {vectorizer_path}", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: failed to load feature vectorizer ({exc}); using on-the-fly vectorizer.", file=sys.stderr)
        feature_vectorizer = DictVectorizer(sparse=False)


INCIDENT_DF: pd.DataFrame | None = None
HOMELESS_DF: pd.DataFrame | None = None
UNIVERSITY_DF: pd.DataFrame | None = None
VIOLENT_KEYWORDS = ("ASSAULT", "SHOOT", "HOMICIDE", "ROBBERY", "CARJACKING", "WEAPON", "GUN")


def resolve_dataset_path() -> Path:
    dataset_path = Path(config.get("incident_dataset_path", "scripts/BPD_Arrests.csv"))
    if not dataset_path.is_absolute():
        dataset_path = ROOT_DIR / dataset_path
    return dataset_path


def load_incident_dataframe() -> pd.DataFrame:
    global INCIDENT_DF
    if INCIDENT_DF is not None:
        return INCIDENT_DF

    dataset_path = resolve_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Incident dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "ArrestDateTime" not in df.columns:
        raise ValueError("Dataset must include 'ArrestDateTime' column.")

    df["ArrestDateTime"] = pd.to_datetime(df["ArrestDateTime"], errors="coerce")
    text_columns = [
        "ArrestLocation",
        "IncidentLocation",
        "IncidentOffence",
        "Charge",
        "ChargeDescription",
        "Neighborhood",
        "District",
    ]
    for col in text_columns:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    INCIDENT_DF = df
    return INCIDENT_DF


def load_homeless_dataframe() -> pd.DataFrame:
    global HOMELESS_DF
    if HOMELESS_DF is not None:
        return HOMELESS_DF

    dataset_path = Path(config.get("homeless_dataset_path"))
    if not dataset_path.is_absolute():
        dataset_path = ROOT_DIR / dataset_path
    if not dataset_path.exists():
        print(f"Warning: Homeless dataset missing at {dataset_path}", file=sys.stderr)
        HOMELESS_DF = pd.DataFrame()
        return HOMELESS_DF

    df = pd.read_csv(dataset_path)
    text_cols = ["NAME", "ADDRESS", "CITY", "STATE", "ZIPCODE", "NGHBRHD", "CNTCT_NME"]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")
    HOMELESS_DF = df
    return HOMELESS_DF


def load_university_dataframe() -> pd.DataFrame:
    global UNIVERSITY_DF
    if UNIVERSITY_DF is not None:
        return UNIVERSITY_DF

    dataset_path = Path(config.get("university_dataset_path"))
    if not dataset_path.is_absolute():
        dataset_path = ROOT_DIR / dataset_path
    if not dataset_path.exists():
        print(f"Warning: University dataset missing at {dataset_path}", file=sys.stderr)
        UNIVERSITY_DF = pd.DataFrame()
        return UNIVERSITY_DF

    df = pd.read_csv(dataset_path)
    for col in ["NAME", "ADDRESS"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")
    UNIVERSITY_DF = df
    return UNIVERSITY_DF


def extract_location(query: str, default: str = "unknown") -> str:
    if not query:
        return default
    lower = query.lower()
    for keyword in ("near", "in", "at", "around"):
        if keyword in lower:
            idx = lower.find(keyword) + len(keyword)
            candidate = query[idx:].strip(" ?.,")
            if candidate:
                return candidate
    return query.strip(" ?.,") or default


def extract_hour(query: str) -> int | None:
    if not query:
        return None
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", query, flags=re.IGNORECASE)
    if not match:
        if re.search(r"after\s+(\d{1,2})\s*(am|pm)", query, flags=re.IGNORECASE):
            match = re.search(r"after\s+(\d{1,2})\s*(am|pm)", query, flags=re.IGNORECASE)
        else:
            return None
    hour = int(match.group(1))
    meridiem = match.group(3).lower() if match.group(3) else None
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return hour % 24


def build_location_mask(df: pd.DataFrame, location_label: str, columns: List[str]) -> pd.Series:
    if not location_label or location_label.lower() == "unknown":
        return pd.Series(True, index=df.index)
    pattern = re.escape(location_label.lower())
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(pattern, na=False)
    return mask


def filter_incidents_for_query(
    df: pd.DataFrame,
    location_query: str,
    hour_filter: int | None,
    recent_days: int,
    max_records: int,
) -> pd.DataFrame:
    filtered = df.copy()

    if recent_days and recent_days > 0:
        reference_date = df["ArrestDateTime"].max()
        if pd.isna(reference_date):
            reference_date = datetime.now(timezone.utc)
        cutoff = reference_date - timedelta(days=recent_days)
        filtered = filtered[filtered["ArrestDateTime"] >= cutoff]

    if location_query and location_query.lower() != "unknown":
        mask = pd.Series(False, index=filtered.index)
        pattern = re.escape(location_query.lower())
        for col in ("ArrestLocation", "IncidentLocation", "Neighborhood", "District"):
            mask = mask | filtered[col].str.lower().str.contains(pattern, na=False)
        if mask.any():
            filtered = filtered[mask]

    if hour_filter is not None:
        filtered = filtered[filtered["ArrestDateTime"].dt.hour >= hour_filter]

    filtered = filtered.sort_values("ArrestDateTime", ascending=False)
    if max_records:
        filtered = filtered.head(max_records)
    return filtered


def summarize_homeless_shelters(location_label: str) -> Tuple[str, List[Dict[str, Any]]]:
    df = load_homeless_dataframe()
    if df.empty:
        return "", []
    mask = build_location_mask(
        df,
        location_label,
        ["ADDRESS", "NAME", "CITY", "STATE", "NGHBRHD"],
    )
    filtered = df[mask] if mask.any() else df
    total = len(filtered)
    total_beds = (
        filtered.get("BEDS_TOT") if "BEDS_TOT" in filtered.columns else pd.Series(dtype=float)
    )
    bed_sum = int(total_beds.sum()) if not total_beds.empty else None
    lines = [f"Homeless shelters matching filters: {total}"]
    if bed_sum is not None:
        lines.append(f"Total reported beds: {bed_sum}")
    sample = filtered.head(config.get("summary_include_samples", 5))
    for _, row in sample.iterrows():
        addr = row.get("ADDRESS") or row.get("NGHBRHD") or "unknown location"
        beds = row.get("BEDS_TOT")
        bed_info = f", beds {int(beds)}" if pd.notna(beds) else ""
        lines.append(f"- {row.get('NAME','Unnamed shelter')} at {addr}{bed_info}")

    records = []
    for _, row in filtered.iterrows():
        records.append(
            {
                "name": row.get("NAME"),
                "address": row.get("ADDRESS"),
                "beds_total": row.get("BEDS_TOT"),
                "population_type": row.get("POP_TYPE"),
            }
        )
    return "\n".join(lines), records


def summarize_universities(location_label: str) -> Tuple[str, List[Dict[str, Any]]]:
    df = load_university_dataframe()
    if df.empty:
        return "", []
    mask = build_location_mask(df, location_label, ["NAME", "ADDRESS"])
    filtered = df[mask] if mask.any() else df
    total = len(filtered)
    lines = [f"Universities/colleges matching filters: {total}"]
    sample = filtered.head(config.get("summary_include_samples", 5))
    for _, row in sample.iterrows():
        addr = row.get("ADDRESS") or "unknown location"
        lines.append(f"- {row.get('NAME','Unnamed institution')} at {addr}")

    records = []
    for _, row in filtered.iterrows():
        records.append(
            {
                "name": row.get("NAME"),
                "address": row.get("ADDRESS"),
            }
        )
    return "\n".join(lines), records


def build_context_from_incidents(
    incidents: pd.DataFrame,
    location_label: str,
    hour_filter: int | None,
    recent_days: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    if incidents.empty:
        text = (
            f"No matching incidents found for '{location_label}' "
            f"within the last {recent_days} days."
        )
        return text, []

    violent_mask = incidents["IncidentOffence"].str.upper().str.contains("|".join(VIOLENT_KEYWORDS), na=False)
    violent_count = int(violent_mask.sum())
    top_offenses = incidents["IncidentOffence"].str.upper().value_counts().head(5)
    recent_rows = incidents.head(config.get("summary_include_samples", 5))

    context_lines = [
        f"Time window: last {recent_days} days",
        f"Location filter: {location_label or 'any'}",
        f"Hour filter: {'>= ' + str(hour_filter) if hour_filter is not None else 'All hours'}",
        f"Total incidents matching filters: {len(incidents)}",
        f"Violent-keyword incidents: {violent_count}",
    ]
    if not top_offenses.empty:
        offense_line = ", ".join(f"{name.title()} ({count})" for name, count in top_offenses.items())
        context_lines.append(f"Top offences: {offense_line}")

    context_lines.append("Recent incidents:")
    for _, row in recent_rows.iterrows():
        ts = row["ArrestDateTime"]
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "unknown time"
        location = row["ArrestLocation"] or row["IncidentLocation"] or "unspecified location"
        context_lines.append(f"- {ts_str}: {row['IncidentOffence']} near {location}")

    max_chars = config.get("max_context_chars", 3000)
    context_text = "\n".join(context_lines)
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars]

    records: List[Dict[str, Any]] = []
    for _, row in incidents.iterrows():
        ts = row["ArrestDateTime"]
        timestamp = ts.timestamp() if pd.notna(ts) else None
        record_text = f"{row['IncidentOffence']} near {row['ArrestLocation'] or row['IncidentLocation']}"
        records.append(
            {
                "text": record_text,
                "incident_offence": row["IncidentOffence"],
                "arrest_location": row["ArrestLocation"],
                "incident_location": row["IncidentLocation"],
                "charge_description": row["ChargeDescription"],
                "timestamp": timestamp,
            }
        )
    return context_text, records


def infer_crime_type(incidents: List[Dict[str, Any]]) -> str:
    keywords = ("shooting", "assault", "robbery", "burglary", "theft", "arson", "carjacking")
    for record in incidents:
        text = " ".join(
            [
                record.get("incident_offence") or "",
                record.get("charge_description") or "",
                record.get("text") or "",
            ]
        ).lower()
        for keyword in keywords:
            if keyword in text:
                return keyword
    return "unknown"


def count_recent_incidents(incidents: List[Dict[str, Any]], window_hours: int) -> int:
    cutoff = datetime.now(timezone.utc).timestamp() - window_hours * 3600.0
    count = 0
    for record in incidents:
        ts_value = record.get("timestamp")
        if ts_value is not None and ts_value >= cutoff:
            count += 1
    return count


def score_severity(model_response: str | None) -> float:
    if not model_response:
        return 0.0
    severity = 0.0
    resp = model_response.lower()
    weights = (
        ("weapon", 0.3),
        ("gun", 0.3),
        ("violent", 0.2),
        ("fatal", 0.3),
        ("critical", 0.2),
        ("repeat", 0.2),
        ("escalating", 0.2),
    )
    for keyword, weight in weights:
        if keyword in resp:
            severity += weight
    return float(min(severity, 1.0))


def extract_features_from_summary(query: str, incidents: List[Dict[str, Any]], model_response: str | None):
    window_hours = config.get("incident_recency_hours", 24)
    return {
        "location": extract_location(query),
        "hour_of_day": extract_hour(query) or datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "crime_type": infer_crime_type(incidents),
        "incident_count_last_24h": count_recent_incidents(incidents, window_hours),
        "severity_score": score_severity(model_response),
    }


def vectorize_features(feature_dict: Dict[str, Any]) -> np.ndarray | None:
    global feature_vectorizer
    if feature_vectorizer is None:
        return None
    if not hasattr(feature_vectorizer, "feature_names_") or not getattr(feature_vectorizer, "feature_names_", []):
        print(
            "Info: Feature vectorizer is not fitted; fitting on the current sample. "
            "For consistent predictions, provide a pre-fitted vectorizer.",
            file=sys.stderr,
        )
        feature_vectorizer.fit([feature_dict])
    vector = feature_vectorizer.transform([feature_dict])
    return np.array(vector).astype(np.float32)[0]


def predict_hotspot_from_features(feature_dict: Dict[str, Any]) -> Tuple[bool | None, float | None]:
    if hotspot_model is None:
        return None, None
    vector = vectorize_features(feature_dict)
    if vector is None:
        return None, None
    vector = vector.reshape(1, -1)
    prediction = bool(hotspot_model.predict(vector)[0])
    risk_score = None
    if hasattr(hotspot_model, "predict_proba"):
        try:
            risk_score = float(hotspot_model.predict_proba(vector)[0][1])
        except Exception:
            risk_score = None
    return prediction, risk_score


def format_prompt(context: str, query: str) -> str:
    return f"""[INST] <<SYS>>
You are a helpful assistant analyzing crime and school data to identify safety risks and hotspots.
<</SYS>>

Context:
{context}

Question:
{query}
[/INST]"""


def run_gemini_inference(prompt: str, stream: bool | None = None) -> str:
    stream = config.get("gemini_stream", True) if stream is None else stream
    max_chars = config.get("gemini_max_prompt_chars", 6000)
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]
    try:
        if stream:
            print("Streaming response:\n", flush=True, file=sys.stderr)
            chunks: List[str] = []
            response = gemini_model.generate_content(prompt, stream=True)
            for chunk in response:
                if getattr(chunk, "text", None):
                    print(chunk.text, end="", flush=True, file=sys.stderr)
                    chunks.append(chunk.text)
            print("\n", file=sys.stderr)
            return "".join(chunks).strip()
        response = gemini_model.generate_content(prompt, stream=False)
        if getattr(response, "text", None):
            return response.text.strip()
        parts: List[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
                parts.append(getattr(part, "text", "") or "")
        return "".join(parts).strip()
    except Exception as exc:
        print("Gemini inference error:", exc, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


def retrieve_context(query: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    df = load_incident_dataframe()
    location_label = extract_location(query)
    hour_filter = extract_hour(query)
    recent_days = config.get("summary_recent_days", 30)
    max_records = config.get("summary_max_records", 50)
    incidents_df = filter_incidents_for_query(df, location_label, hour_filter, recent_days, max_records)
    context_text, incident_records = build_context_from_incidents(
        incidents_df,
        location_label,
        hour_filter,
        recent_days,
    )
    homeless_text, homeless_records = summarize_homeless_shelters(location_label)
    universities_text, university_records = summarize_universities(location_label)

    sections = [context_text] if context_text else []
    if homeless_text:
        sections.append("Homeless shelter snapshot:\n" + homeless_text)
    if universities_text:
        sections.append("Nearby universities snapshot:\n" + universities_text)
    combined_context = "\n\n".join(sections) if sections else "No matching data found for the provided filters."

    extras = {
        "homeless_shelters": homeless_records,
        "universities": university_records,
        "location_label": location_label,
        "hour_filter": hour_filter,
    }
    return combined_context, incident_records, extras


def save_output(
    prompt: str,
    incidents: List[Dict[str, Any]],
    response: str,
    output_dir: str = "outputs",
    analysis: Dict[str, Any] | None = None,
    extras: Dict[str, Any] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    resp_path = Path(output_dir) / f"response_{timestamp}.txt"
    resp_path.write_text(response or "", encoding="utf-8")

    incidents_path = Path(output_dir) / f"incidents_{timestamp}.json"
    incidents_path.write_text(json.dumps(incidents, indent=2), encoding="utf-8")

    if analysis is not None:
        analysis_path = Path(output_dir) / f"analysis_{timestamp}.json"
        analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    if extras:
        extras_path = Path(output_dir) / f"extras_{timestamp}.json"
        extras_path.write_text(json.dumps(extras, indent=2), encoding="utf-8")


def run_rag_pipeline(cfg: Dict[str, Any]):
    context, incidents, extras = retrieve_context(cfg["query"])
    prompt = format_prompt(context, cfg["query"])

    response = run_gemini_inference(
        prompt,
        stream=cfg.get("gemini_stream", config.get("gemini_stream", True)),
    )

    hotspot_analysis = {
        "enabled": bool(cfg.get("enable_hotspot_prediction", True) and hotspot_model is not None),
        "features": None,
        "is_hotspot": None,
        "risk_score": None,
    }
    if hotspot_analysis["enabled"]:
        features = extract_features_from_summary(cfg["query"], incidents, response)
        prediction, risk = predict_hotspot_from_features(features)
        hotspot_analysis.update(
            {
                "features": features,
                "is_hotspot": prediction,
                "risk_score": risk,
            }
        )
        print(
            f"Hotspot prediction: {prediction} with risk score {risk if risk is not None else 'n/a'}",
            file=sys.stderr,
        )

    if cfg.get("save_outputs", True):
        save_output(
            prompt,
            incidents,
            response,
            output_dir=cfg.get("output_dir", "outputs"),
            analysis=hotspot_analysis if hotspot_analysis.get("features") else None,
            extras=extras,
        )

    return prompt, incidents, response, hotspot_analysis, extras


def main():
    parser = argparse.ArgumentParser(description="Generate Gemini analysis over crime incidents.")
    parser.add_argument("--query", type=str, help="Query describing the safety question.")
    parser.add_argument("--skip-save", action="store_true", help="Prevent saving outputs to disk.")
    args = parser.parse_args()

    if args.query:
        config["query"] = args.query
    if args.skip_save:
        config["save_outputs"] = False

    try:
        prompt, incidents, response, hotspot_analysis, extras = run_rag_pipeline(config)
        output = {
            "prompt": prompt,
            "incidents": incidents,
            "response": response,
            "hotspot_analysis": hotspot_analysis,
            "extra_summaries": extras,
        }
        print(json.dumps(output, ensure_ascii=False))
    except Exception:
        print("Python error:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
