# Scrolink Safety Assistant

Scrolink combines a Next.js frontend with a Python “data‑lake” backend to answer neighborhood safety questions for Baltimore. It blends three data sources—Baltimore Police arrests, homeless shelters, and nearby universities—with live ArcGIS incidents, a Gemini text summary, and a trained RandomForest hotspot model.

---

## Architecture Overview

```
Next.js (crimespot/)
│
├── app/api/query/route.ts  ──►  python backend/rag_pipeline.py
├── app/api/chat/route.ts   ──►  python backend/gemini_chat.py
│
├── app/Dashboard/page.tsx        Mapbox + ArcGIS search UI
│                                   ↳ stores latest analysis in ReportProvider
├── app/Reports/page.tsx           reads ReportProvider to render printable report
├── components/widgets/map.tsx     mapbox-gl view with color-coded markers
├── components/widgets/chat-box.tsx fetch-based Gemini chat widget
└── components/providers/report-provider.tsx
```

`backend/` holds all Python assets:

| File | Purpose |
| --- | --- |
| `rag_pipeline.py` | Loads arrests + shelters + universities, filters by query, summarizes with Gemini, and evaluates the RandomForest hotspot model. |
| `gemini_chat.py`  | Lightweight CLI bridge for chatbot requests. |
| `crime_hotspot_model.pkl` / `crime_feature_vectorizer.pkl` | RandomForest classifier + DictVectorizer trained on engineered features. |
| `scripts/*.csv` | Raw datasets (ignored in git by default; regenerate as needed). |
| `data/hotspot_training_features.csv` | Downsampled feature table used for training. |

---

## Getting Started

### 1. Requirements
- Node 18+ (Next 16)
- Python 3.11+ (tested with conda env `ragenv`)
- Mapbox account (for geocoding & tiles)
- Google Gemini API key

### 2. Install dependencies
```bash
cd crimespot/crimespot
npm install

# optional: create/activate a Python env first
pip install -r backend/requirements.txt   # or install pandas, numpy, scikit-learn, faiss-cpu, sentence-transformers, google-generativeai, joblib
```

### 3. Configure secrets
1. Copy `.env.local.example` (or create `.env.local`) with:
   ```
   NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN=pk....
   ```
2. Edit `backend/config.json` and set:
   ```json
   {
     "gemini_api_key": "YOUR_GEMINI_KEY",
     "incident_dataset_path": "backend/scripts/BPD_Arrests.csv",
     ...
   }
   ```
   The config provides defaults for summary windows, Gemini model (`models/gemini-2.5-flash`), etc.

### 4. Run everything
```bash
npm run dev
```
The Next dev server now handles API requests by spawning the backend scripts. Visit `http://localhost:3000` to use the Dashboard.

---

## Data Flow

1. **Search bar** (Dashboard):
   - Geocodes the user’s query with Mapbox.
   - Pulls recent incidents from ArcGIS (24h window, 500m radius).
   - Calls `/api/query` → `backend/rag_pipeline.py`, which:
     1. Filters the arrests dataset by location/hour windows.
     2. Builds textual context, including top offences, violent counts, and a few sample incidents.
     3. Summarizes nearby homeless shelters + universities.
     4. Calls Gemini with this structured prompt.
     5. Runs the RandomForest hotspot model (`crime_hotspot_model.pkl`) to get `is_hotspot` + `risk_score`.
2. **UI updates**:
   - Map markers show ArcGIS incidents with color-coded pins.
   - “Gemini Risk Analysis” card displays the narrative text.
   - “Hotspot Prediction” card shows the RandomForest result.
   - “Contextual Layers” lists the top shelters/universities.
   - The most recent analysis is cached in `ReportProvider`.
3. **Chat widget**:
   - Sends the latest incidents summary + user prompt to `/api/chat`, which shells into `backend/gemini_chat.py`.
   - Response is displayed as chat messages.
4. **Reports tab**:
   - Reads the cached analysis (query, summary, hotspot info, shelters, universities) and renders a printable “Latest Analysis” section.
   - Also lists any manual “Report Incident” submissions stored in localStorage.

---

## CLI helpers / scripts

The backend scripts can also run standalone for quick tests:

```bash
cd crimespot/crimespot
python backend/rag_pipeline.py --query "Patterson High School after 4 PM" --skip-save
python backend/gemini_chat.py --message "Summarize safety near Eastern Ave"
```

Training-related scripts live under `starting_over/` (archived work). The important ones:

| Script | Purpose |
| --- | --- |
| `scripts/build_hotspot_features.py` | Converts raw BPD arrests into engineered features (location/hour/day/crime type/recency/severity). |
| `scripts/train_hotspot_model.py` | Trains the RandomForest classifier + DictVectorizer and writes the `.pkl` files. |

The current repo ships with pre-trained `.pkl` files, but you can regenerate them by running the above scripts (CSV inputs reside in `backend/scripts/`).

---

## API Reference

| Endpoint | Method | Body | Description |
| --- | --- | --- | --- |
| `/api/query` | POST | `{ "query": "text" }` | Runs full pipeline, returns `{ prompt, response, incidents, hotspotAnalysis, extraSummaries }`. |
| `/api/chat` | POST | `{ "prompt": "text" }` | Sends prompt to Gemini chat helper and returns `{ response }`. |

These are the same routes the frontend calls, so they can be scripted via curl/Postman if desired.

---

## Testing / Demo Tips

1. **Dashboard** – Enter a location (e.g., “Patterson High School after 4 PM”) and click “Go”. The map, danger card, Gemini summary, and hotspot result should all update. The Reports tab will show the same summary after a refresh/navigation.
2. **Chat** – Ask follow-up questions (“Is it safe to walk there?”) so Gemini reuses the incident context.
3. **API sanity** – Use curl to hit `/api/query` & `/api/chat` while `npm run dev` is running. Responses should match what the UI shows.
4. **Map token check** – If you see a blank blue map or Mapbox errors, ensure `NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN` is set.

---

## Git Hygiene / Ignored Assets

The repo’s `.gitignore` includes detailed comments explaining why large datasets and model files are excluded. Reviewers can regenerate them locally by following the training instructions above. Runtime directories (`cache/`, `outputs/`, `sandbox/`) are ignored because they’re built on the fly.
we had to create a gitignore file due to a change of plans in our system. we attempted to create a rag system but we ended up editing the architecture to accomidate for time
---

## Future Work

- Support multiple cities by abstracting the datasets/config.
- Add more context layers (vacant buildings, schools) by dropping new CSVs into `backend/scripts/`.
- Host the backend on a serverless runtime (e.g., Vercel + AWS Lambda) so no local Python install is needed.

