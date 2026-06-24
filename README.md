# Energy Consumption Anomaly Profiler

An unsupervised machine learning project that clusters London households by their energy usage behavior and flags anomalous consumption patterns — without any labelled data.

---

## What it does

- Groups ~295 London households into behavioral archetypes (Normal Users, Heavy Users) using K-Means clustering
- Detects anomalous households within each group using Isolation Forest
- Flags suspicious patterns like faulty meters, vacant properties, and extreme consumption spikes
- Compares households against their own peer group — not the entire population

---

## Project structure

```
energy_anomaly_project/
├── notebooks/
│   └── energy_anomaly.ipynb       # Full Colab training notebook
├── models/
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── kmeans.pkl                 # Trained K-Means model (k=3)
│   ├── iso_forests.pkl            # Isolation Forest per cluster
│   └── feature_cols.json          # Feature order the model expects
├── backend/
│   └── main.py                    # FastAPI server (coming soon)
├── frontend/                      # React / HTML frontend (coming soon)
└── README.md
```

---

## Dataset

**Source:** [London Datastore — SmartMeter Energy Use Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/)

- Collected by UK Power Networks as part of the Low Carbon London project
- Half-hourly energy readings (kWh) per household
- Date range: November 2011 — February 2014
- ~5,567 households total across 168 CSV block files
- **Used in this project:** 10 blocks (~295 households, 10 million rows)

---

## ML pipeline

```
10 million raw readings
        ↓
   Feature engineering (Cell 4)
   — one behavioral profile per household —
        ↓
   StandardScaler (Cell 5)
   — normalize all features to same scale —
        ↓
   K-Means clustering (Cell 6 & 7)
   — find best k using silhouette score —
   — assign each household to a cluster —
        ↓
   Isolation Forest per cluster (Cell 8)
   — flag anomalies within each peer group —
        ↓
   15 anomalies flagged across 295 households
```

---

## Features engineered

| Feature | Description |
|---|---|
| `mean_kwh` | Average energy use per half-hour |
| `std_kwh` | Variability in usage |
| `max_kwh` | Highest single reading |
| `total_kwh` | Total consumption over full period |
| `morning_mean` | Average usage 6am–9am |
| `evening_mean` | Average usage 5pm–10pm |
| `night_mean` | Average usage midnight–5am |
| `weekend_mean` | Average usage on weekends |
| `weekday_mean` | Average usage on weekdays |
| `peak_ratio` | Max usage / mean usage — measures spikiness |
| `zero_rate` | Fraction of readings that are exactly zero |

---

## Clusters discovered

| Cluster | Name | Households | Avg kWh | Key characteristic |
|---|---|---|---|---|
| 0 | Normal Users | 234 | 0.157 | Typical London household, moderate usage |
| 1 | Heavy Users | 61 | 0.459 | Consistently high consumption |

> Note: A third cluster of 1 household (MAC000037) was discovered and merged into Cluster 0 — it had a peak ratio of 589 and 99.2% zero readings, making it the most anomalous household in the dataset.

---

## Anomalies flagged (15 total)

### Most significant findings

| Household | Cluster | Why flagged |
|---|---|---|
| MAC000037 | Normal | 99.2% zero readings, peak ratio 589 — faulty meter |
| MAC004454 | Normal | Peak ratio 66.4 — near dormant with extreme spikes |
| MAC004308 | Normal | Peak ratio 55.3 — same erratic pattern |
| MAC000004 | Normal | 76.7% zero readings — likely vacant property |
| MAC004488 | Heavy | Peak ratio 42.7 among heavy users — completely out of pattern |
| MAC000358 | Heavy | Mean 0.97 kWh — nearly 2× the heavy user average |

---

## Model performance

| Metric | Value | Meaning |
|---|---|---|
| Silhouette score (k=3) | 0.560 | Strong cluster separation (above 0.5 is good) |
| Silhouette score (k=2) | 0.552 | Close second |
| Contamination rate | 5% | ~15 households flagged per run |
| Anomaly rate | 5.1% | 15 out of 295 households flagged |

---

## How to run

### 1. Training (Google Colab)

Open `notebooks/energy_anomaly.ipynb` in Google Colab and run cells top to bottom.

You will need to upload block files manually:

```
From: https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/
Files: LCL-June2015v2_0.csv through LCL-June2015v2_9.csv (10 files)
```

At the end, 4 model files will download automatically:

```
scaler.pkl
kmeans.pkl
iso_forests.pkl
feature_cols.json
```

Place them in the `models/` folder.

### 2. Backend (coming soon)

```bash
cd backend
pip install fastapi uvicorn joblib scikit-learn pandas numpy
uvicorn main:app --reload
```

### 3. Frontend (coming soon)

Upload a household's CSV of readings → get cluster assignment + anomaly flag back.

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install all at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## Key findings

- **MAC000037** is the clearest anomaly in the dataset — 99.2% zero readings with occasional spikes strongly suggests a faulty meter or long-term vacant property
- **MAC000004 and MAC000016** were flagged in both the 30-household pilot run and the full 295-household run — consistent findings across different dataset sizes
- **Peer group comparison matters** — MAC004488 would not stand out globally, but within the Heavy Users cluster its peak ratio of 42.7 is dramatically unusual
- The unsupervised approach discovered a natural split between normal and heavy users without any predefined labels

---

## What's next

- [ ] FastAPI backend to serve predictions on new household data
- [ ] React frontend with CSV upload and anomaly report
- [ ] Deployment on Railway / Render
- [ ] Add more blocks (15–20) for more robust clustering
- [ ] Experiment with Gaussian Mixture Models for soft cluster assignments

---

## License

Dataset: [Open Government Licence v2.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/) — UK Power Networks / London Datastore

Code: MIT
