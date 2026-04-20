# Quick Start Guide

## 1. Prerequisites

- **Python 3.10** (required — TensorFlow-CPU 2.13 supports 3.8–3.11, but 3.10 is the tested version)
- 4 GB+ RAM recommended
- Windows: VC++ Redistributable 2015–2022 x64 (usually already present)

---

## 2. Create and Activate a Virtual Environment

### Windows (Command Prompt or PowerShell)
```bash
py -3.10 -m venv bitcoin_forecast_env
bitcoin_forecast_env\Scripts\activate
```

### macOS / Linux
```bash
python3.10 -m venv bitcoin_forecast_env
source bitcoin_forecast_env/bin/activate
```

You should see `(bitcoin_forecast_env)` at the start of your prompt.

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes 5–10 minutes on the first run (TensorFlow is large).

---

## 4. Verify the Installation

```bash
python -c "import tensorflow as tf; print('TF', tf.__version__)"
python -c "from prophet import Prophet; print('Prophet OK')"
python -c "import streamlit; print('Streamlit', streamlit.__version__)"
```

All three should print without errors.

---

## 5. Run the App

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`. If it does not, open that URL manually.

---

## 6. Prepare Your Data

The app expects a Bitcoin historical CSV (or a ZIP containing one).  
The primary tested dataset is the Kaggle minute-level file (`btcusd_1-min_data.csv`).

**If the file is large (> 200 MB), zip it before uploading:**

```bash
# Windows PowerShell
Compress-Archive -Path btcusd_1-min_data.csv -DestinationPath bitcoin_data.zip

# macOS / Linux
zip bitcoin_data.zip btcusd_1-min_data.csv
```

The app will detect the large row count and offer automatic daily resampling in the sidebar.

Alternatively, run the offline downsampler first:
```bash
python downsample_bitcoin_data.py
# creates btcusd_daily.csv — upload that directly (no downsampling needed in-app)
```

---

## 7. Typical Workflow Inside the App

1. **Upload** your CSV or ZIP in the sidebar
2. **Select price column** — `Close` is recommended
3. **Set date range** — default is the last 3 years of available data
4. **Choose a model** — start with Prophet for speed
5. **Set forecast horizon** — e.g. 30 days
6. **Click Generate Forecast**
7. Review metrics, inspect the chart, and download the forecast CSV if needed

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on TensorFlow | Confirm `python --version` is 3.10.x |
| DLL load error on Windows | Re-install VC++ Redistributable and restart |
| Prophet compilation warning | Safe to ignore — Prophet still works correctly |
| App is slow on first model run | Normal; TensorFlow initialises on first use |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

---

## Daily Workflow

```bash
# Activate environment
bitcoin_forecast_env\Scripts\activate   # Windows
source bitcoin_forecast_env/bin/activate  # macOS/Linux

# Run app
streamlit run app.py

# Deactivate when done
deactivate
```
