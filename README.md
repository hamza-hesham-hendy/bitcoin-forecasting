# ₿ Bitcoin Price Forecasting Portal

An interactive Streamlit web app for time-series analysis and multi-model forecasting of Bitcoin prices.

---

## Overview

The app lets you upload a Bitcoin historical dataset (CSV or ZIP), pick a training date range, and run one of four forecasting models — all from a clean sidebar UI. It is designed as an educational project for the ITI AI Track time-series module.

---

## Dataset

This project was built and tested using the **Bitcoin Historical Data** dataset from Kaggle:

> 📦 [Bitcoin Historical Data (2012–2026) — Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

The dataset provides minute-level BTC/USD OHLCV data from January 2012 onwards. Because it is ~200 MB at minute resolution, it is recommended to either:
- **ZIP it before uploading** — the app auto-extracts and resamples to daily frequency, or
- **Run the offline downsampler first** — `python downsample_bitcoin_data.py` — then upload the resulting `btcusd_daily.csv` directly.

---

## Features

| Feature | Detail |
|---|---|
| **File upload** | CSV or ZIP up to 1 GB; auto-extracts the first CSV from a ZIP |
| **Auto-detection** | Date column and price columns are detected automatically |
| **Downsampling** | Minute/hourly data is offered for automatic daily resampling |
| **Date range selector** | Choose exactly which historical period to train on |
| **Four models** | Prophet · ARIMA · LSTM · Hybrid (Prophet + LSTM) |
| **Dynamic test set** | Test window = chosen forecast horizon (not a fixed 80/20 split) |
| **Confidence bands** | 80 % – 95 % selectable uncertainty intervals |
| **Metrics** | MAE, RMSE, MAPE reported on the held-out test window |
| **Interactive chart** | Plotly chart with zoom, pan, and unified hover |
| **CSV export** | One-click download of the forecast table |

---

## Models

### Prophet
Facebook's open-source forecasting library. Decomposes the series into trend, weekly seasonality, and yearly seasonality. Well-suited to Bitcoin's 24/7 trading and long bull/bear cycles. Handles missing days and outliers gracefully. Training time: ~10–30 s.

### ARIMA
Classical AutoRegressive Integrated Moving Average model. `auto_arima` from `pmdarima` searches for the best (p, d, q) order automatically using AIC. Serves as a statistical baseline. Training time: ~20–60 s.

### LSTM (Deep Learning)
Two-layer Long Short-Term Memory neural network built with Keras on TensorFlow-CPU. Learns non-linear temporal patterns from a sliding lookback window (configurable 30–120 days). Forecasting is done autoregressively — each predicted value feeds the next step. Confidence intervals are estimated from held-out prediction errors. Training time: ~60–120 s.

**Architecture:**
```
Input  (lookback × 1)
  └─ LSTM 128 units  + Dropout 30 %
      └─ LSTM 64 units  + Dropout 30 %
          └─ Dense 32 (ReLU)
              └─ Dense 1   ← price prediction
```

### Hybrid (Prophet + LSTM)
Prophet fits the global trend and seasonality first. The LSTM then learns the residuals — the portion Prophet got wrong. Final forecast = Prophet trend + LSTM residual correction. This typically outperforms either model alone on volatile series. Training time: ~90–150 s.

---

## Evaluation Strategy

The test window equals the chosen forecast horizon, so reported metrics directly measure the model's ability to forecast as far ahead as requested.

```
 Selected range:  start_date ─────────────────── end_date
                  │                             │
                  │←───── Training ────────────►│←─ Test ─►│
                                              (forecast_days)
```

Minimum training window: 100 days. The app will warn if the selected range is too short.

---

## Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Average absolute error in USD |
| **RMSE** | Like MAE but penalises large individual errors more heavily |
| **MAPE** | Error expressed as a percentage of the actual price |

---

## Project Structure

```
bitcoin-forecasting/
├── app.py                      # Main application (all four models)
├── requirements.txt            # Pinned dependencies — do not change
├── downsample_bitcoin_data.py  # Optional offline pre-processing script
├── README.md                   # This file
├── QUICKSTART.md               # Environment setup + running the app
└── ARCHITECTURE.md             # Code walkthrough and design decisions
```

---

## Supported Data Format

Any CSV with:
- A **date column** — detected from common names (`Date`, `Timestamp`, `time`, …) or by attempting to parse column values as datetime.
- At least one **price column** — detected from columns whose names contain `close`, `open`, `high`, `low`, or `price`.

For the Kaggle minute-level dataset, zip it first (reduces ~200 MB → ~60 MB). The app detects the large row count and offers automatic daily resampling — no pre-processing needed.

---
