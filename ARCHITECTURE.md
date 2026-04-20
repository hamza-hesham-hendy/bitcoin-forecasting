# Code Architecture

A function-by-function walkthrough of `app.py`.

---

## High-Level Structure

```
app.py
│
├── Imports & page config
│
├── Helper / Data functions
│   ├── extract_csv_from_zip()
│   ├── detect_date_column()
│   ├── detect_price_columns()
│   ├── validate_and_prepare_data()
│   └── calculate_moving_average()
│
├── Prophet functions
│   └── train_prophet_model()
│
├── ARIMA functions
│   └── train_arima_model()
│
├── LSTM functions
│   ├── create_lstm_sequences()
│   ├── build_lstm_model()
│   ├── train_lstm_model()
│   └── calculate_lstm_confidence_interval()
│
├── Hybrid functions
│   └── train_hybrid_model()
│
├── Visualisation & metrics
│   ├── calculate_metrics()
│   └── create_forecast_plot()
│
└── main()           ← Streamlit UI + orchestration
```

---

## Data Layer

### `extract_csv_from_zip(zip_file)`
Opens a ZIP archive and reads the first `.csv` it finds (skipping macOS `__MACOSX` metadata). Returns `(DataFrame, filename)` or `(None, error_string)`.

### `detect_date_column(df)`
Three-tier detection:
1. Exact name match against a list of common names (`Date`, `Timestamp`, …)
2. Fuzzy match — column name contains `date`, `time`, or `timestamp`
3. Type inference — try `pd.to_datetime()` on each column

### `detect_price_columns(df)`
Returns all columns whose names contain `open`, `high`, `low`, `close`, or `price` (case-insensitive). The user then picks one from a selectbox.

### `validate_and_prepare_data(df, date_col, price_col)`
1. Parses the date column, trying string format → UNIX seconds → UNIX milliseconds in order.
2. Sorts ascending (oldest first).
3. Keeps only the two relevant columns.
4. Forward-fills then backward-fills missing prices; drops any remaining NaN rows.

### `calculate_moving_average(df, price_col, window)`
Rolling mean with `min_periods=1` so the chart line starts from day 1 rather than showing NaN for the first `window - 1` days.

---

## Prophet

### `train_prophet_model(train_data, date_col, price_col, forecast_days, confidence_interval)`
Prophet requires columns named exactly `ds` (datestamp) and `y` (value). The model is initialised with daily, weekly, and yearly seasonality enabled — all relevant for a 24/7 market. `interval_width` maps directly to the chosen confidence level. `make_future_dataframe` extends the timeline by `forecast_days` calendar days, and `predict` returns a DataFrame with columns `yhat`, `yhat_lower`, `yhat_upper`.

---

## ARIMA

### `train_arima_model(train_data, price_col, forecast_days, confidence_level)`
`auto_arima` from `pmdarima` performs a stepwise AIC search over (p, 0–2, q) with p, q ≤ 5. The best order is handed to `statsmodels.ARIMA` for final fitting. `get_forecast` returns point forecasts and confidence intervals; `alpha = 1 − confidence_level` translates the UI percentage to the statistical parameter.

---

## LSTM

### `create_lstm_sequences(data, lookback_window)`
Converts a 1-D scaled price array into supervised learning pairs:

```
Input sequence             Target
[t-60, t-59, …, t-1]  →   t
[t-59, t-58, …,  t ]  →   t+1
…
```

Output shapes: `X` is `(n_samples, lookback, 1)`, `y` is `(n_samples,)`.

### `build_lstm_model(lookback_window)`
```
LSTM(128, return_sequences=True)  → learns short-term patterns
Dropout(0.3)
LSTM(64,  return_sequences=False) → distils into a single vector
Dropout(0.3)
Dense(32, activation='relu')
Dense(1)                          → scalar price prediction
```
Compiled with Adam optimiser and MSE loss.

### `train_lstm_model(train_data, price_col, forecast_days, lookback_window=60)`
1. Scales prices to [0, 1] with `MinMaxScaler`.
2. Creates sequences with `create_lstm_sequences`.
3. Trains for up to 50 epochs with early stopping (patience = 10).
4. Generates the multi-step forecast autoregressively: predict day 1, append to the input window, predict day 2, and so on.
5. Inverse-transforms predictions back to USD.

### `calculate_lstm_confidence_interval(forecast_values, historical_errors, confidence_level)`
LSTM does not produce uncertainty natively. The standard deviation of test-set errors is used as a proxy. Bounds are:

```
margin = z_score × std(errors)
lower  = forecast − margin
upper  = forecast + margin
```

where `z_score` is the normal quantile for the chosen confidence level (e.g. 1.28 for 80 %).

---

## Hybrid (Prophet + LSTM)

### `train_hybrid_model(df_clean, date_col, price_col, forecast_days, lookback_window)`
Three phases:

**Phase 1 — Global trend with Prophet.**  
Fit Prophet on all training data. Compute residuals = actual − Prophet fitted values.

**Phase 2 — Local residuals with LSTM.**  
Scale residuals to [−1, 1]. Create sequences. Train a separate LSTM to predict future residuals autoregressively.

**Phase 3 — Combine.**  
`hybrid_forecast = Prophet_future_trend + LSTM_future_residuals`

This lets Prophet handle the macro structure while LSTM corrects systematic errors in the trend.

---

## Visualisation

### `calculate_metrics(actual, predicted)`
Returns `(MAE, RMSE)`. MAPE is computed inline in `main()` as `MAE / mean(actual) × 100`.

### `create_forecast_plot(...)`
Plotly traces are stacked in this order (important for `fill='tonexty'` to work correctly):

1. Historical price — solid blue line
2. Forecast (`yhat`) — dashed coloured line
3. Upper confidence bound — invisible line (width = 0)
4. Lower confidence bound — fills to the previous trace, creating the shaded band

A vertical `add_vline` marks the training/test split date. `hovermode='x unified'` shows all trace values at the cursor position.

Model colour scheme:
| Model | Colour |
|---|---|
| Prophet | Orange (#ff7f0e) |
| ARIMA | Orange (#ff7f0e) |
| LSTM | Green (#2ca02c) |
| Hybrid | Purple (#9467bd) |

---

## `main()` — Streamlit Orchestration

The sidebar collects all configuration in numbered sections:

1. File upload (CSV or ZIP)
2. Price column selector
3. Training date range (`start_date`, `end_date`)
4. Model choice (radio)
5. Forecast horizon (slider, 7–90 days)
6. Confidence interval (slider, 80–95 %)
7. LSTM lookback window (slider, 30–120 days) — shown only when an LSTM model is selected
8. Moving average toggle

**Train/Test Split Logic**

```python
test_size   = min(forecast_days, len(df_clean) - 100)
split_index = len(df_clean) - test_size
train_data  = df_clean[:split_index]
test_data   = df_clean[split_index:]
```

Guard rails: at least 100 rows must remain for training; if an LSTM model is chosen the lookback window must not exceed the training row count.

**Model dispatch** — a simple `if/elif` block calls the relevant training function and normalises the result into a common `forecast` DataFrame with columns `ds`, `yhat`, `yhat_lower`, `yhat_upper` before passing it to the plotting function.

---

## Design Decisions

**Why dynamic test size instead of fixed 80/20?**  
The test window matches the forecast horizon, so reported metrics (MAE, RMSE, MAPE) directly answer "how accurate is a N-day forecast?" rather than measuring accuracy on an arbitrarily sized window.

**Why MinMaxScaler for LSTM?**  
LSTMs with sigmoid/tanh activations saturate on large raw values. Scaling to [0, 1] keeps gradients healthy. The scaler is fit only on training data and applied to test data to prevent data leakage.

**Why autoregressive (one-step-at-a-time) multi-step forecasting?**  
Direct multi-output forecasting would require retraining for each horizon length. Autoregressive forecasting reuses a single model and naturally handles any horizon, at the cost of error accumulation over longer horizons.

**Why Dropout 30 % instead of the more common 20 %?**  
Bitcoin is extremely noisy. Higher dropout reduces the risk of the LSTM memorising short-term noise patterns rather than learning generalizable structure.
