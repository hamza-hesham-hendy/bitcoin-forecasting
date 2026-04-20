# 📚 Code Architecture & Design Decisions
## Deep Dive into the Bitcoin Forecasting Portal

---

## 🏗️ Application Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  • Streamlit components (sidebar, main area, metrics)       │
│  • File uploader, sliders, buttons                          │
│  • Plotly chart rendering                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                      │
│  • CSV parsing and validation                               │
│  • Column detection (dates, prices)                         │
│  • Data cleaning (sorting, missing values)                  │
│  • Train/test splitting                                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Forecasting Layer                          │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │   Prophet       │        │     ARIMA       │            │
│  │  Model          │        │   Model         │            │
│  └─────────────────┘        └─────────────────┘            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Visualization Layer                         │
│  • Plotly interactive charts                                │
│  • Metrics calculation (MAE, RMSE, MAPE)                    │
│  • Confidence interval plotting                             │
│  • CSV export functionality                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Code Breakdown by Function

### 1. Data Detection Functions

#### `detect_date_column(df)`
**Purpose**: Automatically find the date/timestamp column in uploaded CSV

**Strategy (3-tier detection)**:
1. **Exact Match**: Check for common names (Date, Timestamp, Time)
2. **Fuzzy Match**: Look for columns containing 'date', 'time', 'timestamp'
3. **Type Detection**: Try to parse each column as datetime

**Why this approach?**
- Kaggle datasets use inconsistent naming
- Some use "Date", others use "Timestamp" or "time"
- Robust to variations in capitalization

**Example**:
```python
# Works with all these column names:
"Date", "date", "Timestamp", "timestamp", "DateTime", "time_stamp"
```

---

#### `detect_price_columns(df)`
**Purpose**: Find all available price columns (Open, High, Low, Close, Price)

**Strategy**: Search for keywords in column names
- Keywords: ['open', 'high', 'low', 'close', 'price']
- Case-insensitive matching
- Returns all matches for user selection

**Why multiple columns?**
- Different analyses need different prices
- Close: Most common for forecasting (end-of-day settled price)
- Open: Useful for intraday analysis
- High/Low: Volatility studies

---

### 2. Data Preparation Functions

#### `validate_and_prepare_data(df, date_col, price_col)`
**Purpose**: Clean and validate data before forecasting

**Steps**:
1. **Convert dates**: `pd.to_datetime()` handles various date formats
2. **Sort chronologically**: Oldest to newest (required for time-series)
3. **Extract relevant columns**: Only keep date and price
4. **Handle missing values**: 
   - Forward fill: Use last known price
   - Backward fill: Fill from future if forward fails
   - Drop remaining NaN: Last resort

**Why this cleaning pipeline?**
```python
# Before cleaning:
Date: 2024-01-03, Price: 42000
Date: 2024-01-01, Price: NaN    # Out of order + missing
Date: 2024-01-02, Price: 41500

# After cleaning:
Date: 2024-01-01, Price: 41500  # Filled from next day (backfill)
Date: 2024-01-02, Price: 41500  # Sorted chronologically
Date: 2024-01-03, Price: 42000
```

**Design Decision**: Forward fill is preferred for crypto
- Price tends to persist (no gaps in 24/7 trading)
- Better than dropping rows (preserves more data)
- Backward fill only as fallback for start-of-series gaps

---

### 3. Technical Indicators

#### `calculate_moving_average(df, price_col, window)`
**Purpose**: Add Simple Moving Average (SMA) for context

**Formula**:
```
SMA = (P1 + P2 + ... + Pn) / n
```

**Implementation**:
```python
df[price_col].rolling(window=window, min_periods=1).mean()
```

**Why min_periods=1?**
- Early periods don't have full 30 days of history
- Instead of NaN, use available data (1 day, 2 days, etc.)
- Creates smooth transition from start

**Use Case**: Visual context on chart
- Shows medium-term trend
- Helps identify if price is above/below average
- Common in technical analysis

---

### 4. Prophet Model Implementation

#### `train_prophet_model(train_data, date_col, price_col, forecast_days, confidence_interval)`

**Prophet Requirements**:
- Column names MUST be 'ds' (datestamp) and 'y' (value)
- This is Facebook's API design choice

**Key Parameters**:

1. **`interval_width`**: Controls confidence bands
   ```python
   interval_width=0.8  # 80% confidence interval
   # Means: 80% chance actual price falls in shaded area
   ```

2. **`daily_seasonality`**: Captures day-of-week patterns
   - Bitcoin trades 24/7
   - Monday vs Sunday behavior differs
   - Prophet auto-detects these patterns

3. **`weekly_seasonality`**: Captures weekly cycles
   - Weekend vs weekday effects
   - Trading volume differences

4. **`yearly_seasonality`**: Long-term patterns
   - Bull/bear market cycles
   - Regulatory events
   - Halving events (every 4 years)

**Workflow**:
```python
# 1. Convert to Prophet format
prophet_df = pd.DataFrame({
    'ds': dates,  # Must be named 'ds'
    'y': prices   # Must be named 'y'
})

# 2. Initialize and fit
model = Prophet(interval_width=0.8, ...)
model.fit(prophet_df)

# 3. Create future dates
future = model.make_future_dataframe(periods=30, freq='D')

# 4. Predict
forecast = model.predict(future)
# Returns: yhat (prediction), yhat_lower, yhat_upper
```

**Why Prophet for Bitcoin?**
- ✅ Handles non-stationary data (trend changes)
- ✅ Robust to outliers (crypto has extreme volatility)
- ✅ Built-in uncertainty quantification
- ✅ No manual parameter tuning needed
- ✅ Captures multiple seasonalities

---

### 5. ARIMA Model Implementation

#### `train_arima_model(train_data, price_col, forecast_days, confidence_level)`

**ARIMA Components** (p, d, q):
- **p**: AutoRegressive order (how many past values to use)
- **d**: Differencing order (how many times to difference data)
- **q**: Moving Average order (how many past errors to use)

**Why Auto-ARIMA?**
```python
auto_model = auto_arima(
    y,
    seasonal=False,      # Bitcoin doesn't have strict seasons like retail
    stepwise=True,       # Faster search (greedy algorithm)
    max_p=5, max_q=5,   # Limit search space
    max_d=2             # Usually don't need more than 2nd differencing
)
```

**Search Process**:
1. Tests combinations: (0,0,0), (1,0,0), (0,1,0), etc.
2. Calculates AIC (Akaike Information Criterion) for each
3. Returns best model (lowest AIC)

**Example**:
```
Testing: (1,1,1) → AIC: 5000.2
Testing: (2,1,1) → AIC: 4998.5  ← Best so far
Testing: (2,1,2) → AIC: 4999.1
...
Final: (2,1,1) selected
```

**Confidence Intervals**:
```python
# alpha = 1 - confidence_level
# 80% confidence → alpha = 0.2
# 95% confidence → alpha = 0.05

forecast_summary = model.get_forecast(steps=30, alpha=0.2)
conf_int = forecast_summary.conf_int()
# Returns: lower bound and upper bound
```

**Why ARIMA for Comparison?**
- ✅ Classical benchmark (understand improvements)
- ✅ Interpretable parameters
- ✅ Good for stationary data
- ❌ Struggles with crypto volatility
- ❌ Assumes data is stationary (crypto isn't)

---

### 6. Metrics Calculation

#### `calculate_metrics(actual, predicted)`

**Mean Absolute Error (MAE)**:
```
MAE = (|actual₁ - predicted₁| + |actual₂ - predicted₂| + ... + |actualₙ - predictedₙ|) / n
```

**Interpretation**:
- "On average, predictions are off by $X"
- Units: Same as price (USD)
- Example: MAE = $500 means average error is $500

**Root Mean Squared Error (RMSE)**:
```
RMSE = √[(actual₁ - predicted₁)² + ... + (actualₙ - predictedₙ)²] / n
```

**Why use both?**
```python
# Example predictions:
Actual:     [100, 100, 100, 100]
Model A:    [90,  90,  90,  90]   # Consistent error
Model B:    [100, 100, 100, 60]   # One big error

MAE:
Model A: (10+10+10+10)/4 = 10
Model B: (0+0+0+40)/4 = 10       # Same MAE!

RMSE:
Model A: √(10²+10²+10²+10²)/4 = 10
Model B: √(0+0+0+40²)/4 = 20     # Different RMSE!

# RMSE penalizes large errors more heavily
```

**Use Cases**:
- **MAE**: Better for understanding average error
- **RMSE**: Better when large errors are very bad (risk management)
- **Both**: Gives complete picture of model performance

---

### 7. Visualization

#### `create_forecast_plot(...)`

**Plotly Trace Layers** (Order matters!):

1. **Historical Price** (Blue solid line)
   ```python
   go.Scatter(
       x=dates,
       y=prices,
       mode='lines',
       line=dict(color='#1f77b4', width=2)
   )
   ```

2. **Forecast** (Orange dashed line)
   ```python
   go.Scatter(
       x=future_dates,
       y=predictions,
       line=dict(color='#ff7f0e', dash='dash')
   )
   ```

3. **Confidence Interval** (Shaded area)
   ```python
   # Upper bound (invisible line)
   go.Scatter(y=upper_bound, line=dict(width=0))
   
   # Lower bound (filled to upper)
   go.Scatter(
       y=lower_bound,
       fill='tonexty',  # Fill to previous trace
       fillcolor='rgba(255, 127, 14, 0.2)'  # Transparent orange
   )
   ```

**Why this layer order?**
- Historical first: Forms the baseline
- Forecast next: Extends the timeline
- Confidence last: Adds context without obscuring data

**Interactive Features**:
```python
hovermode='x unified'  # Show all values at cursor position
template='plotly_white'  # Clean background
height=600  # Consistent sizing
```

**Vertical Line for Split**:
```python
fig.add_vline(
    x=split_date,
    line_dash="dash",
    annotation_text="Forecast Start"
)
# Clearly marks where prediction begins
```

---

## 🎯 Design Decisions Explained

### 1. Train/Test Split: Why 80/20?

**Standard Practice**:
- 80% train, 20% test is ML convention
- Balances: enough data to train, enough to validate

**For Time-Series** (Important!):
```python
# WRONG: Random split
train = df.sample(frac=0.8)  # ❌ Violates time order

# RIGHT: Sequential split
split_idx = int(len(df) * 0.8)
train = df[:split_idx]  # First 80%
test = df[split_idx:]   # Last 20%
```

**Why Sequential?**
- Time-series has temporal dependency
- Can't train on future to predict past
- Mimics real-world: use past to predict future

**Alternative Splits**:
- 70/30: More validation data, less training
- 90/10: More training, less validation
- 80/20: Good balance for most cases

---

### 2. Confidence Intervals: 80% vs 95%

**80% Confidence**:
```
|-------------------|  Range: Narrower
        ^
   Prediction
```
- 80% chance actual price falls in range
- Tighter bands, more precise
- Higher risk of actual price falling outside

**95% Confidence**:
```
|-------------------------|  Range: Wider
        ^
   Prediction
```
- 95% chance actual price falls in range
- Wider bands, more conservative
- Lower risk, but less informative

**Use Cases**:
- **80%**: General analysis, presentations
- **95%**: Risk management, conservative estimates

---

### 3. Model Choice: When to Use Which?

**Prophet**:
- ✅ Long-term forecasts (30+ days)
- ✅ Data with strong trends
- ✅ Multiple seasonalities
- ✅ Volatile markets (Bitcoin!)
- ❌ Very short-term (< 7 days)

**ARIMA**:
- ✅ Short-term forecasts (7-14 days)
- ✅ Stationary data
- ✅ Understanding baseline
- ✅ Educational comparison
- ❌ Long-term trends
- ❌ High volatility

**Bitcoin Specifically**:
- Prophet typically performs better
- Bitcoin has: trends, weekly patterns, volatility
- ARIMA struggles with non-stationarity

---

### 4. Error Handling Strategy

**Graceful Degradation**:
```python
try:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    return  # Stop execution, don't crash
```

**User Feedback Levels**:
1. **Info** 🔵: Normal operation ("Data split complete")
2. **Success** ✅: Successful action ("File uploaded")
3. **Warning** ⚠️: Potential issues ("Large file, may be slow")
4. **Error** ❌: Problems requiring action ("Invalid CSV format")

**Why Important?**
- Non-technical users need clear guidance
- Prevents frustration from cryptic errors
- Helps debugging

---

## 🧪 Testing & Validation

### Data Validation Checks

1. **File Format**:
   ```python
   try:
       df = pd.read_csv(uploaded_file)
   except:
       return "Invalid CSV"
   ```

2. **Column Detection**:
   ```python
   if date_col is None:
       return "No date column found"
   if not price_cols:
       return "No price columns found"
   ```

3. **Data Quality**:
   ```python
   # Check for sufficient data
   if len(df) < 100:
       warning("Dataset very small")
   
   # Check for missing values
   missing_pct = df[price_col].isna().mean()
   if missing_pct > 0.1:
       warning("More than 10% missing data")
   ```

---

## 🔧 Performance Optimizations

### 1. Streamlit Caching
```python
@st.cache_data
def load_data(file):
    return pd.read_csv(file)
# Prevents re-reading on every interaction
```

**Current Implementation**: No caching
**Future**: Could cache model training for same parameters

### 2. Data Sampling
For very large datasets:
```python
# Option: Resample to daily if minute-level data
if len(df) > 100000:
    df = df.resample('D').agg({...})
```

### 3. Model Training Speed
- Prophet: ~10-30 seconds (one-time compilation)
- ARIMA: ~20-60 seconds (parameter search)
- Subsequent runs: Faster (cached models)

---

## 📊 Future Enhancements

### Possible Additions:

1. **More Models**:
   - LSTM (Deep Learning)
   - XGBoost (Gradient Boosting)
   - Ensemble (combine multiple models)

2. **Additional Features**:
   - Volume analysis
   - Sentiment data integration
   - Multiple cryptocurrency comparison

3. **Advanced Metrics**:
   - Directional accuracy (% correct up/down)
   - Sharpe ratio (risk-adjusted returns)
   - Maximum drawdown

4. **UI Improvements**:
   - Side-by-side model comparison
   - Historical backtest visualization
   - Parameter tuning interface

5. **Export Options**:
   - PDF reports
   - PowerPoint slides
   - JSON API endpoint

---

## 🎓 Learning Resources

### Understanding the Code

**Key Concepts to Study**:
1. **Pandas**: Data manipulation (`merge`, `resample`, `rolling`)
2. **Plotly**: Interactive visualizations
3. **Streamlit**: Web app framework
4. **Prophet**: Facebook's forecasting library
5. **ARIMA**: Classical time-series

### Recommended Tutorials:
- [Streamlit Documentation](https://docs.streamlit.io)
- [Prophet Quick Start](https://facebook.github.io/prophet/docs/quick_start.html)
- [ARIMA Explained](https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arima.html)
- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)

---

## 🔍 Code Quality Checklist

✅ **Documentation**:
- Every function has docstring
- Complex logic has inline comments
- README explains high-level architecture

✅ **Error Handling**:
- Try/except blocks for file operations
- User-friendly error messages
- Graceful degradation

✅ **Code Organization**:
- Logical function grouping
- Clear naming conventions
- Consistent formatting

✅ **User Experience**:
- Progress indicators
- Helpful tooltips
- Clear visual feedback

✅ **Performance**:
- Efficient data processing
- Minimal redundant calculations
- Responsive UI updates

---

**This architecture supports the full assignment requirements while maintaining code clarity and educational value.**
