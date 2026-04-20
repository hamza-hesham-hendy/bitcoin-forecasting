# 🎯 Quick Start Guide - New Features

## 🆕 What's New in This Version

### 1. **Date Range Selection** ⭐
- Choose which period of historical data to use for training
- Default: Last 3 years (or all data if less)
- Helps exclude old data that may not be relevant

### 2. **Dynamic Test Set** ⭐  
- Test set size automatically matches forecast horizon
- If you want to forecast 30 days → test on last 30 days
- More realistic evaluation of forecasting ability

---

## 🚀 How to Use the New Features

### **Step 1: Upload Your Data** (unchanged)
- Click "Browse files" in sidebar
- Upload Bitcoin CSV or ZIP file

### **Step 2: Select Price Column** (unchanged)
- Choose which price to forecast (Close, Open, etc.)

### **Step 3: Select Training Data Period** ⭐ **NEW!**
```
You'll see:
📅 Available data: 2014-01-01 to 2024-12-31

Two date inputs:
┌─────────────────┬─────────────────┐
│ Start date:     │ End date:       │
│ 2021-12-31      │ 2024-12-31      │
└─────────────────┴─────────────────┘

📊 Selected: 1,096 days of data
```

**Tips:**
- **Recent data focus**: 2022-01-01 to 2024-12-31 (last 3 years)
- **Long-term analysis**: 2018-01-01 to 2024-12-31 (6+ years)
- **Short test**: 2024-01-01 to 2024-12-31 (1 year)

### **Step 4: Choose Model** (renumbered to 4)
- Prophet / ARIMA / LSTM
- Same as before

### **Step 5: Set Forecast Horizon** (renumbered to 5)
```
Days to forecast: [30] ←── This also sets test size!
```

**What happens:**
- If forecast = 30 days → Last 30 days used for testing
- If forecast = 60 days → Last 60 days used for testing
- Training uses everything before the test period

### **Step 6-8: Other Settings** (unchanged)
- Confidence interval
- LSTM parameters (if selected)
- Technical indicators

### **Step 9: Generate Forecast**
Click "🚀 Generate Forecast" and you'll see:

```
Training Setup:
- Selected period: 2022-01-01 to 2024-12-31 (1,096 days total)
- Training data: 2022-01-01 to 2024-12-01 (1,066 days)
- Test data (last 30 days): 2024-12-02 to 2024-12-31 (30 days)
- Forecast horizon: 30 days into future

Evaluation Strategy:
Model trained up to 2024-12-01, tested on last 30 days.
This evaluates the 30-day forecasting accuracy.
```

---

## 📊 Example Workflows

### **Workflow 1: Recent Market Focus (Recommended)**
```
1. Start date: 2022-01-01
2. End date: 2024-12-31
3. Forecast: 30 days
4. Model: Prophet or LSTM

Result: Focuses on last 3 years of market behavior
```

### **Workflow 2: Long-Term Analysis**
```
1. Start date: 2017-01-01
2. End date: 2024-12-31
3. Forecast: 60 days
4. Model: LSTM (needs more data)

Result: Learns from 8 years of history
```

### **Workflow 3: Quick Test**
```
1. Start date: 2024-01-01
2. End date: 2024-12-31
3. Forecast: 7 days
4. Model: Prophet (fastest)

Result: Quick test on recent data
```

### **Workflow 4: Comparing Periods**
```
Try these separately:
A) 2020-2022 (COVID period)
B) 2022-2024 (Recent period)

See which period's patterns work better!
```

---

## ⚠️ Important Notes

### **Minimum Data Requirements:**
- **At least 100 days** needed for training
- If forecast = 60 days, you need 160+ days total (100 train + 60 test)
- LSTM needs even more (lookback + 100 training days)

### **Test Size Adjustment:**
The app will automatically adjust test size if needed:
```
⚠️ Test size adjusted to 50 days (requested 60).
```
This happens when there's not enough data to maintain 100 training days.

### **LSTM Lookback Warning:**
If your lookback window is larger than training data:
```
❌ LSTM lookback window (60) > training data (50). Please reduce lookback.
```
Solution: Reduce lookback window or select longer date range.

---

## 🎯 Best Practices

### **For Best Results:**
1. **Use recent data:** Last 2-3 years usually works best
2. **Match forecast to test:** If forecasting 30 days, test on 30 days
3. **More data for LSTM:** LSTM needs 500+ days, prefer 1000+
4. **Prophet for quick tests:** Faster and works well on shorter data

### **For Learning:**
1. **Try different periods:** See how 2020 vs 2023 patterns differ
2. **Compare forecast horizons:** Try 7, 30, 60 days
3. **Compare models:** Prophet vs ARIMA vs LSTM on same period

### **For Real Analysis:**
1. **Use 3+ years** of recent data
2. **Test on realistic horizon:** Don't test on 7 days if you want 30-day forecasts
3. **Check metrics:** Lower MAE = better model

---

## 🔍 Understanding the Output

### **Training Setup Box:**
```
Selected period: 2022-01-01 to 2024-12-31 (1,096 days)
                 ↑                     ↑
            Your selection         Your selection

Training data: 2022-01-01 to 2024-12-01 (1,066 days)
               ↑                     ↑
          Start of range       End - forecast_days

Test data: 2024-12-02 to 2024-12-31 (30 days)
           ↑                     ↑
     Start of test         End of range
     (= End - forecast_days + 1)
```

### **Evaluation:**
- **MAE**: Average error in USD on test period
- **RMSE**: Error with more penalty for large mistakes
- **MAPE**: Error as % of price

Lower values = better model!

---

## 🐛 Troubleshooting

### **"Start date must be before end date!"**
- Make sure start date < end date

### **"Insufficient data after filtering!"**
- Select a longer date range
- Need at least 100 days total

### **"Insufficient training data!"**
- After reserving test days, not enough left for training
- Solutions:
  - Select longer date range
  - Reduce forecast horizon

### **"LSTM lookback window too large!"**
- Lookback > available training data
- Solutions:
  - Reduce lookback window slider
  - Select longer date range

---

## 📚 Quick Reference

| Setting | Old Behavior | New Behavior |
|---------|--------------|--------------|
| **Training Data** | All data (2014-2024) | Your selection (e.g., 2022-2024) |
| **Test Size** | 20% (fixed) | = forecast_days (dynamic) |
| **Evaluation** | Generic | Direct forecast evaluation |

---

## ✅ Checklist Before Running

- [ ] Date range selected (at least 100 days)
- [ ] Start date < End date
- [ ] Forecast horizon set
- [ ] Model selected
- [ ] For LSTM: Lookback < training data available

---

**You're ready to generate forecasts with the new features! 🚀**
