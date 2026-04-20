# 🚀 Quick Start Guide
## Get Your Bitcoin Forecasting App Running in 5 Minutes

---

## Step 1: Setup (2 minutes)

### Windows
```bash
# Open Command Prompt or PowerShell

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### macOS/Linux
```bash
# Open Terminal

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**⏳ Installation will take 3-5 minutes** (Prophet needs to compile)

---

## Step 2: Get Sample Data (1 minute)

### Option A: Quick Test (Recommended for First Run)
Use Yahoo Finance for quick 1-year data:

1. Visit: https://finance.yahoo.com/quote/BTC-USD/history
2. Click "Time Period" → Select "1Y" (1 year)
3. Click "Download"
4. Save as `bitcoin.csv` in your project folder

### Option B: Full Kaggle Dataset (Complete Analysis)

**The app now supports ZIP files!** If your CSV is too large (>200MB):

1. Go to: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
2. Download `btcusd_1-min_data.csv`
3. **Compress to ZIP**: Right-click → "Compress" or "Send to ZIP"
4. Upload the **ZIP file** to the app (it auto-extracts!)
5. Check "Downsample to daily data" when prompted

**See HANDLING_LARGE_FILES.md for detailed instructions!**

---

## Step 3: Run the App (30 seconds)

```bash
# Make sure virtual environment is activated
# You should see (venv) in your terminal

streamlit run app.py
```

**The app will automatically open in your browser at http://localhost:8501**

If it doesn't open automatically:
- Open your browser manually
- Go to: http://localhost:8501

---

## Step 4: Use the App (1 minute)

1. **Upload Data**
   - Click "Browse files" in the sidebar
   - Select your `bitcoin.csv` file
   - ✅ You'll see "File uploaded successfully"

2. **Configure Settings**
   - Model: Choose "Prophet" (recommended for first try)
   - Forecast Horizon: Set to "30" days
   - Confidence: Keep at "80%"

3. **Generate Forecast**
   - Click the big orange "🚀 Generate Forecast" button
   - Wait 10-20 seconds while the model trains
   - View your results!

---

## ✅ You're Done!

You should now see:
- 📊 Interactive chart with historical and forecasted prices
- 📈 Performance metrics (MAE, RMSE)
- 🔮 Forecast summary with price predictions
- 💾 Download button for forecast CSV

---

## 🎯 Next Steps

### Try Different Models
1. Change model to "ARIMA"
2. Click "Generate Forecast" again
3. Compare results with Prophet

### Experiment with Settings
- **Increase horizon** to 60 or 90 days (see longer forecasts)
- **Change confidence** to 95% (see wider uncertainty bands)
- **Enable Moving Average** checkbox (add technical indicator)

### Use Your Own Data
- Upload different Bitcoin datasets
- Try different price columns (Open, High, Low, Close)
- Test with different time periods

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
# Look for (venv) in your terminal

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Then run the app again
streamlit run app.py
```

### "Port 8501 already in use"
```bash
# Kill existing Streamlit process
# Then restart

# Or use different port:
streamlit run app.py --server.port 8502
```

### App is slow / stuck on "Training model..."
- This is normal for first run (30-60 seconds)
- Prophet needs to compile on first use
- ARIMA auto-search takes longer
- Subsequent runs will be faster

### Chart not showing
- Clear browser cache (Ctrl+Shift+Delete)
- Try a different browser
- Check browser console for errors (F12)

---

## 📝 Quick Reference

### Starting the App
```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Run app
streamlit run app.py
```

### Stopping the App
```bash
# Press Ctrl+C in the terminal where app is running
```

### Updating Dependencies
```bash
pip install --upgrade -r requirements.txt
```

---

## 💡 Tips for Best Results

1. **For Presentations**
   - Use 30-day forecast (good balance)
   - Keep confidence at 80%
   - Enable Moving Average for context

2. **For Learning**
   - Compare Prophet vs ARIMA on same data
   - Try different forecast horizons
   - Note how metrics change

3. **For Analysis**
   - Use longer historical data (3+ years)
   - Try both 80% and 95% confidence
   - Check forecast against actual recent prices

---

## 🎓 Understanding the Output

### Metrics
- **MAE** (Mean Absolute Error): Average $ difference between prediction and actual
  - Lower is better
  - Easy to interpret ($500 MAE = off by $500 on average)

- **RMSE** (Root Mean Squared Error): Emphasizes larger errors
  - Always ≥ MAE
  - Better for identifying when model makes big mistakes

- **MAPE** (Mean Absolute % Error): Error as percentage
  - Good for comparing across different price ranges
  - 5% MAPE = predictions off by 5% on average

### Chart Elements
- **Blue Line**: Historical actual prices
- **Orange Dashed Line**: Forecast predictions
- **Shaded Area**: Confidence interval (uncertainty range)
- **Red Vertical Line**: Where forecast begins (end of training data)

---

## 🎉 Success Checklist

After following this guide, you should be able to:
- [ ] Install all dependencies without errors
- [ ] Upload a Bitcoin CSV file
- [ ] Generate a forecast with Prophet
- [ ] View the interactive chart
- [ ] See performance metrics (MAE, RMSE)
- [ ] Download forecast as CSV
- [ ] Switch between Prophet and ARIMA
- [ ] Adjust forecast horizon and confidence

---

**🎊 Congratulations!** You now have a working Bitcoin forecasting application!

For detailed documentation, see **README.md**

For dataset information, see **DATASET_LINKS.md**

---

**Need Help?** 
- Check the full README.md
- Review error messages carefully
- Verify virtual environment is activated
- Ensure all dependencies installed correctly
