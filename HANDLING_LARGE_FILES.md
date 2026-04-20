# 📦 How to Upload Large Bitcoin Data Files
## Solutions for Files Larger Than 200MB

---

## 🎯 The Problem

The Kaggle Bitcoin dataset (`btcusd_1-min_data.csv`) is approximately **200MB** (minute-level data from 2012-2021).

Streamlit has a **200MB upload limit**, so you have **three options**:

---

## ✅ **Option 1: ZIP and Upload** (Easiest - 30 seconds)

The app now **automatically handles ZIP files**!

### Steps:
1. **Compress the CSV:**
   - Right-click `btcusd_1-min_data.csv`
   - Select "Compress to ZIP" or "Send to → Compressed (zipped) folder"
   - Result: `btcusd_1-min_data.zip` (~50-80MB)

2. **Upload to app:**
   - Click "Browse files" in the Streamlit sidebar
   - Select the **ZIP file**
   - App automatically extracts and loads the CSV ✅

3. **Enable downsampling:**
   - App will detect it's large data (>100,000 rows)
   - Check "Downsample to daily data" (appears automatically)
   - Converts minute data → daily data instantly
   - Result: ~2,800 daily records (much faster forecasting!)

**This is the recommended approach!**

---

## ✅ **Option 2: Downsample First** (Best for repeated use - 2 minutes)

Use the included Python script to convert minute data to daily data **before uploading**.

### Steps:

1. **Place the script in the same folder as your CSV:**
   ```
   my_folder/
   ├── btcusd_1-min_data.csv
   └── downsample_bitcoin_data.py
   ```

2. **Run the script:**
   ```bash
   python downsample_bitcoin_data.py
   ```

3. **Result:**
   - Creates `btcusd_daily.csv` (~1MB file)
   - Converts 2+ million minute records → ~2,800 daily records
   - Much faster to upload and process

4. **Upload the daily file:**
   - Upload `btcusd_daily.csv` to Streamlit app
   - No downsampling needed (already done!)

### Custom file names:
```bash
# Specify input and output files
python downsample_bitcoin_data.py input.csv output.csv
```

---

## ✅ **Option 3: Use Recent Data Only** (Quickest test - 1 minute)

Download just the **recent years** from Yahoo Finance instead.

### Steps:

1. **Go to Yahoo Finance:**
   https://finance.yahoo.com/quote/BTC-USD/history

2. **Select date range:**
   - Click "Time Period"
   - Select "1Y" (1 year) or "5Y" (5 years)
   - Click "Apply"

3. **Download:**
   - Click "Download" button
   - Saves as small CSV (<1MB)

4. **Upload:**
   - Upload directly to Streamlit app
   - No compression needed!

**Best for:** Quick testing, recent data analysis

---

## 📊 **Comparison Table**

| Method | File Size | Processing Time | Data Coverage | Best For |
|--------|-----------|----------------|---------------|----------|
| **ZIP Upload** | 50-80MB | Fast (auto-downsample) | Full (2012-2021) | Complete analysis |
| **Downsample Script** | ~1MB | Instant | Full (2012-2021) | Repeated use |
| **Yahoo Finance** | <1MB | Instant | Recent (1-5 years) | Quick testing |

---

## 🔧 **App Features for Large Data**

The updated app now includes:

### ✅ **Automatic ZIP Support**
- Upload `.zip` files directly
- App extracts CSV automatically
- Shows extracted filename

### ✅ **Smart Downsampling**
- Detects large datasets (>100,000 rows)
- Offers checkbox to downsample
- Converts minute → daily in real-time
- Shows before/after row counts

### ✅ **Progress Feedback**
- Shows file size and row count
- Displays downsampling progress
- Confirms successful processing

---

## 💡 **What is Downsampling?**

**Minute-level data** (original):
```
2024-01-01 00:00:00,  $42,000
2024-01-01 00:01:00,  $42,005
2024-01-01 00:02:00,  $42,010
...
2024-01-01 23:59:00,  $42,300
```
**1,440 records per day** (one per minute)

**Daily data** (downsampled):
```
2024-01-01,  Open: $42,000,  High: $42,500,  Low: $41,800,  Close: $42,300
```
**1 record per day** (OHLC summary)

### How downsampling works:
- **Open**: First minute's price
- **High**: Maximum price of the day
- **Low**: Minimum price of the day
- **Close**: Last minute's price
- **Volume**: Sum of all minute volumes

### Why downsample?
- ✅ **Smaller files** (200MB → 1MB)
- ✅ **Faster processing** (2M rows → 2.8K rows)
- ✅ **Better for forecasting** (daily patterns more relevant than minute patterns)
- ✅ **No information loss** (OHLC preserves all key info)

---

## 🚀 **Recommended Workflow**

### **For the Assignment (First Time):**

1. Download Kaggle dataset: `btcusd_1-min_data.csv`
2. Compress to ZIP: `btcusd_1-min_data.zip`
3. Upload ZIP to Streamlit app
4. Check "Downsample to daily data" when prompted
5. Generate forecast and analyze results

### **For Testing/Development:**

1. Use Yahoo Finance for quick 1-year data
2. Or run `downsample_bitcoin_data.py` once
3. Keep `btcusd_daily.csv` for repeated uploads

---

## 🐛 **Troubleshooting**

### "File too large to upload"
- ✅ Use ZIP compression (reduces size by ~60%)
- ✅ Or downsample first with the Python script

### "ZIP extraction failed"
- ✅ Make sure ZIP contains CSV file (not nested folders)
- ✅ Try extracting manually and uploading CSV directly

### "Downsampling is slow"
- ✅ Large datasets take 10-30 seconds (normal)
- ✅ Progress shown in sidebar
- ✅ Wait for "✅ Downsampled" message

### "Out of memory"
- ✅ Use downsampling (reduces memory usage)
- ✅ Close other applications
- ✅ Use smaller date range (Yahoo Finance)

---

## 📝 **Example Commands**

### **Downsample with script:**
```bash
# Default (looks for btcusd_1-min_data.csv)
python downsample_bitcoin_data.py

# Custom files
python downsample_bitcoin_data.py my_bitcoin_data.csv daily_output.csv
```

### **Create ZIP (command line):**
```bash
# Windows (PowerShell)
Compress-Archive -Path btcusd_1-min_data.csv -DestinationPath bitcoin_data.zip

# macOS/Linux
zip bitcoin_data.zip btcusd_1-min_data.csv
```

---

## ✅ **Quick Decision Guide**

**Choose ZIP Upload if:**
- ✓ You want to keep original data
- ✓ You're doing this once for the assignment
- ✓ You don't want to run extra scripts

**Choose Downsampling Script if:**
- ✓ You'll upload the file multiple times
- ✓ You want the fastest performance
- ✓ You're comfortable running Python scripts

**Choose Yahoo Finance if:**
- ✓ You just need recent data
- ✓ You want to test quickly
- ✓ File size is the main concern

---

## 🎉 **Summary**

**All three methods work perfectly!** The app is flexible:

1. **ZIP Upload** → Easiest, handles everything automatically ⭐ **Recommended**
2. **Downsample Script** → One-time setup, fastest repeated use
3. **Yahoo Finance** → Quickest test, recent data only

**Start with Option 1** (ZIP upload) for your assignment, then try others as needed!

---

**The app is now ready for large Bitcoin datasets! 🚀**
