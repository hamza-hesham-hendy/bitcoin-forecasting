# Bitcoin Dataset Links
# ======================

## Recommended Datasets for Testing

### 1. Bitcoin Historical Data (Primary Recommendation)
**Link**: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
**Description**: Minute-level Bitcoin price data from 2012 to 2021
**Columns**: Timestamp, Open, High, Low, Close, Volume (BTC), Volume (Currency), Weighted Price
**Size**: ~4 million rows (can be resampled to daily)
**Format**: CSV
**Best For**: Complete historical analysis, long-term forecasting

**How to Use**:
1. Download the CSV from Kaggle
2. If the file is too large, you can:
   - Use only recent years (e.g., 2020-2024)
   - Resample to daily data using pandas:
     ```python
     df_daily = df.resample('D', on='Timestamp').agg({
         'Open': 'first',
         'High': 'max',
         'Low': 'min',
         'Close': 'last'
     })
     ```

---

### 2. Cryptocurrency Historical Prices
**Link**: https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory
**Description**: Daily cryptocurrency prices for multiple coins
**Columns**: Date, Open, High, Low, Close, Volume, Market Cap
**Format**: CSV (one file per coin)
**Best For**: Daily analysis, multiple coin comparison

**File to Use**: `coin_Bitcoin.csv`

---

### 3. Bitcoin Price Dataset (2014-2024)
**Link**: https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd
**Description**: Daily Bitcoin prices in USD
**Columns**: Date, Price, Open, High, Low, Vol., Change%
**Format**: CSV
**Best For**: Quick testing, smaller file size

---

## Alternative Data Sources (Non-Kaggle)

### Yahoo Finance
**URL**: https://finance.yahoo.com/quote/BTC-USD/history
**Ticker**: BTC-USD
**How to Download**:
1. Go to the Yahoo Finance link
2. Click "Historical Data"
3. Select date range
4. Click "Download" to get CSV

**Columns**: Date, Open, High, Low, Close, Adj Close, Volume

---

### CoinGecko API (Free)
**URL**: https://www.coingecko.com/en/api
**Documentation**: https://www.coingecko.com/api/documentation
**Format**: JSON (can be converted to CSV)
**Best For**: Most recent data, real-time updates

---

### CryptoCompare
**URL**: https://www.cryptocompare.com/
**Free Tier**: Yes (with rate limits)
**Format**: CSV/JSON
**Best For**: Multiple cryptocurrencies

---

## Sample Data Format

Your CSV should look like this:

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,42000.50,42500.00,41800.00,42300.00,1500000
2024-01-02,42300.00,43000.00,42100.00,42800.00,1600000
2024-01-03,42800.00,43500.00,42500.00,43200.00,1700000
...
```

**Minimum Requirements**:
- Date column (any name: Date, Timestamp, Time, etc.)
- At least one price column (Close, Open, High, or Low)
- Chronologically ordered (app will auto-sort if needed)

---

## Quick Test Dataset

If you want to test quickly without downloading large files:

### Option 1: Use Recent Yahoo Finance Data
```bash
# Download last 1 year of Bitcoin data
# Visit: https://finance.yahoo.com/quote/BTC-USD/history
# Set: 1 year range, Daily frequency
# Click: Download
```

### Option 2: Use Sample Data (Create Manually)
Create a simple CSV with ~100 rows of recent Bitcoin prices:
```csv
Date,Close
2024-01-01,42300
2024-01-02,42800
2024-01-03,43200
...
```

---

## Data Preparation Tips

1. **For Large Files** (>1 million rows):
   - Resample to daily data
   - Use only recent years (last 3-5 years)
   - Filter unnecessary columns

2. **For Missing Dates**:
   - App automatically handles gaps
   - Prophet fills missing values
   - No manual preprocessing needed

3. **For Multiple Price Columns**:
   - Use "Close" for end-of-day price (recommended)
   - Use "Open" for start-of-day analysis
   - Use "High"/"Low" for volatility studies

4. **Column Name Variations**:
   - The app auto-detects common variations
   - Supported: Date, date, Timestamp, timestamp, Time
   - Supported: Close, close, Price, price, Open, open, High, high, Low, low

---

## Testing Checklist

Before uploading to the app, verify:
- [ ] File is in CSV format (.csv extension)
- [ ] Has a date/timestamp column
- [ ] Has at least one price column
- [ ] No completely empty rows
- [ ] Dates are in consistent format (YYYY-MM-DD recommended)
- [ ] File size < 100MB (for faster loading)

---

**Note**: The app is tested with Kaggle's Bitcoin Historical Data but should work with any standard CSV following the format above.
