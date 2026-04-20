# ₿ Bitcoin Price Forecasting Portal

An interactive web application for time-series analysis and forecasting of Bitcoin prices using Streamlit, Prophet, ARIMA, and LSTM Deep Learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange)
![License](https://img.shields.io/badge/License-Educational-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Explanations](#model-explanations)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This Streamlit application provides a comprehensive platform for forecasting Bitcoin prices using advanced time-series models and deep learning. It's designed for:

- **Students**: Learning time-series forecasting techniques and deep learning
- **Analysts**: Quick prototyping and analysis of crypto trends
- **Researchers**: Comparing traditional and modern forecasting approaches

The app supports three different forecasting models with interactive visualizations and backtesting capabilities.

---

## ✨ Features

### Core Functionality
- ✅ **CSV/ZIP Upload**: Support for Kaggle-style Bitcoin historical data
- ✅ **Automatic Detection**: Smart detection of date and price columns
- ✅ **Three Models**: Prophet (traditional), ARIMA (classical), LSTM (deep learning)
- ✅ **Customizable Forecasts**: Adjust horizon (7-90 days) and confidence intervals (80-95%)
- ✅ **Backtesting**: Automatic train/test split with performance metrics
- ✅ **Interactive Charts**: Plotly-based visualizations with zoom, pan, and hover

### Forecasting Models

1. **Prophet** (Facebook's Forecasting Tool)
   - Handles seasonality automatically
   - Robust to missing data and outliers
   - Built-in uncertainty intervals
   
2. **ARIMA** (AutoRegressive Integrated Moving Average)
   - Classical statistical approach
   - Auto-parameter selection with `auto_arima`
   - Good baseline for comparison

3. **LSTM** (Long Short-Term Memory Neural Network) ⭐ **NEW!**
   - Deep learning model using TensorFlow/Keras
   - Learns complex temporal patterns
   - Best for long-term dependencies
   - Sliding window approach (configurable lookback period)

### Metrics & Evaluation
- **MAE** (Mean Absolute Error): Average prediction error in USD
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **MAPE** (Mean Absolute Percentage Error): Relative error percentage

---

## 🚀 Installation

### Prerequisites
- **Python 3.10** (required for TensorFlow 2.10)
- pip (Python package manager)
- 4GB+ RAM (recommended for LSTM training)
- **VC++ Redistributable** (Windows): Already installed ✅

### Step 1: Environment Setup

See **ENVIRONMENT_SETUP.md** for detailed instructions on:
- Deleting old environments
- Creating fresh Python 3.10 environment
- Installing dependencies correctly

**Quick version:**
```bash
# Create virtual environment
python -m venv bitcoin_forecast_env

# Activate it
# Windows:
bitcoin_forecast_env\Scripts\activate

# macOS/Linux:
source bitcoin_forecast_env/bin/activate
```

### Step 2: Install Dependencies

```bash
# RECOMMENDED: Install from requirements.txt
pip install -r requirements.txt

# This installs (takes 5-10 minutes):
# - TensorFlow 2.10.1 (CPU version)
# - Prophet 1.1.5
# - ARIMA/pmdarima
# - Streamlit 1.32.2
# - All supporting libraries
```

**Manual installation order** (if requirements.txt fails):
```bash
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install tensorflow==2.10.1
pip install prophet==1.1.5
pip install pmdarima==2.0.4
pip install streamlit==1.32.2
pip install plotly==5.18.0
pip install scikit-learn==1.3.2
```

### Step 3: Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "from prophet import Prophet; print('Prophet OK')"
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"
```

All should print without errors ✅

---

## 📖 Usage

### Running the Application

1. **Start the Streamlit server:**
```bash
streamlit run app.py
```

2. **Open your browser:**
   - The app should automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

3. **Upload your data:**
   - Click "Browse files" in the sidebar
   - Select a Bitcoin CSV or ZIP file (see [Dataset](#dataset) section)

4. **Configure settings:**
   - Choose a forecasting model (Prophet / ARIMA / LSTM)
   - Set forecast horizon (e.g., 30 days)
   - Adjust confidence interval (e.g., 80%)
   - For LSTM: Set lookback window (default: 60 days)

5. **Generate forecast:**
   - Click the "🚀 Generate Forecast" button
   - Wait for the model to train:
     - Prophet: 10-30 seconds
     - ARIMA: 20-60 seconds
     - LSTM: 60-120 seconds (trains neural network)
   - View results, metrics, and interactive chart

6. **Export results:**
   - Download forecast data as CSV
   - Use charts for presentations or reports

---

## 🧠 Model Explanations

### How Prophet Handles Crypto Volatility

Prophet is particularly well-suited for Bitcoin forecasting because:

1. **Trend Detection**
   - Identifies long-term growth or decline patterns
   - Adapts to trend changes (e.g., bull/bear markets)

2. **Seasonality Handling**
   - **Daily**: Bitcoin trades 24/7, Prophet captures intra-week patterns
   - **Weekly**: Weekend vs weekday trading behavior
   - **Yearly**: Long-term cyclical patterns (e.g., halving events)

3. **Robustness to Outliers**
   - Crypto markets have extreme price swings
   - Prophet's multiplicative model handles volatility better than linear models

4. **Uncertainty Quantification**
   - Provides confidence intervals (shaded areas on chart)
   - Wider intervals = higher uncertainty
   - Useful for risk assessment in volatile markets

### Why ARIMA as Comparison?

ARIMA (AutoRegressive Integrated Moving Average) serves as a classical baseline:

1. **Strengths**
   - Well-established statistical foundation
   - Good for short-term forecasting
   - Interpretable parameters (p, d, q)

2. **Challenges with Crypto**
   - Assumes stationarity (crypto is highly non-stationary)
   - Struggles with sudden volatility spikes
   - No built-in seasonality handling

3. **Use Case**
   - Compare against modern methods (Prophet, LSTM)
   - Understand baseline performance
   - Educational value in understanding classical time-series

### LSTM Deep Learning Model ⭐ **NEW!**

LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network designed for sequence prediction:

#### **How LSTM Works:**

1. **Sliding Window Approach**
   ```
   Lookback window = 60 days
   
   Input:  [Day 1, Day 2, ..., Day 60] → Output: Day 61
   Input:  [Day 2, Day 3, ..., Day 61] → Output: Day 62
   ...
   ```

2. **Network Architecture**
   ```
   Input Layer (60 time steps × 1 feature)
      ↓
   LSTM Layer 1 (50 units) + Dropout (20%)
      ↓
   LSTM Layer 2 (50 units) + Dropout (20%)
      ↓
   Dense Layer (25 units)
      ↓
   Output Layer (1 price prediction)
   ```

3. **Why LSTM for Bitcoin?**
   - ✅ **Temporal Dependencies**: Remembers patterns from 60 days ago
   - ✅ **Non-Linear Relationships**: Captures complex price dynamics
   - ✅ **No Stationarity Assumption**: Works with trending data
   - ✅ **Long-Term Memory**: LSTM cells maintain information across long sequences

4. **Training Process**
   - **Data Normalization**: Prices scaled to 0-1 range using MinMaxScaler
   - **Sequence Creation**: Creates 60-day windows from historical data
   - **Model Training**: Uses Adam optimizer with MSE loss
   - **Early Stopping**: Stops if no improvement for 10 epochs
   - **Multi-Step Forecasting**: Predicts one day at a time, feeds prediction back

5. **Confidence Intervals**
   - LSTM doesn't naturally provide uncertainty estimates
   - We calculate them from historical prediction errors
   - Uses normal distribution assumption with z-scores

#### **When to Use LSTM:**
- ✅ Long-term forecasts (30+ days)
- ✅ Complex, non-linear price patterns
- ✅ When you have sufficient data (1000+ days recommended)
- ✅ Comparing deep learning vs traditional methods

#### **LSTM vs Prophet vs ARIMA:**

| Aspect | Prophet | ARIMA | LSTM |
|--------|---------|-------|------|
| **Type** | Modern statistical | Classical statistical | Deep learning |
| **Best For** | Seasonal trends | Short-term, stable data | Complex patterns |
| **Training Time** | Fast (10-30s) | Medium (20-60s) | Slow (60-120s) |
| **Data Needed** | Moderate (500+) | Moderate (500+) | More (1000+) |
| **Interpretability** | Medium | High | Low |
| **Handling Volatility** | Excellent | Poor | Excellent |
| **Customization** | Low | Medium | High |

---

## 📊 Dataset

### Recommended Kaggle Datasets

1. **Bitcoin Historical Data** (Recommended)
   - Link: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
   - Frequency: Minute-level data (resample to daily)
   - Columns: Timestamp, Open, High, Low, Close, Volume
   - Coverage: 2012 - 2021

2. **Cryptocurrency Historical Prices**
   - Link: https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory
   - Multiple cryptocurrencies
   - Daily data

3. **Alternative**: Yahoo Finance
   - Link: https://finance.yahoo.com/quote/BTC-USD/history
   - Download manually as CSV
   - Recent data (1 year recommended for testing)

### Data Requirements

Your CSV must have:
- ✅ **Date column**: `Date`, `Timestamp`, or similar
- ✅ **Price column**: `Close`, `Open`, `High`, `Low`, or `Price`
- ✅ **Format**: Standard CSV with headers
- ✅ **Chronological order**: Sorted by date (app auto-sorts if needed)

### Large File Handling

See **HANDLING_LARGE_FILES.md** for:
- ZIP file upload (recommended for >200MB files)
- Automatic downsampling (minute → daily data)
- Downsample script for preprocessing

---

## 📁 Project Structure

```
bitcoin-forecasting/
│
├── app.py                       # Main Streamlit application (LSTM added!)
├── requirements.txt             # Python dependencies (TensorFlow included)
├── README.md                    # This file
│
├── ENVIRONMENT_SETUP.md         # Fresh environment creation guide ⭐ **NEW!**
├── QUICKSTART.md               # 5-minute setup guide
├── DATASET_LINKS.md            # Data sources
├── HANDLING_LARGE_FILES.md     # ZIP upload & downsampling
├── ARCHITECTURE.md             # Code explanations
│
├── downsample_bitcoin_data.py  # Helper script (minute → daily)
│
└── bitcoin_forecast_env/       # Virtual environment (created by you)
```

---

## 🔧 Technical Details

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│  (User Interface, Controls, File Upload)            │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              Data Processing Layer                   │
│  • CSV/ZIP Parsing                                  │
│  • Date/Price Detection                             │
│  • Validation & Cleaning                            │
│  • Train/Test Split (80/20)                         │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              Forecasting Engine                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │
│  │ Prophet  │  │  ARIMA   │  │  LSTM (NEW!) │     │
│  │  •Trend  │  │ •Auto(p, │  │ •TensorFlow  │     │
│  │  •Season │  │  d,q)    │  │ •2 LSTM      │     │
│  │  •Outlier│  │ •Forecast│  │  layers      │     │
│  └──────────┘  └──────────┘  │ •Dropout     │     │
│                               │ •Sequence    │     │
│                               └──────────────┘     │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│            Visualization & Metrics                   │
│  • Plotly Interactive Charts                        │
│  • MAE, RMSE, MAPE Calculation                     │
│  • Confidence Intervals                             │
│  • CSV Export                                       │
└─────────────────────────────────────────────────────┘
```

### Performance Considerations

1. **Training Time**
   - Prophet: 10-30 seconds (depends on data size)
   - ARIMA: 20-60 seconds (auto-parameter search)
   - **LSTM**: 60-120 seconds (neural network training)
     - First epoch: ~5-10 seconds
     - Subsequent epochs: ~2-3 seconds each
     - Early stopping reduces training time

2. **Memory Usage**
   - Typical: 500MB - 1GB
   - **LSTM**: Additional 200-300MB for TensorFlow
   - Large datasets (10+ years): 2GB+

3. **Optimization Tips**
   - Use daily data instead of minute/hourly
   - Limit historical data to 2-5 years for faster training
   - For LSTM: Reduce lookback window to 30 if slow
   - Close other memory-intensive applications

---

## 🐛 Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Fails
```
ERROR: Could not find a version that satisfies tensorflow==2.10.1
```

**Solution**:
- **MUST use Python 3.10** (not 3.11, 3.12, or 3.14!)
- TensorFlow 2.10.1 only supports Python 3.7-3.10
- Delete environment and recreate with Python 3.10:
```bash
py -3.10 -m venv bitcoin_forecast_env
```

#### 2. DLL Load Failed (Windows)
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

**Solution**:
1. Install/Reinstall VC++ Redistributable:
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install even if already installed
   - Restart computer

2. Check Windows Updates:
   - Install all pending updates
   - Restart

#### 3. LSTM Training is Slow
```
Training taking 5+ minutes...
```

**Solution**:
- **Normal for first run** (TensorFlow initialization)
- Reduce lookback window: Try 30 instead of 60
- Use smaller dataset: Last 2-3 years only
- Enable downsampling for minute-level data
- Close Chrome/Firefox to free memory

#### 4. Prophet Shows Warnings
```
WARNING:prophet:Disabling yearly seasonality...
```

**Solution**:
- **This is NORMAL** - ignore warnings
- Prophet works perfectly fine with warnings
- Warnings are just informational

#### 5. "No module named 'sklearn'"
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**:
```bash
pip install scikit-learn==1.3.2
```

---

## 📝 Assignment Compliance Checklist

- ✅ **Data Ingestion**: CSV/ZIP upload with auto-detection
- ✅ **Configuration Sidebar**: Model, horizon, confidence controls
- ✅ **Forecasting Engine**: Prophet, ARIMA, + LSTM (bonus!)
- ✅ **Backtesting**: 80/20 split with MAE/RMSE/MAPE metrics
- ✅ **Interactive Visualization**: Plotly charts with all required elements
- ✅ **Streamlit UI**: Complete app.py with professional layout
- ✅ **Documentation**: Comprehensive README + setup guides
- ✅ **Dependencies**: Complete requirements.txt
- ✅ **Dataset Link**: Kaggle links provided
- ✅ **Error Handling**: Validation and user feedback
- ✅ **BONUS**: Deep learning model (LSTM)

---

## 🎓 Learning Outcomes

By completing this project, you will understand:

1. **Time-Series Forecasting**
   - Trend, seasonality, and noise components
   - Prophet's additive/multiplicative models
   - ARIMA's (p,d,q) parameters
   - **LSTM's sequence learning**

2. **Deep Learning**
   - **Neural network architecture**
   - **Recurrent neural networks (RNNs)**
   - **LSTM cells and memory gates**
   - **Training with backpropagation**
   - **Overfitting prevention (Dropout, Early Stopping)**

3. **Model Evaluation**
   - Train/test split methodology
   - MAE vs RMSE interpretation
   - Confidence interval meaning
   - Comparing traditional vs deep learning

4. **Production Web Apps**
   - Streamlit framework
   - Interactive visualizations with Plotly
   - User input validation and error handling

5. **Financial Data Analysis**
   - Handling volatile time-series
   - Understanding crypto market patterns
   - Risk quantification with uncertainty bands

---

## ⚠️ Disclaimer

This application is for **educational purposes only**. The forecasts generated should **NOT** be used as:
- Investment advice
- Trading signals
- Financial guidance

Cryptocurrency markets are extremely volatile and unpredictable. Always:
- Do your own research (DYOR)
- Consult with licensed financial advisors
- Never invest more than you can afford to lose

---

## 📧 Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review **ENVIRONMENT_SETUP.md** for installation help
3. Verify all dependencies are installed correctly
4. Search for similar issues online

---

## 📚 Additional Resources

### Learning Materials
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [ARIMA Tutorial](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [TensorFlow/Keras Tutorials](https://www.tensorflow.org/tutorials)
- [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)

### Research Papers
- Taylor & Letham (2018): "Forecasting at Scale" (Prophet paper)
- Box & Jenkins (1970): "Time Series Analysis" (ARIMA foundations)
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory" (LSTM paper)

---

## 📄 License

This project is released under the **MIT License** for educational use.

---

**Built with ❤️ for the ITI AI Track**

*Last Updated: April 2026*
*Now with LSTM Deep Learning Support! 🧠*
