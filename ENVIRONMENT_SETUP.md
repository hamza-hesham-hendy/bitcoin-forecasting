# 🔄 Complete Environment Reset & Fresh Setup Guide
## Delete Old Environment + Create New Stable Setup

---

## 🎯 **What This Guide Does**

This guide will help you:
1. **Delete** the old `timeseries_project` environment completely
2. **Create** a fresh, stable Python 3.10 environment
3. **Install** TensorFlow + Prophet + ARIMA dependencies correctly
4. **Test** the installation to ensure everything works
5. **Run** the Bitcoin forecasting app with LSTM support

---

## ⚠️ **IMPORTANT: Read Before Starting**

- This will **permanently delete** the `timeseries_project` environment
- You will need to **reinstall** all packages (takes 5-10 minutes)
- The new environment will be called `bitcoin_forecast_env`
- **Python 3.10** is required (you have it installed ✅)
- **VC_redist.x64** is already installed ✅

---

## 📋 **Step-by-Step Instructions**

### **Step 1: Delete the Old Environment**

#### **Option A: Using Conda (if you used Conda)**

```bash
# List all conda environments to confirm the name
conda env list

# Delete the timeseries_project environment
conda env remove --name timeseries_project

# Verify it's deleted
conda env list
```

#### **Option B: Using venv (if you used Python venv)**

```bash
# Navigate to where the environment is located
# Usually in your project folder or a central location

# Delete the entire folder
# Windows:
rmdir /s timeseries_project

# Linux/Mac:
rm -rf timeseries_project

# Or if it's in a specific path:
rmdir /s C:\path\to\timeseries_project
```

**How to find where it is:**
```bash
# If you're inside the environment:
where python
# This shows the path - the environment folder is the parent directory

# Or look in common locations:
# - Your project folder: C:\Users\YourName\bitcoin-project\timeseries_project
# - Central venv location: C:\Users\YourName\.virtualenvs\timeseries_project
```

---

### **Step 2: Navigate to Your Project Folder**

```bash
# Open Command Prompt or PowerShell
# Navigate to where you saved the Bitcoin app files

cd C:\path\to\your\bitcoin-forecasting-project

# Or create a new folder:
mkdir C:\Users\YourName\Documents\bitcoin_forecast
cd C:\Users\YourName\Documents\bitcoin_forecast

# Place all the downloaded files here:
# - app.py
# - requirements.txt
# - README.md
# - etc.
```

---

### **Step 3: Create Fresh Python 3.10 Environment**

```bash
# Verify Python 3.10 is available
python --version
# Should show: Python 3.10.x

# If it shows a different version, specify Python 3.10 explicitly:
py -3.10 --version

# Create new virtual environment
# Use Python 3.10 specifically:
py -3.10 -m venv bitcoin_forecast_env

# Or if 'python' already points to 3.10:
python -m venv bitcoin_forecast_env
```

**Expected output:**
```
Creating virtual environment...
(No errors = success!)
```

---

### **Step 4: Activate the New Environment**

#### **Windows Command Prompt:**
```bash
bitcoin_forecast_env\Scripts\activate
```

#### **Windows PowerShell:**
```powershell
bitcoin_forecast_env\Scripts\Activate.ps1
```

**If you get an error about execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
bitcoin_forecast_env\Scripts\Activate.ps1
```

#### **Linux/Mac:**
```bash
source bitcoin_forecast_env/bin/activate
```

**You know it worked when you see:**
```
(bitcoin_forecast_env) C:\path\to\project>
```

---

### **Step 5: Upgrade pip (IMPORTANT!)**

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Verify pip version
pip --version
# Should show: pip 24.x or higher
```

---

### **Step 6: Install Dependencies (Carefully!)**

#### **🚨 CRITICAL: Install in This EXACT Order**

```bash
# Step 6.1: Install core dependencies first
pip install numpy==1.24.3
pip install pandas==2.0.3

# Step 6.2: Install scientific stack
pip install scipy==1.10.1
pip install scikit-learn==1.3.2

# Step 6.3: Install TensorFlow (this takes longest - 2-3 minutes)
pip install tensorflow==2.10.1

# Step 6.4: Install Prophet dependencies first
pip install cmdstanpy==1.2.1
pip install prophet==1.1.5

# Step 6.5: Install ARIMA
pip install statsmodels==0.14.1
pip install pmdarima==2.0.4

# Step 6.6: Install visualization
pip install plotly==5.18.0
pip install matplotlib==3.7.5

# Step 6.7: Install Streamlit
pip install streamlit==1.32.2
```

**OR use requirements.txt (easier):**
```bash
# Install everything from requirements.txt
pip install -r requirements.txt

# This will take 5-10 minutes total
# You'll see lots of output - this is normal
```

---

### **Step 7: Verify Installation (CRITICAL STEP)**

Run each command and make sure there are NO errors:

```bash
# Test 1: NumPy
python -c "import numpy; print(f'NumPy {numpy.__version__} ✅')"
# Expected: NumPy 1.24.3 ✅

# Test 2: Pandas
python -c "import pandas; print(f'Pandas {pandas.__version__} ✅')"
# Expected: Pandas 2.0.3 ✅

# Test 3: TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} ✅')"
# Expected: TensorFlow 2.10.1 ✅

# Test 4: Keras (comes with TensorFlow)
python -c "from tensorflow import keras; print('Keras ✅')"
# Expected: Keras ✅

# Test 5: Prophet
python -c "from prophet import Prophet; print('Prophet ✅')"
# Expected: Prophet ✅
# (might show some warnings - ignore them)

# Test 6: ARIMA
python -c "from pmdarima import auto_arima; print('ARIMA ✅')"
# Expected: ARIMA ✅

# Test 7: Streamlit
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} ✅')"
# Expected: Streamlit 1.32.2 ✅

# Test 8: Plotly
python -c "import plotly; print(f'Plotly {plotly.__version__} ✅')"
# Expected: Plotly 5.18.0 ✅
```

**If ALL tests pass, you're ready to go! 🎉**

---

### **Step 8: Run the Application**

```bash
# Make sure you're in the project folder
# Make sure environment is activated (bitcoin_forecast_env)

# Run the app
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

**The app should automatically open in your browser!**

---

## ✅ **Verification Checklist**

Before using the app, verify:

- [ ] Old environment deleted (`conda env list` or folder doesn't exist)
- [ ] New environment created (`bitcoin_forecast_env` folder exists)
- [ ] Environment activated (see `(bitcoin_forecast_env)` in terminal)
- [ ] All 8 import tests passed with ✅
- [ ] Streamlit runs without errors
- [ ] App opens in browser at http://localhost:8501

---

## 🐛 **Troubleshooting Common Issues**

### **Issue 1: "Python not found" or wrong version**

```bash
# Find all Python versions installed
py --list

# Use specific version:
py -3.10 -m venv bitcoin_forecast_env
```

### **Issue 2: TensorFlow installation fails**

**Error:** "Could not find a version that satisfies the requirement tensorflow==2.10.1"

**Fix:**
```bash
# Make sure you're using Python 3.10 (not 3.11, 3.12, or 3.14!)
python --version
# Should show 3.10.x

# If not, recreate environment with correct Python:
deactivate
rmdir /s bitcoin_forecast_env
py -3.10 -m venv bitcoin_forecast_env
bitcoin_forecast_env\Scripts\activate
```

### **Issue 3: Prophet installation takes forever / fails**

**Solution:**
```bash
# Install cmdstanpy first (Prophet's backend)
pip install cmdstanpy==1.2.1

# Then install prophet
pip install prophet==1.1.5

# If still fails, install Prophet without dependencies first:
pip install --no-deps prophet==1.1.5
pip install cmdstanpy==1.2.1
```

### **Issue 4: DLL load failed errors**

**Error:** "ImportError: DLL load failed while importing..."

**Fix:**
1. **Verify VC++ Redistributable is installed:**
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install (even if already installed - reinstall)
   - Restart computer

2. **Check Windows Updates:**
   - Windows Update → Check for updates
   - Install all updates
   - Restart

### **Issue 5: Streamlit doesn't start**

```bash
# Reinstall Streamlit
pip uninstall streamlit
pip install streamlit==1.32.2

# Clear Streamlit cache
streamlit cache clear

# Try running again
streamlit run app.py
```

### **Issue 6: Prophet shows warnings about Stan**

**Warning:** "WARNING:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this."

**This is NORMAL - ignore it!** Prophet works fine with warnings.

---

## 📊 **What Each Package Does**

| Package | Purpose | Why This Version? |
|---------|---------|-------------------|
| **Python 3.10** | Base interpreter | Most stable for TensorFlow 2.10 |
| **NumPy 1.24.3** | Array operations | Last pre-2.0 version (Prophet compatibility) |
| **TensorFlow 2.10.1** | LSTM deep learning | Last version with stable Windows CPU support |
| **Prophet 1.1.5** | Time-series forecasting | Latest stable version |
| **pmdarima 2.0.4** | Auto-ARIMA | Latest version with statsmodels 0.14 support |
| **Streamlit 1.32.2** | Web interface | Stable, no protobuf conflicts |
| **Plotly 5.18.0** | Interactive charts | Latest stable |

---

## 🎯 **Quick Reference Commands**

### **Daily Workflow:**
```bash
# 1. Activate environment
cd C:\path\to\bitcoin-forecasting
bitcoin_forecast_env\Scripts\activate

# 2. Run app
streamlit run app.py

# 3. When done, deactivate
deactivate
```

### **Update packages:**
```bash
pip list --outdated          # Check for updates
pip install --upgrade <package>  # Update specific package
```

### **Check installed versions:**
```bash
pip list                     # All packages
pip show tensorflow          # Specific package details
```

---

## 🧪 **Final Test: Run a Simple Forecast**

1. **Download test data:**
   - Go to: https://finance.yahoo.com/quote/BTC-USD/history
   - Select "1Y" (1 year)
   - Click "Download"

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Test each model:**
   - Upload the CSV
   - Select **Prophet** → Generate Forecast
   - Should complete in ~10-20 seconds ✅
   
   - Select **ARIMA** → Generate Forecast
   - Should complete in ~30-60 seconds ✅
   
   - Select **LSTM (Deep Learning)** → Generate Forecast
   - Should complete in ~60-120 seconds ✅
   - **This is the NEW model!**

4. **If all three models work, you're done! 🎉**

---

## 💡 **Tips for Success**

1. **Always activate the environment first:**
   - You'll see `(bitcoin_forecast_env)` in your terminal
   - If you don't see it, run: `bitcoin_forecast_env\Scripts\activate`

2. **Keep the environment activated:**
   - One terminal window = one environment
   - If you close the terminal, you need to reactivate

3. **Don't install packages globally:**
   - Always activate environment before `pip install`
   - This keeps projects isolated

4. **If something breaks:**
   - Don't panic!
   - Deactivate environment
   - Delete `bitcoin_forecast_env` folder
   - Start again from Step 3

---

## 📝 **Summary**

**What you deleted:**
- ❌ Old `timeseries_project` environment (broken dependencies)

**What you created:**
- ✅ New `bitcoin_forecast_env` environment (Python 3.10)
- ✅ Stable dependency stack (TensorFlow 2.10.1 + Prophet + ARIMA)
- ✅ LSTM deep learning support
- ✅ Working Streamlit app

**What you can now do:**
- ✅ Forecast Bitcoin prices with 3 models
- ✅ Compare Prophet vs ARIMA vs LSTM
- ✅ Generate interactive charts
- ✅ Export forecast data

---

## 🎊 **You're Ready!**

Next steps:
1. ✅ Environment created and tested
2. 📖 Read `README.md` for app features
3. 🚀 Run `streamlit run app.py`
4. 📊 Upload Bitcoin data
5. 🧠 Try all three models!

**For detailed app usage, see:**
- `QUICKSTART.md` - 5-minute quick start
- `README.md` - Complete documentation
- `ARCHITECTURE.md` - Code explanations

---

**Good luck with your Bitcoin forecasting project! 🚀**
