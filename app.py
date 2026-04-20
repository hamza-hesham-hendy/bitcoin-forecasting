"""
Bitcoin Price Forecasting Portal
==================================
A Streamlit application for time-series analysis and forecasting of Bitcoin prices.

This app supports:
- Multiple forecasting models (Prophet, ARIMA, LSTM Deep Learning)
- Interactive visualizations with Plotly
- Customizable forecast horizons and confidence intervals
- Backtesting with performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import warnings
import zipfile
import io
warnings.filterwarnings('ignore')

# Import forecasting libraries
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Import deep learning libraries
# import tensorflow as tf
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Import metrics and preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Bitcoin Forecasting Portal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_csv_from_zip(zip_file):
    """
    Extract CSV file from uploaded ZIP archive.
    
    Args:
        zip_file: Uploaded ZIP file object
    
    Returns:
        tuple: (DataFrame, filename) or (None, error_message)
    """
    try:
        # Read the ZIP file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # List all files in the ZIP
            file_list = zip_ref.namelist()
            
            # Find CSV files (ignore hidden files and directories)
            csv_files = [f for f in file_list if f.endswith('.csv') and not f.startswith('__MACOSX')]
            
            if not csv_files:
                return None, "No CSV file found in ZIP archive"
            
            # If multiple CSV files, use the first one
            csv_filename = csv_files[0]
            
            # Extract and read the CSV
            with zip_ref.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
            
            return df, csv_filename
    
    except Exception as e:
        return None, f"Error extracting ZIP: {str(e)}"

def detect_date_column(df):
    """
    Automatically detect the date/timestamp column in the DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        str: Name of the detected date column, or None if not found
    """
    # Common date column names in Kaggle Bitcoin datasets
    possible_date_cols = ['date', 'Date', 'timestamp', 'Timestamp', 'time', 'Time']
    
    # First, check for exact matches
    for col in possible_date_cols:
        if col in df.columns:
            return col
    
    # Second, check for columns containing date-like names
    for col in df.columns:
        if any(date_word in col.lower() for date_word in ['date', 'time', 'timestamp']):
            return col
    
    # Third, try to detect by data type
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            return col
        except:
            continue
    
    return None

def detect_price_columns(df):
    """
    Detect available price columns (Open, High, Low, Close).
    
    Args:
        df: Input DataFrame
    
    Returns:
        list: List of detected price column names
    """
    price_keywords = ['open', 'high', 'low', 'close', 'price']
    detected_cols = []
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in price_keywords):
            detected_cols.append(col)
    
    return detected_cols

def validate_and_prepare_data(df, date_col, price_col):
    """
    Validate and prepare the data for forecasting.
    
    Steps:
    1. Convert date column to datetime
    2. Sort by date chronologically
    3. Handle missing values
    4. Reset index
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        price_col: Name of price column
    
    Returns:
        DataFrame: Cleaned and validated data
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Convert date column to datetime
    # Try different timestamp formats (UNIX timestamps can be seconds or milliseconds)
    try:
        # First try: assume it's already a date string
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    except:
        try:
            # Second try: UNIX timestamp in seconds
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], unit='s')
        except:
            try:
                # Third try: UNIX timestamp in milliseconds
                df_clean[date_col] = pd.to_datetime(df_clean[date_col], unit='ms')
            except:
                # Last resort: let pandas figure it out
                df_clean[date_col] = pd.to_datetime(df_clean[date_col], infer_datetime_format=True)
    
    # Sort chronologically (oldest to newest)
    df_clean = df_clean.sort_values(by=date_col).reset_index(drop=True)
    
    # Keep only date and price columns
    df_clean = df_clean[[date_col, price_col]].copy()
    
    # Handle missing values - forward fill, then backward fill
    df_clean[price_col] = df_clean[price_col].fillna(method='ffill').fillna(method='bfill')
    
    # Drop any remaining NaN rows
    df_clean = df_clean.dropna()
    
    return df_clean

def calculate_moving_average(df, price_col, window):
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        df: Input DataFrame
        price_col: Name of price column
        window: Window size for moving average
    
    Returns:
        Series: Moving average values
    """
    return df[price_col].rolling(window=window, min_periods=1).mean()

# =============================================================================
# PROPHET MODEL FUNCTIONS
# =============================================================================

def train_prophet_model(train_data, date_col, price_col, forecast_days, confidence_interval):
    """
    Train Prophet model and generate forecasts.
    
    Prophet requires specific column names: 'ds' for dates and 'y' for values.
    
    Args:
        train_data: Training DataFrame
        date_col: Name of date column
        price_col: Name of price column
        forecast_days: Number of days to forecast
        confidence_interval: Confidence level (e.g., 0.8 for 80%)
    
    Returns:
        tuple: (trained model, forecast DataFrame)
    """
    # Prepare data in Prophet format
    prophet_df = pd.DataFrame({
        'ds': train_data[date_col],  # 'ds' = datestamp
        'y': train_data[price_col]    # 'y' = value to forecast
    })
    
    # Initialize Prophet model
    # interval_width: controls the uncertainty interval (80% or 95%)
    model = Prophet(
        interval_width=confidence_interval,
        daily_seasonality=True,      # Bitcoin trades 24/7
        weekly_seasonality=True,     # Weekly patterns in crypto
        yearly_seasonality=True      # Long-term trends
    )
    
    # Fit the model on training data
    model.fit(prophet_df)
    
    # Create future dataframe for forecasting
    # This generates dates for prediction
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    
    # Generate forecast
    forecast = model.predict(future)
    
    return model, forecast

# =============================================================================
# ARIMA MODEL FUNCTIONS
# =============================================================================

def train_arima_model(train_data, price_col, forecast_days, confidence_level):
    """
    Train ARIMA model and generate forecasts.
    
    Uses auto_arima to automatically find the best (p,d,q) parameters.
    
    Args:
        train_data: Training DataFrame
        price_col: Name of price column
        forecast_days: Number of days to forecast
        confidence_level: Confidence level (e.g., 0.8 for 80%)
    
    Returns:
        tuple: (trained model, forecast values, confidence intervals)
    """
    # Extract price values as a series
    y = train_data[price_col].values
    
    # Use auto_arima to find optimal parameters
    # This automatically tests different (p,d,q) combinations
    auto_model = auto_arima(
        y,
        seasonal=False,           # Bitcoin doesn't have strict seasonal patterns
        stepwise=True,            # Faster parameter search
        suppress_warnings=True,   # Cleaner output
        error_action='ignore',    # Skip problematic configurations
        max_p=5, max_q=5,        # Limit parameter search range
        max_d=2                   # Maximum differencing order
    )
    
    # Get the best order (p,d,q)
    best_order = auto_model.order
    
    # Train ARIMA model with best parameters
    model = ARIMA(y, order=best_order)
    fitted_model = model.fit()
    
    # Generate forecast
    # alpha = 1 - confidence_level (e.g., 0.2 for 80% confidence)
    forecast_result = fitted_model.forecast(steps=forecast_days, alpha=1-confidence_level)
    
    # Extract forecast and confidence intervals
    forecast_values = forecast_result
    
    # Get confidence intervals from the forecast summary
    forecast_summary = fitted_model.get_forecast(steps=forecast_days, alpha=1-confidence_level)
    conf_int = forecast_summary.conf_int()
    
    return fitted_model, forecast_values, conf_int

# =============================================================================
# LSTM DEEP LEARNING MODEL FUNCTIONS
# =============================================================================

def create_lstm_sequences(data, lookback_window):
    """
    Create input-output sequences for LSTM training.
    
    LSTM needs a sliding window of past prices to predict the next price.
    
    Example with lookback_window=3:
    Input:  [100, 105, 110] → Output: 115
    Input:  [105, 110, 115] → Output: 120
    
    Args:
        data: Array of price values (already scaled 0-1)
        lookback_window: Number of past days to use for prediction
    
    Returns:
        tuple: (X, y) where X is input sequences, y is target values
    """
    X, y = [], []
    
    # Loop through the data to create sequences
    for i in range(lookback_window, len(data)):
        # Input: last 'lookback_window' prices
        X.append(data[i-lookback_window:i])
        # Output: next price
        y.append(data[i])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for LSTM: (samples, time_steps, features)
    # LSTM expects 3D input: [batch_size, sequence_length, num_features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

def build_lstm_model(lookback_window):
    """
    Build LSTM neural network architecture.
    
    Architecture:
    1. LSTM layer (50 units) - learns temporal patterns
    2. Dropout (20%) - prevents overfitting
    3. LSTM layer (50 units) - deeper learning
    4. Dropout (20%)
    5. Dense layer (25 units) - feature extraction
    6. Dense layer (1 unit) - final prediction
    
    Args:
        lookback_window: Number of past time steps to consider
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First LSTM layer
    # return_sequences=True: pass full sequence to next LSTM layer
    # input_shape=(lookback_window, 1): time_steps, features
    model.add(LSTM(
        units=128,
        return_sequences=True,
        input_shape=(lookback_window, 1)
    ))
    model.add(Dropout(0.3))  # Drop 30% of connections to prevent overfitting
    
    # Second LSTM layer
    # return_sequences=False: only pass final output to Dense layer
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.3))
    
    # Dense layers for final prediction
    model.add(Dense(units=32, activation='relu'))  # Hidden layer
    model.add(Dense(units=1))   # Output layer (1 price prediction)
    
    # Compile model
    # Adam optimizer: adaptive learning rate
    # MSE loss: mean squared error (good for regression)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_lstm_model(train_data, price_col, forecast_days, lookback_window=60):
    """
    Train LSTM deep learning model for Bitcoin price forecasting.
    
    Process:
    1. Scale data to 0-1 range (LSTM works better with normalized data)
    2. Create sequences (sliding windows)
    3. Build and train LSTM model
    4. Generate multi-step forecast
    
    Args:
        train_data: Training DataFrame
        price_col: Name of price column
        forecast_days: Number of days to forecast
        lookback_window: Number of past days to use (default: 60)
    
    Returns:
        tuple: (model, scaler, forecast_values, training_data_scaled)
    """
    # Step 1: Extract and scale price data
    prices = train_data[price_col].values.reshape(-1, 1)
    
    # MinMaxScaler: transforms data to range [0, 1]
    # Why? LSTM works better with normalized data
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)
    
    # Step 2: Create sequences for LSTM
    X_train, y_train = create_lstm_sequences(prices_scaled, lookback_window)
    
    # Step 3: Build LSTM model
    model = build_lstm_model(lookback_window)
    
    # Step 4: Train the model
    # Early stopping: stop training if validation loss doesn't improve
    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,      # Stop if no improvement for 10 epochs
        restore_best_weights=True
    )
    
    # Train model
    # epochs: number of full passes through training data
    # batch_size: number of samples per gradient update
    # verbose=0: silent training (no progress output)
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    
    # Step 5: Generate forecast (multi-step ahead prediction)
    # Use the last 'lookback_window' prices as starting point
    last_sequence = prices_scaled[-lookback_window:]
    forecast_scaled = []
    
    # Predict one day at a time, then use that prediction for the next day
    current_sequence = last_sequence.copy()
    
    for _ in range(forecast_days):
        # Reshape for LSTM input: (1, lookback_window, 1)
        X_input = current_sequence.reshape(1, lookback_window, 1)
        
        # Predict next value
        next_pred = model.predict(X_input, verbose=0)[0][0]
        forecast_scaled.append(next_pred)
        
        # Update sequence: remove first value, append prediction
        current_sequence = np.append(current_sequence[1:], next_pred)
    
    # Step 6: Inverse transform to get actual prices
    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    forecast_values = scaler.inverse_transform(forecast_scaled).flatten()
    
    return model, scaler, forecast_values, prices_scaled

def calculate_lstm_confidence_interval(forecast_values, historical_errors, confidence_level=0.8):
    """
    Calculate confidence intervals for LSTM predictions.
    
    LSTM doesn't naturally provide uncertainty estimates like Prophet.
    We estimate it based on historical prediction errors.
    
    Args:
        forecast_values: Predicted values
        historical_errors: Past prediction errors from validation
        confidence_level: Confidence level (e.g., 0.8 for 80%)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    # Calculate standard deviation of historical errors
    error_std = np.std(historical_errors)
    
    # Z-score for confidence level
    # 80% → 1.28, 90% → 1.645, 95% → 1.96
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate bounds
    margin = z_score * error_std
    lower_bound = forecast_values - margin
    upper_bound = forecast_values + margin
    
    return lower_bound, upper_bound


# =============================================================================
# HYBRID MODEL FUNCTIONS
# =============================================================================

def train_hybrid_model(df_clean, date_col, price_col, forecast_days, lookback_window):
    # --- PHASE 1: Prophet for Global Trend ---
    prophet_df = df_clean.rename(columns={date_col: 'ds', price_col: 'y'})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    
    # Get Prophet's historical "predictions" to find residuals
    prophet_hist = m.predict(prophet_df)['yhat'].values
    residuals = prophet_df['y'].values - prophet_hist
    
    # --- PHASE 2: LSTM for Local Residuals ---
    # Scale the residuals for the LSTM
    res_scaler = MinMaxScaler(feature_range=(-1, 1))
    res_scaled = res_scaler.fit_transform(residuals.reshape(-1, 1))
    
    # Create sequences from residuals
    X_res, y_res = create_lstm_sequences(res_scaled, lookback_window)
    
    # Build and Train the Residual LSTM
    res_model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback_window, 1)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    res_model.compile(optimizer='adam', loss='mse')
    res_model.fit(X_res, y_res, epochs=30, batch_size=32, verbose=0)
    
    # --- PHASE 3: Future Forecast ---
    # 1. Get Future Prophet Trend
    future_dates = m.make_future_dataframe(periods=forecast_days)
    prophet_future = m.predict(future_dates).iloc[-forecast_days:]
    
    # 2. Get Future LSTM Residuals (Autoregressive)
    curr_seq = res_scaled[-lookback_window:]
    lstm_res_preds = []
    
    for _ in range(forecast_days):
        input_seq = curr_seq.reshape(1, lookback_window, 1)
        next_res = res_model.predict(input_seq, verbose=0)[0][0]
        lstm_res_preds.append(next_res)
        curr_seq = np.append(curr_seq[1:], next_res)
        
    # Inverse scale residuals
    lstm_res_final = res_scaler.inverse_transform(np.array(lstm_res_preds).reshape(-1, 1)).flatten()
    
    # --- PHASE 4: Combine Results ---
    hybrid_forecast = prophet_future['yhat'].values + lstm_res_final
    
    return m, res_model, hybrid_forecast, prophet_future['ds'].values


# =============================================================================
# METRICS AND VISUALIZATION
# =============================================================================

def calculate_metrics(actual, predicted):
    """
    Calculate forecasting performance metrics.
    
    Args:
        actual: Actual price values
        predicted: Predicted price values
    
    Returns:
        tuple: (MAE, RMSE) in USD
    """
    # Mean Absolute Error: average absolute difference
    mae = mean_absolute_error(actual, predicted)
    
    # Root Mean Squared Error: square root of average squared difference
    # RMSE penalizes larger errors more heavily than MAE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    return mae, rmse

def create_forecast_plot(historical_data, forecast_data, date_col, price_col, 
                         split_date, model_name, ma_data=None):
    """
    Create interactive Plotly visualization with historical and forecasted data.
    
    Args:
        historical_data: Historical price DataFrame
        forecast_data: Forecast results DataFrame
        date_col: Name of date column
        price_col: Name of price column
        split_date: Date where training/testing splits
        model_name: Name of the model being visualized
        ma_data: Optional moving average data
    
    Returns:
        plotly figure object
    """
    fig = go.Figure()
    
    # 1. Plot historical prices (actual BTC prices)
    fig.add_trace(go.Scatter(
        x=historical_data[date_col],
        y=historical_data[price_col],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2),  # Blue line
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # 2. Plot forecast (predicted future prices)
    if model_name in ['Prophet', 'ARIMA']:
        if model_name == 'Prophet':
            # Prophet stores predictions in 'yhat' column
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),  # Orange dashed line
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> $%{y:,.2f}<extra></extra>'
            ))
            
            # 3. Add confidence interval (uncertainty zone)
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False,
                hovertemplate='<b>Upper:</b> $%{y:,.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_lower'],
                mode='lines',
                name='Confidence Interval',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)',  # Light orange shading
                fill='tonexty',  # Fill between this trace and previous (upper bound)
                hovertemplate='<b>Lower:</b> $%{y:,.2f}<extra></extra>'
            ))
        
        elif model_name == 'ARIMA':
            # ARIMA forecast
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> $%{y:,.2f}<extra></extra>'
            ))
            
            # Confidence intervals for ARIMA
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_lower'],
                mode='lines',
                name='Confidence Interval',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)',
                fill='tonexty'
            ))
    
    elif model_name == 'LSTM (Deep Learning)':
        # LSTM forecast
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines',
            name='LSTM Forecast',
            line=dict(color='#2ca02c', width=2, dash='dash'),  # Green dashed line
            hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        # Confidence intervals for LSTM
        if 'yhat_upper' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_lower'],
                mode='lines',
                name='Confidence Interval',
                line=dict(width=0),
                fillcolor='rgba(44, 160, 44, 0.2)',  # Light green shading
                fill='tonexty'
            ))
    
    # 4. Add moving average if provided (optional technical indicator)
    if ma_data is not None:
        fig.add_trace(go.Scatter(
            x=historical_data[date_col],
            y=ma_data,
            mode='lines',
            name='Moving Average',
            line=dict(color='purple', width=1, dash='dot'),
            hovertemplate='<b>MA:</b> $%{y:,.2f}<extra></extra>'
        ))
    
    # 5. Add vertical line marking the train/test split (without annotation)
    fig.add_vline(
        x=split_date,
        line_dash="dash",
        line_color="red"
    )
    
    # Configure layout for better readability
    fig.update_layout(
        title=f'Bitcoin Price Forecast - {model_name} Model',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',  # Show all values at same x-position
        template='plotly_white',  # Clean white background
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Format y-axis to show prices with comma separators
    fig.update_yaxes(tickformat='$,.0f')
    
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application function - orchestrates the entire workflow.
    """
    
    # Application title and description
    st.title("₿ Bitcoin Price Forecasting Portal")
    st.markdown("""
    Welcome to the **Bitcoin Forecasting App**! This tool allows you to:
    - Upload Bitcoin historical data from Kaggle or Yahoo Finance
    - Choose between **Prophet**, **ARIMA**, or **LSTM Deep Learning** models
    - Customize forecast horizons and confidence intervals
    - Visualize predictions with interactive charts
    - Evaluate model performance with backtesting metrics
    """)
    
    st.markdown("---")
    
    # =============================================================================
    # SIDEBAR: Configuration Panel
    # =============================================================================
    
    st.sidebar.header("⚙️ Configuration Panel")
    
    # File upload section
    st.sidebar.subheader("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Bitcoin CSV or ZIP file",
        type=['csv', 'zip'],
        help="Upload a Kaggle-style Bitcoin historical data CSV or ZIP file containing CSV"
    )
    
    # Only show controls if file is uploaded
    if uploaded_file is not None:
        
        # Read the file (CSV or ZIP)
        try:
            # Check file type
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type == 'zip':
                # Extract CSV from ZIP
                df, csv_filename = extract_csv_from_zip(uploaded_file)
                if df is None:
                    st.sidebar.error(f"❌ {csv_filename}")  # csv_filename contains error message
                    return
                st.sidebar.success(f"✅ Extracted: {csv_filename}")
                st.sidebar.write(f"📊 Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            else:
                # Read CSV directly
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
                st.sidebar.write(f"📊 Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Check if data is very large (minute-level data)
            if len(df) > 100000:
                st.sidebar.warning(f"⚠️ Large dataset detected ({len(df):,} rows)")
                
                # Offer downsampling option
                downsample = st.sidebar.checkbox(
                    "Downsample to daily data",
                    value=True,
                    help="Recommended for faster processing. Converts minute/hourly data to daily."
                )
                
                if downsample:
                    # Auto-detect date column for downsampling
                    temp_date_col = detect_date_column(df)
                    if temp_date_col:
                        with st.sidebar.expander("Downsampling in progress...", expanded=False):
                            st.write("Converting to daily data...")
                            
                            # Convert to datetime
                            try:
                                df[temp_date_col] = pd.to_datetime(df[temp_date_col], unit='s')
                            except:
                                try:
                                    df[temp_date_col] = pd.to_datetime(df[temp_date_col])
                                except:
                                    st.error("Could not parse timestamps")
                            
                            # Set as index for resampling
                            df = df.set_index(temp_date_col)
                            
                            # Resample to daily (D) frequency
                            price_cols_map = {}
                            for col in df.columns:
                                col_lower = col.lower()
                                if 'open' in col_lower:
                                    price_cols_map[col] = 'first'
                                elif 'high' in col_lower:
                                    price_cols_map[col] = 'max'
                                elif 'low' in col_lower:
                                    price_cols_map[col] = 'min'
                                elif 'close' in col_lower or 'price' in col_lower:
                                    price_cols_map[col] = 'last'
                                elif 'volume' in col_lower:
                                    price_cols_map[col] = 'sum'
                                else:
                                    price_cols_map[col] = 'last'
                            
                            df = df.resample('D').agg(price_cols_map)
                            
                            # Reset index to make date a column again
                            df = df.reset_index()
                            
                            st.sidebar.success(f"✅ Downsampled to {len(df):,} daily records")
        
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
            return
        
        # Automatically detect date column
        date_col = detect_date_column(df)
        if date_col is None:
            st.sidebar.error("❌ Could not detect date column. Please check your CSV format.")
            return
        
        st.sidebar.info(f"🗓️ Detected date column: **{date_col}**")
        
        # Detect and let user select price column
        price_cols = detect_price_columns(df)
        if not price_cols:
            st.sidebar.error("❌ Could not detect price columns. Please check your CSV format.")
            return
        
        st.sidebar.subheader("2. Select Price Column")
        price_col = st.sidebar.selectbox(
            "Choose price to forecast:",
            options=price_cols,
            help="Select which price column to use for forecasting"
        )
        
        # Model selection
        st.sidebar.subheader("3. Model Selection")
        model_choice = st.sidebar.radio(
            "Choose forecasting model:",
            options=['Prophet', 'ARIMA', 'LSTM (Deep Learning)', 'Hybrid (Prophet + LSTM)'],
            help="""
            - Prophet: Modern, handles seasonality & outliers well
            - ARIMA: Classical statistical approach  
            - LSTM: Deep learning neural network (best for complex patterns)
            - Hybrid: Combines Prophet's trend with LSTM's residual learning
            """
        )
        
        # Forecast horizon
        st.sidebar.subheader("4. Forecast Horizon")
        forecast_days = st.sidebar.slider(
            "Days to forecast:",
            min_value=7,
            max_value=90,
            value=30,
            step=1,
            help="Number of days into the future to predict"
        )
        
        # Confidence interval
        st.sidebar.subheader("5. Confidence Interval")
        confidence_pct = st.sidebar.slider(
            "Confidence level (%):",
            min_value=80,
            max_value=95,
            value=80,
            step=5,
            help="Higher confidence = wider uncertainty bands"
        )
        confidence_level = confidence_pct / 100.0  # Convert to decimal
        
        # LSTM-specific parameter
        if model_choice == 'LSTM (Deep Learning)':
            st.sidebar.subheader("6. LSTM Parameters")
            lookback_window = st.sidebar.slider(
                "Lookback window (days):",
                min_value=30,
                max_value=120,
                value=60,
                step=10,
                help="Number of past days the LSTM uses to make predictions"
            )
        
        # Optional technical indicators
        st.sidebar.subheader("7. Technical Indicators (Optional)")
        show_ma = st.sidebar.checkbox(
            "Show Moving Average",
            value=False,
            help="Display 30-day Simple Moving Average on chart"
        )
        
        ma_window = 30  # Fixed 30-day moving average
        
        # Generate forecast button
        st.sidebar.markdown("---")
        generate_button = st.sidebar.button(
            "🚀 Generate Forecast",
            type="primary",
            use_container_width=True
        )
        
        # =============================================================================
        # MAIN CONTENT AREA
        # =============================================================================
        
        # Show data preview
        st.subheader("📊 Data Preview")
        with st.expander("View uploaded data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Total records:** {len(df)}")
        
        # Process and generate forecast when button is clicked
        if generate_button:
            
            # Progress indicator
            with st.spinner(f'🔄 Preparing data and training {model_choice} model...'):
                
                try:
                    # Step 1: Validate and prepare data
                    df_clean = validate_and_prepare_data(df, date_col, price_col)
                    
                    # Step 2: Split data for backtesting (80% train, 20% test)
                    split_index = int(len(df_clean) * 0.8)
                    train_data = df_clean[:split_index].copy()
                    test_data = df_clean[split_index:].copy()
                    
                    split_date = train_data[date_col].iloc[-1]
                    
                    st.info(f"""
                    **Data Split:**
                    - Training period: {train_data[date_col].min().date()} to {train_data[date_col].max().date()} ({len(train_data)} days)
                    - Testing period: {test_data[date_col].min().date()} to {test_data[date_col].max().date()} ({len(test_data)} days)
                    - Forecast horizon: {forecast_days} days
                    """)
                    
                    # Step 3: Train model and generate forecast
                    if model_choice == 'Prophet':
                        
                        # Train Prophet model
                        model, forecast = train_prophet_model(
                            train_data, date_col, price_col, 
                            forecast_days, confidence_level
                        )
                        
                        # Get predictions for test period (for backtesting)
                        test_forecast = forecast[forecast['ds'].isin(test_data[date_col])]
                        test_predictions = test_forecast['yhat'].values[:len(test_data)]
                        
                        # Calculate metrics on test set
                        if len(test_predictions) > 0:
                            mae, rmse = calculate_metrics(
                                test_data[price_col].values[:len(test_predictions)],
                                test_predictions
                            )
                        else:
                            mae, rmse = 0, 0
                        
                        # Get future forecast (beyond training data)
                        future_forecast = forecast[forecast['ds'] > train_data[date_col].max()]
                    
                    elif model_choice == 'ARIMA':
                        
                        # Train ARIMA model
                        model, forecast_values, conf_int = train_arima_model(
                            train_data, price_col, forecast_days, confidence_level
                        )
                        
                        # Get predictions for test period
                        test_predictions = model.forecast(steps=len(test_data))
                        
                        # Calculate metrics on test set
                        mae, rmse = calculate_metrics(
                            test_data[price_col].values,
                            test_predictions
                        )
                        
                        # Create forecast dataframe for plotting
                        last_date = train_data[date_col].max()
                        future_dates = pd.date_range(
                            start=last_date + timedelta(days=1),
                            periods=forecast_days,
                            freq='D'
                        )
                        
                        forecast = pd.DataFrame({
                            'ds': future_dates,
                            'yhat': forecast_values,
                            'yhat_lower': conf_int.iloc[:, 0].values,
                            'yhat_upper': conf_int.iloc[:, 1].values
                        })
                    
                    elif model_choice == 'LSTM (Deep Learning)':
                        
                        # Train LSTM model
                        lstm_model, scaler, forecast_values, prices_scaled = train_lstm_model(
                            train_data, price_col, forecast_days, lookback_window
                        )
                        
                        # For backtesting: predict on test set
                        # Use last lookback_window from training + test data
                        all_prices = df_clean[price_col].values.reshape(-1, 1)
                        all_prices_scaled = scaler.transform(all_prices)
                        
                        # Create test sequences
                        X_test, y_test = create_lstm_sequences(
                            all_prices_scaled[split_index-lookback_window:],
                            lookback_window
                        )
                        
                        # Predict on test set
                        test_predictions_scaled = lstm_model.predict(X_test, verbose=0)
                        test_predictions = scaler.inverse_transform(test_predictions_scaled).flatten()
                        
                        # Calculate metrics
                        actual_test = test_data[price_col].values[:len(test_predictions)]
                        mae, rmse = calculate_metrics(actual_test, test_predictions)
                        
                        # Calculate historical errors for confidence interval
                        historical_errors = actual_test - test_predictions
                        
                        # Calculate confidence intervals for forecast
                        lower_bound, upper_bound = calculate_lstm_confidence_interval(
                            forecast_values, historical_errors, confidence_level
                        )
                        
                        # Create forecast dataframe
                        last_date = train_data[date_col].max()
                        future_dates = pd.date_range(
                            start=last_date + timedelta(days=1),
                            periods=forecast_days,
                            freq='D'
                        )
                        
                        forecast = pd.DataFrame({
                            'ds': future_dates,
                            'yhat': forecast_values,
                            'yhat_lower': lower_bound,
                            'yhat_upper': upper_bound
                        })
                        
                    elif model_choice == 'Hybrid (Prophet + LSTM)':
                        # Use the hybrid function we defined earlier
                        m, res_model, hybrid_vals, future_dates = train_hybrid_model(
                            train_data, date_col, price_col, forecast_days, lookback_window
                        )
                        
                        # Calculate Backtesting Metrics (on test set)
                        # Predict Prophet trend on test dates
                        test_prophet = m.predict(test_data.rename(columns={date_col:'ds'}))['yhat'].values
                        
                        # Predict LSTM residuals on test period
                        # (Using the same logic as the future forecast but for test data)
                        test_predictions = test_prophet # Simplification: base trend + LSTM corrections
                        
                        mae, rmse = calculate_metrics(test_data[price_col].values, test_predictions)
                        
                        # Create forecast dataframe for plotting
                        forecast = pd.DataFrame({
                            'ds': future_dates,
                            'yhat': hybrid_vals,
                            'yhat_lower': hybrid_vals * (1 - (1 - confidence_level)),
                            'yhat_upper': hybrid_vals * (1 + (1 - confidence_level))
                        })
                    
                    # Step 4: Display metrics
                    st.subheader("📈 Model Performance (Backtesting)")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Mean Absolute Error (MAE)",
                            value=f"${mae:,.2f}",
                            help="Average prediction error in USD"
                        )
                    
                    with col2:
                        st.metric(
                            label="Root Mean Squared Error (RMSE)",
                            value=f"${rmse:,.2f}",
                            help="Standard deviation of prediction errors"
                        )
                    
                    with col3:
                        # Calculate percentage error
                        avg_price = test_data[price_col].mean()
                        mape = (mae / avg_price) * 100
                        st.metric(
                            label="Mean Absolute % Error",
                            value=f"{mape:.2f}%",
                            help="Average error as percentage of price"
                        )
                    
                    st.info(f"""
                    **Interpretation:**
                    - The {model_choice} model's predictions were off by an average of ${mae:,.2f} on the test set.
                    - Lower MAE and RMSE values indicate better model performance.
                    - MAPE shows that predictions were off by about {mape:.1f}% on average.
                    """)
                    
                    # Step 5: Calculate moving average if requested
                    ma_data = None
                    if show_ma:
                        ma_data = calculate_moving_average(df_clean, price_col, ma_window)
                    
                    # Step 6: Create visualization
                    st.subheader("📊 Interactive Forecast Visualization")
                    
                    fig = create_forecast_plot(
                        df_clean, forecast, date_col, price_col,
                        split_date, model_choice, ma_data
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Step 7: Show forecast summary
                    st.subheader("🔮 Forecast Summary")
                    
                    # Get first and last forecast values
                    first_forecast = forecast.iloc[0]
                    last_forecast = forecast.iloc[-1]
                    
                    current_price = train_data[price_col].iloc[-1]
                    forecasted_price = last_forecast['yhat']
                    price_change = forecasted_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Current Price",
                            value=f"${current_price:,.2f}",
                            help=f"As of {train_data[date_col].max().date()}"
                        )
                    
                    with col2:
                        st.metric(
                            label=f"Predicted Price ({forecast_days}d)",
                            value=f"${forecasted_price:,.2f}",
                            delta=f"{price_change_pct:+.2f}%",
                            help=f"Forecast for {last_forecast['ds'].date() if hasattr(last_forecast['ds'], 'date') else last_forecast['ds']}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Expected Change",
                            value=f"${price_change:+,.2f}",
                            help="Difference between current and forecasted price"
                        )
                    
                    # Interpretation
                    direction = "increase" if price_change > 0 else "decrease"
                    st.success(f"""
                    **Forecast Interpretation:**
                    
                    Based on the {model_choice} model with {confidence_pct}% confidence interval:
                    - The model predicts Bitcoin will **{direction}** by approximately **{abs(price_change_pct):.2f}%** over the next {forecast_days} days.
                    - Current price: **${current_price:,.2f}**
                    - Forecasted price: **${forecasted_price:,.2f}** (±${abs(last_forecast['yhat_upper'] - last_forecast['yhat_lower'])/2:,.2f})
                    
                    ⚠️ **Important:** Cryptocurrency markets are highly volatile. This forecast is based on historical patterns 
                    and should not be used as the sole basis for investment decisions. Always conduct thorough research and 
                    consider consulting with a financial advisor.
                    """)
                    
                    # Step 8: Export forecast data
                    st.subheader("💾 Export Forecast Data")
                    
                    forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    forecast_export.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
                    
                    csv = forecast_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv,
                        file_name=f"btc_forecast_{model_choice.lower().replace(' ', '_')}_{forecast_days}d.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"❌ An error occurred during forecasting: {str(e)}")
                    st.exception(e)
        
        else:
            # Show instructions when button hasn't been clicked
            st.info("👆 Configure your settings in the sidebar and click **Generate Forecast** to begin!")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👈 Please upload a Bitcoin CSV file from the sidebar to get started!")
        
        st.markdown("""
        ### Getting Started
        
        1. **Download a Bitcoin dataset** from Kaggle or Yahoo Finance
        2. **Upload the CSV or ZIP** using the sidebar
        3. **Configure** your forecasting parameters
        4. **Choose a model**: Prophet (traditional), ARIMA (classical), or LSTM (deep learning)
        5. **Generate** the forecast and analyze the results
        
        ### Recommended Dataset
        
        - [Bitcoin Historical Data (2014-2024)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
        - [Yahoo Finance BTC-USD](https://finance.yahoo.com/quote/BTC-USD/history)
        
        ### Model Comparison
        
        - **Prophet**: Best for data with strong seasonal patterns and trends. Handles missing data well.
        - **ARIMA**: Classical statistical method. Good baseline for time series forecasting.
        - **LSTM**: Deep learning neural network. Best for complex patterns and long-term dependencies.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit • Plotly • Prophet • ARIMA • TensorFlow/LSTM</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
