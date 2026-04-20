"""
Bitcoin Data Downsampler
=========================
Helper script to convert minute-level Bitcoin data to daily data.

This reduces file size from ~200MB to ~1MB, making it easier to upload.

Usage:
    python downsample_bitcoin_data.py

The script will:
1. Look for 'btcusd_1-min_data.csv' in the current directory
2. Convert minute-level data to daily data
3. Save as 'btcusd_daily.csv' (much smaller file)
"""

import pandas as pd
import sys

def downsample_bitcoin_data(input_file='btcusd_1-min_data.csv', output_file='btcusd_daily.csv'):
    """
    Downsample minute-level Bitcoin data to daily frequency.
    
    Args:
        input_file: Path to the input CSV file (minute-level data)
        output_file: Path to save the daily CSV file
    """
    
    print("=" * 60)
    print("Bitcoin Data Downsampler")
    print("=" * 60)
    print()
    
    # Step 1: Read the CSV file
    print(f"📖 Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df):,} rows")
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found!")
        print()
        print("Please make sure the file is in the same directory as this script.")
        print("Or specify the correct path:")
        print(f"    python downsample_bitcoin_data.py <input_file> <output_file>")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        sys.exit(1)
    
    print()
    print("📊 Original data info:")
    print(f"   - Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   - Columns: {', '.join(df.columns)}")
    print(f"   - Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print()
    
    # Step 2: Convert timestamp to datetime
    print("🔄 Converting timestamp to datetime...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    print("✅ Conversion complete")
    print()
    
    # Step 3: Set timestamp as index
    print("🔄 Setting timestamp as index...")
    df = df.set_index('Timestamp')
    print("✅ Index set")
    print()
    
    # Step 4: Resample to daily frequency
    print("🔄 Resampling to daily frequency...")
    print("   Strategy:")
    print("   - Open: First value of the day")
    print("   - High: Maximum value of the day")
    print("   - Low: Minimum value of the day")
    print("   - Close: Last value of the day")
    print("   - Volume: Sum of the day")
    print()
    
    # Resample to daily (D) frequency
    # Use appropriate aggregation for each column
    df_daily = df.resample('D').agg({
        'Open': 'first',      # First open price of the day
        'High': 'max',        # Highest price of the day
        'Low': 'min',         # Lowest price of the day
        'Close': 'last',      # Last close price of the day
        'Volume_(BTC)': 'sum',       # Total volume in BTC
        'Volume_(Currency)': 'sum',  # Total volume in currency
        'Weighted_Price': 'mean'     # Average weighted price
    })
    
    # Drop rows with NaN values (days with no trading data)
    df_daily = df_daily.dropna()
    
    print(f"✅ Resampled to {len(df_daily):,} daily records")
    print()
    
    # Step 5: Reset index to make Timestamp a column again
    print("🔄 Resetting index...")
    df_daily = df_daily.reset_index()
    
    # Rename Timestamp to Date for clarity
    df_daily = df_daily.rename(columns={'Timestamp': 'Date'})
    print("✅ Index reset")
    print()
    
    # Step 6: Save to CSV
    print(f"💾 Saving to {output_file}...")
    df_daily.to_csv(output_file, index=False)
    
    # Get file sizes
    import os
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    new_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    
    print("✅ Save complete!")
    print()
    print("=" * 60)
    print("📈 Summary")
    print("=" * 60)
    print(f"Original file:  {input_file}")
    print(f"   - Rows: {len(df):,}")
    print(f"   - Size: {original_size:.2f} MB")
    print()
    print(f"Daily file:     {output_file}")
    print(f"   - Rows: {len(df_daily):,}")
    print(f"   - Size: {new_size:.2f} MB")
    print(f"   - Reduction: {((original_size - new_size) / original_size * 100):.1f}%")
    print()
    print(f"📊 Daily data preview:")
    print(df_daily.head(10).to_string(index=False))
    print()
    print("=" * 60)
    print("✅ Done! You can now upload the daily file to the Streamlit app.")
    print("=" * 60)


if __name__ == "__main__":
    # Check if custom file paths are provided
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        downsample_bitcoin_data(input_file, output_file)
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        downsample_bitcoin_data(input_file)
    else:
        # Use default file names
        downsample_bitcoin_data()
