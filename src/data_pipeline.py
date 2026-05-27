import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings

# Suppress warnings for cleaner execution
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Loading daily dataset...")
# Load the cleaned daily data
df = pd.read_csv('../data/processed/Cleaned_Daily_Final.csv', parse_dates=['Date'])

# Ensure chronological order (critical for time-series and rolling windows)
df = df.sort_values('Date').reset_index(drop=True)

# Define the core continuous features
core_features = [
    'AVERAGE TEMPERATURE (Deg C)', 'MAXIMUM TEMPERATURE (Deg C)',
    'MINIMUM TEMPERATURE (Deg C)', 'RAINFALL (mm)',
    'RELATIVE HUMIDITY (%)', 'SOLAR RADIATION (MJ/m2)',
    'WIND SPEED (kt)', 'EVAPOTRANSPIRATION (mm)'
]

# ==========================================
# 1. DATA INTEGRITY & OUTLIER TREATMENT
# ==========================================
print("Running Anomaly Detection (Isolation Forest)...")

# Machine learning models cannot handle NaNs during training. 
# We temporarily interpolate missing values just for the Isolation Forest to run.
df_imputed = df.copy()

# Set index to Date for time interpolation
df_imputed.set_index('Date', inplace=True)
for col in core_features:
    df_imputed[col] = df_imputed[col].interpolate(method='time').bfill().ffill()
df_imputed.reset_index(inplace=True)

# Initialize Isolation Forest
# contamination=0.01 assumes ~1% of the data points are severe hardware/logging errors
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

# Fit and predict (-1 = Outlier, 1 = Normal)
df['Anomaly_Flag'] = iso_forest.fit_predict(df_imputed[core_features])
df['Is_Outlier'] = df['Anomaly_Flag'] == -1
df.drop(columns=['Anomaly_Flag'], inplace=True)

print(f"Detected {df['Is_Outlier'].sum()} anomalous records out of {len(df)}.")

# Treat the Outliers: Replace true anomalies with NaN, then interpolate them using surrounding days
# This smooths out hardware spikes without losing the temporal sequence
for col in core_features:
    df.loc[df['Is_Outlier'], col] = np.nan

# Set index to Date for time interpolation
df.set_index('Date', inplace=True)
for col in core_features:
    df[col] = df[col].interpolate(method='time').bfill().ffill()
df.reset_index(inplace=True)


# ==========================================
# 2. FEATURE ENGINEERING FOR TIME-SERIES
# ==========================================
print("Engineering temporal features...")

# Targets we specifically want historical context for (e.g., to predict tomorrow's weather)
target_vars = ['MAXIMUM TEMPERATURE (Deg C)', 'RAINFALL (mm)']

# A. LAGGED FEATURES (What happened 1, 2, and 3 days ago?)
for col in target_vars:
    for lag in [1, 2, 3]:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)

# B. ROLLING STATISTICS (Smooth trends over the last week and month)
for col in target_vars:
    # 7-day and 30-day moving averages
    df[f'{col}_7d_mean'] = df[col].rolling(window=7).mean()
    df[f'{col}_30d_mean'] = df[col].rolling(window=30).mean()
    
    # Rolling standard deviation (Captures current weather volatility/storminess)
    df[f'{col}_7d_std'] = df[col].rolling(window=7).std()

# C. CYCLICAL ENCODING (Teaching the ML model the shape of a year)
# Extract numeric month and day of year
df['Month_Num'] = df['Date'].dt.month
df['DayOfYear'] = df['Date'].dt.dayofyear

# Sine and Cosine transformations for Month (Period = 12)
df['Month_sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12.0)
df['Month_cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12.0)

# Sine and Cosine transformations for Day of Year (Period = ~365.25)
df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

# Drop temporary cyclical helper columns
df.drop(columns=['Month_Num', 'DayOfYear'], inplace=True)

# ==========================================
# 3. FINAL CLEANUP & EXPORT
# ==========================================
# Lagging and rolling introduces NaNs at the very beginning of the dataset (e.g., day 1 has no "day -1" to look back at).
# We must drop these initial rows so the ML model has a perfectly clean matrix.
df_ml_ready = df.dropna().reset_index(drop=True)

print(f"Final ML-Ready Dataset Shape: {df_ml_ready.shape}")

# Save to environment
df_ml_ready.to_csv("ML_Ready_Daily.csv", index=False)

# If running in Colab, trigger download automatically: