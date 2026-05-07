import os
import random
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score
from torch.utils.data import TensorDataset, DataLoader

# Ensure these classes match your local filenames/class names exactly
from LoadRegressor import LoadRegressor
from StressClassifier import StressClassifier

# -------------------------
# Setup
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_prep_data(weather_path, load_path1, load_path2):
    weather_df = pd.read_csv(weather_path, low_memory=False)
    weather_df['join_key'] = weather_df['DATE'].astype(str).str.slice(0, 10)

    l1 = pd.read_csv(load_path1)
    l2 = pd.read_csv(load_path2)
    load_combined = pd.concat([l1, l2], axis=0, ignore_index=True)

    time_col = \
    [c for c in load_combined.columns if 'time' in c.lower() or 'utc' in c.lower() or 'datetime' in c.lower()][0]
    load_combined['temp_dt'] = pd.to_datetime(load_combined[time_col], errors='coerce')
    load_combined = load_combined.dropna(subset=['temp_dt']).copy()
    load_combined['join_key'] = load_combined['temp_dt'].dt.strftime('%Y-%m-%d')

    load_col = [c for c in load_combined.columns if c.lower() in ('mw', 'load', 'demand')][0]
    load_combined = load_combined.rename(columns={load_col: 'mw'})

    useful_cols = ['TAVG', 'TMAX', 'TMIN', 'join_key']
    weather_slim = weather_df[[c for c in useful_cols if c in weather_df.columns]].drop_duplicates(subset=['join_key'])

    merged_df = pd.merge(load_combined, weather_slim, on='join_key', how='inner')
    merged_df = merged_df.set_index('temp_dt').sort_index()
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    return merged_df


def get_features(df, use_fourier=False):
    df = df.copy()

    if 'TAVG' not in df.columns or df['TAVG'].isna().all():
        if 'TMAX' in df.columns and 'TMIN' in df.columns:
            df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2
        else:
            df['TAVG'] = 0.0

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['load_lag_1h'] = df['mw'].shift(1)
    df['load_lag_24h'] = df['mw'].shift(24)
    df['temp_rolling_3h'] = df['TAVG'].rolling(window=3, min_periods=1).mean()

    # Classification Target
    threshold = df['mw'].quantile(0.90)
    df['grid_stress'] = (df['mw'] > threshold).astype(int)

    base_cols = ['TAVG', 'month', 'is_weekend', 'load_lag_1h', 'load_lag_24h', 'temp_rolling_3h']

    if use_fourier:
        df['h_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['h_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['d_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['d_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        feature_cols = base_cols + ['h_sin', 'h_cos', 'd_sin', 'd_cos']
    else:
        feature_cols = base_cols + ['hour', 'day_of_week']

    df = df.ffill().bfill()
    df_final = df.dropna(subset=['mw', 'grid_stress'])
    return df_final[feature_cols].values, df_final['mw'].values.reshape(-1, 1), df_final['grid_stress'].values.reshape(
        -1, 1)


def train_model(X_train, y_train, model_type="regressor", name="Model"):
    input_size = X_train.shape[1]
    if model_type == "regressor":
        model = LoadRegressor(input_size=input_size).to(DEVICE)
        criterion = nn.MSELoss()
    else:
        model = StressClassifier(input_size=input_size).to(DEVICE)
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=False)

    print(f"\n--- Training {name} ({model_type.upper()}) ---")
    model.train()
    for epoch in range(1, 51):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Avg Loss: {epoch_loss / len(loader):.6f}")

    return model


if __name__ == "__main__":
    try:
        print(f"Initializing process on {DEVICE}...")
        raw_data = load_and_prep_data("Data/4280832.csv", "Data/hrl_load_metered.csv", "Data/hrl_load_metered (1).csv")

        X1_raw, y1_reg_raw, y1_clf_raw = get_features(raw_data, use_fourier=False)
        X2_raw, y2_reg_raw, _ = get_features(raw_data, use_fourier=True)

        s1_x, s1_y = MinMaxScaler(), MinMaxScaler()
        s2_x, s2_y = MinMaxScaler(), MinMaxScaler()

        X1_s = torch.tensor(s1_x.fit_transform(X1_raw)).float().to(DEVICE)
        y1_reg_s = torch.tensor(s1_y.fit_transform(y1_reg_raw)).float().to(DEVICE)
        y1_clf_s = torch.tensor(y1_clf_raw).float().to(DEVICE)

        X2_s = torch.tensor(s2_x.fit_transform(X2_raw)).float().to(DEVICE)
        y2_reg_s = torch.tensor(s2_y.fit_transform(y2_reg_raw)).float().to(DEVICE)

        split = int(len(X1_s) * 0.8)

        # Training the three requested models
        reg_orig = train_model(X1_s[:split], y1_reg_s[:split], "regressor", "Original")
        clf_orig = train_model(X1_s[:split], y1_clf_s[:split], "classifier", "Original")
        reg_four = train_model(X2_s[:split], y2_reg_s[:split], "regressor", "Fourier")

        print("\n" + "=" * 50)
        print("FINAL EVALUATION METRICS")
        print("=" * 50)

        reg_orig.eval();
        reg_four.eval();
        clf_orig.eval()
        with torch.no_grad():
            # Regression predictions
            p1_reg = s1_y.inverse_transform(reg_orig(X1_s[split:]).cpu().numpy())
            p2_reg = s2_y.inverse_transform(reg_four(X2_s[split:]).cpu().numpy())
            actual_reg = s1_y.inverse_transform(y1_reg_s[split:].cpu().numpy())

            # Classification predictions
            p1_clf = (clf_orig(X1_s[split:]).cpu().numpy() > 0.5).astype(int)
            actual_clf = y1_clf_s[split:].cpu().numpy()

            # Metric Calculations
            rmse_orig = np.sqrt(mean_squared_error(actual_reg, p1_reg))
            mae_orig = mean_absolute_error(actual_reg, p1_reg)
            r2_orig = r2_score(actual_reg, p1_reg)

            rmse_four = np.sqrt(mean_squared_error(actual_reg, p2_reg))
            mae_four = mean_absolute_error(actual_reg, p2_reg)
            r2_four = r2_score(actual_reg, p2_reg)

            f1_orig = f1_score(actual_clf, p1_clf)

            print(f"{'Metric':<20} | {'Original MLP':<15} | {'Fourier MLP':<15}")
            print("-" * 55)
            print(f"{'Regressor R²':<20} | {r2_orig:<15.4f} | {r2_four:<15.4f}")
            print(f"{'Regressor RMSE':<20} | {rmse_orig:<15.2f} | {rmse_four:<15.2f}")
            print(f"{'Regressor MAE':<20} | {mae_orig:<15.2f} | {mae_four:<15.2f}")
            print("-" * 55)
            print(f"{'Classifier F1':<20} | {f1_orig:<15.4f} | {'N/A':<15}")
            print("=" * 55)

            # Plotting
            n = 200
            plt.figure(figsize=(14, 7))
            plt.plot(actual_reg[:n], label='Actual Load', color='black', lw=2)
            plt.plot(p1_reg[:n], label=f'Original (R²: {r2_orig:.2f})', ls=':', color='blue')
            plt.plot(p2_reg[:n], label=f'Fourier (R²: {r2_four:.2f})', ls='--', color='red')
            plt.title("Regression Comparison: Original vs. Fourier Features")
            plt.ylabel("Power Demand (MW)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")