import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, roc_auc_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
import shap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import matplotlib.dates as mdates

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = '#FAFAFA';  DARK  = '#1D3557';  RED  = '#E63946'
BLUE  = '#457B9D';  TEAL  = '#2A9D8F';  GOLD = '#E9C46A'
GREEN = '#2DC653';  ORANGE= '#F4A261';  PURPLE='#7B2FBE'

MONTH_ORDER = ['Jan','Feb','Mar','April','May','June',
               'July','Aug','Sep','Oct','Nov','Dec']
M_MAP = {'Apr':'April','Jun':'June','Jul':'July'}

# 1. DATA LOADING
def load_data(daily_path, monthly_path):
    daily   = pd.read_csv(daily_path)
    monthly = pd.read_csv(monthly_path)
    for df in [daily, monthly]:
        df['Month'] = df['Month'].replace(M_MAP)
    monthly['Month_Num'] = monthly['Month'].map({m:i+1 for i,m in enumerate(MONTH_ORDER)})
    monthly['Date'] = pd.to_datetime(
        monthly['Year'].astype(int).astype(str)+'-'+
        monthly['Month_Num'].astype(int).astype(str).str.zfill(2)+'-01')
    monthly = monthly.sort_values('Date').reset_index(drop=True)
    daily['Date']      = pd.to_datetime(daily['Date'])
    daily['Month_Num'] = daily['Month'].map({m:i+1 for i,m in enumerate(MONTH_ORDER)})
    daily = daily.sort_values('Date').reset_index(drop=True)
    return daily, monthly


# 2. INTELLIGENCE BLOCK FEATURE ENGINEERING
def build_intelligence_blocks(df, is_daily=True):
    """
    Compute all intelligence-block features and derived indices.

    Block A  Atmosphere     → heat stress proxy, wind vector, pressure tendency
    Block B  Water Balance  → rain intensity class, moisture proxy
    Block C  Solar/Energy   → ET drivers, radiation efficiency
    Block D  Indices        → CSI, FRI, RPS, IDD, WHI  (all 0–100 normalised)
    """
    d = df.copy()

    # ── cyclic encodings ────────────────────────────────────────────────────
    d['month_sin'] = np.sin(2*np.pi * d['Month_Num'] / 12)
    d['month_cos'] = np.cos(2*np.pi * d['Month_Num'] / 12)
    if 'Day' in d.columns:
        d['doy'] = d['Date'].dt.dayofyear
        d['doy_sin'] = np.sin(2*np.pi * d['doy'] / 365)
        d['doy_cos'] = np.cos(2*np.pi * d['doy'] / 365)
    d['year_scaled'] = (d['Year'] - d['Year'].min()) / 24.0

    # ── Block A: Atmosphere ────────────────────────────────────────────────
    T   = d['AVERAGE TEMPERATURE (Deg C)']
    Tx  = d['MAXIMUM TEMPERATURE (Deg C)']
    Tn  = d['MINIMUM TEMPERATURE (Deg C)']
    RH  = d['RELATIVE HUMIDITY (%)']
    WS  = d['WIND SPEED (kt)']
    P   = d['BAROMETRIC PRESSURE (hPa)']
    WD  = d['WIND DIRECTION (Degrees)']

    d['DTR']          = Tx - Tn                            # diurnal range
    d['heat_index']   = T + 0.33*RH/10 - 4.0              # simplified HI
    d['wind_dir_sin'] = np.sin(np.radians(WD))
    d['wind_dir_cos'] = np.cos(np.radians(WD))
    d['press_lag1']   = P.shift(1)
    d['press_tend']   = P.diff(1)                          # pressure tendency
    d['press_tend3']  = P.diff(3)                          # 3-period tendency

    # Humidity deficit (vapour pressure deficit proxy)
    d['VPD_proxy']    = (1 - RH/100) * (0.6108 * np.exp(17.27*T/(T+237.3)))

    # ── Block B: Water Balance ─────────────────────────────────────────────
    R = d['RAINFALL (mm)']
    d['log_rain']      = np.log1p(R)
    d['rain_binary']   = (R > 0.1).astype(float)
    d['rain_lag1']     = R.shift(1)
    d['rain_lag3']     = R.shift(3)
    d['rain_roll7']    = R.shift(1).rolling(7, min_periods=1).sum()
    d['rain_roll30']   = R.shift(1).rolling(30, min_periods=1).sum()
    d['rain_deficit']  = d['rain_roll30'] - d['rain_roll30'].mean()  # anomaly

    # Cloud cover change (storm precursor)
    CC = d['CLOUD COVER (Oktas)']
    d['cloud_tend']    = CC.diff(1)
    d['cloud_tend3']   = CC.diff(3)
    d['hum_tend']      = RH.diff(1)

    # ── Block C: Solar & Energy ────────────────────────────────────────────
    SR  = d['SOLAR RADIATION (MJ/m2)']
    SH  = d['SUNSHINE HOURS']
    ET  = d['EVAPOTRANSPIRATION (mm)']
    DP  = d['DEWPOINT TEMPERATURE (Deg C)']

    d['solar_x_tmax']  = SR * Tx
    d['solar_x_vpd']   = SR * d['VPD_proxy']
    d['PM_proxy']       = SR * d['DTR'] / (RH + 0.01)   # Penman-Monteith proxy
    d['rad_efficiency'] = SH / (SR + 0.01)               # cloud suppression factor
    d['dew_depression'] = T - DP                          # T-Td (aridity signal)

    # ET lags & rolls (for ET forecasting)
    d['ET_lag1']   = ET.shift(1)
    d['ET_lag2']   = ET.shift(2)
    d['ET_lag12']  = ET.shift(12)
    d['ET_roll3']  = ET.shift(1).rolling(3, min_periods=1).mean()
    d['ET_roll6']  = ET.shift(1).rolling(6, min_periods=1).mean()

    # ── Block D: Derived Intelligence Indices (0–100 scale) ───────────────

    # 1. Crop Stress Index (CSI)
    #    High temp + low humidity + high solar + high wind → 0=calm, 100=extreme
    heat_score  = np.clip((Tx - 25) / 20, 0, 1)          # 25°→45°C maps 0→1
    dry_score   = np.clip((60 - RH) / 60, 0, 1)          # 60%→0% maps 0→1
    rad_score   = np.clip((SR - 10) / 20, 0, 1)          # 10→30 MJ maps 0→1
    wind_score  = np.clip((WS - 3) / 8, 0, 1)            # 3→11 kt maps 0→1
    d['CSI']    = (0.35*heat_score + 0.35*dry_score +
                   0.20*rad_score  + 0.10*wind_score) * 100

    # 2. Fire Risk Index (FRI) — critical for Northern Nigeria
    #    Low RH + high temp + high wind + dry antecedent → bushfire danger
    d['FRI']    = (0.40*dry_score + 0.30*heat_score +
                   0.20*wind_score +
                   0.10*np.clip(1 - d['rain_roll7']/30, 0, 1)) * 100

    # 3. Rain Prediction Score (RPS) — local nowcast signal
    #    Pressure drop + humidity rise + cloud build-up + wind from S/SW
    #    (physical nowcasting, no ML needed for real-time)
    press_signal = np.clip(-d['press_tend'] / 2, 0, 1)
    hum_signal   = np.clip(d['hum_tend'] / 15, 0, 1)
    cloud_signal = np.clip(CC / 8, 0, 1)
    south_wind   = np.clip(np.sin(np.radians(WD - 180)), 0, 1)  # southerly
    d['RPS']    = (0.30*press_signal + 0.25*hum_signal  +
                   0.30*cloud_signal + 0.15*south_wind) * 100

    # 4. Irrigation Demand Driver (IDD) — when should farmers irrigate?
    #    High ET + low recent rainfall + high VPD
    et_scaled   = np.clip((ET - 3) / 7, 0, 1)
    low_rain    = np.clip(1 - d['rain_roll7']/30, 0, 1)
    vpd_scaled  = np.clip(d['VPD_proxy'] / 3, 0, 1)
    d['IDD']    = (0.40*et_scaled + 0.35*low_rain + 0.25*vpd_scaled) * 100

    # 5. Weather Hazard Index (WHI) — general hazard (flood + wind + heat)
    flood_risk  = np.clip(R / 50, 0, 1)
    heat_haz    = np.clip((Tx - 35) / 10, 0, 1)
    wind_haz    = np.clip((WS - 5) / 5, 0, 1)
    d['WHI']    = (0.40*flood_risk + 0.35*heat_haz + 0.25*wind_haz) * 100

    return d


# 3. LEVEL 1 DECISION ENGINE
#    Rule-based thresholds → farm alerts 
ALERT_RULES = {
    "Fungal Risk Spike": {
        "condition": lambda r: r['RELATIVE HUMIDITY (%)'] > 85 and 20 <= r['AVERAGE TEMPERATURE (Deg C)'] <= 30,
        "crops": "Tomato, Pepper, Onion",
        "sms":   "Heavy moist air detected. Check leaves for spots. Apply fungicide if needed.",
        "severity": "MEDIUM"
    },
    "Heat Stress": {
        "condition": lambda r: r['MAXIMUM TEMPERATURE (Deg C)'] > 38,
        "crops": "Cowpea, Groundnut, Maize",
        "sms":   "Extreme heat may cause flower drop and yield loss. Irrigate in the evening.",
        "severity": "HIGH"
    },
    "Extreme Heat": {
        "condition": lambda r: r['MAXIMUM TEMPERATURE (Deg C)'] > 42,
        "crops": "All crops",
        "sms":   "DANGER: Temperatures above 42°C. Emergency irrigation needed. Mulch immediately.",
        "severity": "CRITICAL"
    },
    "Dry Heat — Pest Risk": {
        "condition": lambda r: r['RELATIVE HUMIDITY (%)'] < 40 and r['MAXIMUM TEMPERATURE (Deg C)'] > 32,
        "crops": "Cowpea, Sorghum",
        "sms":   "Dry heat favours aphids and thrips. Inspect crops. Consider neem spray.",
        "severity": "MEDIUM"
    },
    "Wind Damage Risk": {
        "condition": lambda r: r['WIND SPEED (kt)'] > 8,
        "crops": "All standing crops",
        "sms":   "Strong winds expected. Stake tall crops and secure structures.",
        "severity": "HIGH"
    },
    "Fire Risk Alert": {
        "condition": lambda r: r.get('FRI', 0) > 65,
        "crops": "All crops, grassland",
        "sms":   "High bushfire danger. Low humidity, hot and windy conditions. Do not burn.",
        "severity": "HIGH"
    },
    "Irrigation Required": {
        "condition": lambda r: r.get('IDD', 0) > 65,
        "crops": "All irrigated crops",
        "sms":   "High water demand today. Irrigate before 7am or after 6pm to reduce evaporation.",
        "severity": "INFO"
    },
    "Rain Likely": {
        "condition": lambda r: r.get('RPS', 0) > 60,
        "crops": "All crops",
        "sms":   "Signs of approaching rain. Hold off irrigation and spraying today.",
        "severity": "INFO"
    },
    "Planting Window Open": {
        "condition": lambda r: (
            r['AVERAGE TEMPERATURE (Deg C)'] >= 25 and
            r['RELATIVE HUMIDITY (%)'] >= 55 and
            r.get('rain_roll7', 0) >= 10
        ),
        "crops": "Millet, Sorghum, Cowpea",
        "sms":   "Good planting conditions. Soil warm and moisture adequate. Plant within 48 hours.",
        "severity": "POSITIVE"
    },
    "High Crop Stress": {
        "condition": lambda r: r.get('CSI', 0) > 60,
        "crops": "All crops",
        "sms":   "Crop stress is high today. Heat, dryness and wind are combined. Shade or irrigate.",
        "severity": "HIGH"
    },
}

SEVERITY_COLOR = {
    "CRITICAL": "#8B0000", "HIGH": RED, "MEDIUM": GOLD,
    "INFO": BLUE, "POSITIVE": GREEN
}

def run_decision_engine(df):
    """Apply all Level-1 rules to each row. Returns alert log DataFrame."""
    records = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        for name, rule in ALERT_RULES.items():
            try:
                triggered = rule["condition"](row_dict)
            except Exception:
                triggered = False
            if triggered:
                records.append({
                    "Date": row['Date'],
                    "Alert": name,
                    "Severity": rule["severity"],
                    "Crops": rule["crops"],
                    "SMS_Text": rule["sms"],
                })
    return pd.DataFrame(records)


# 4. ML MODELS
def build_regressors():
    return {
        "Ridge":    Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "RF":       RandomForestRegressor(n_estimators=400, max_features='sqrt',
                                           min_samples_leaf=2, random_state=42, n_jobs=-1),
        "XGBoost":  xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                                      verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                                       verbose=-1),
    }

def build_classifiers():
    return {
        "Logistic":  Pipeline([("sc", StandardScaler()),
                                ("m",  LogisticRegression(class_weight='balanced',
                                                           max_iter=500, C=1.0))]),
        "RF-Clf":    RandomForestClassifier(n_estimators=400, class_weight='balanced',
                                             max_depth=10, random_state=42, n_jobs=-1),
        "XGB-Clf":   xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                         scale_pos_weight=4, random_state=42, verbosity=0,
                                         eval_metric='logloss'),
        "LGBM-Clf":  lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                          class_weight='balanced', random_state=42, verbose=-1),
    }


# ── metric helpers ─────────────────────────────────────────────────────────
def nse(obs, sim):
    return 1 - np.sum((obs-sim)**2)/np.sum((obs-obs.mean())**2)

def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0,1]
    return 1 - np.sqrt((r-1)**2 + (sim.std()/obs.std()-1)**2 + (sim.mean()/obs.mean()-1)**2)

def reg_metrics(obs, pred, label=""):
    rmse=np.sqrt(mean_squared_error(obs,pred))
    mae=mean_absolute_error(obs,pred)
    r2=r2_score(obs,pred)
    nse_=nse(obs,pred); kge_=kge(obs,pred)
    mbe=np.mean(pred-obs)
    print(f"  {label:12s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  NSE={nse_:.4f}  KGE={kge_:.4f}  MBE={mbe:+.4f}")
    return dict(Model=label,RMSE=rmse,MAE=mae,R2=r2,NSE=nse_,KGE=kge_,MBE=mbe)

def clf_metrics(obs, probs, pred, label=""):
    auc=roc_auc_score(obs,probs)
    f1 =f1_score(obs,pred)
    acc=(obs==pred).mean()
    print(f"  {label:12s}  AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
    return dict(Model=label,AUC=auc,F1=f1,Accuracy=acc)


# 5. EXPERIMENT A — ET₀ ESTIMATION  (monthly)
def experiment_et0(daily, monthly, outdir):
    print("\n" + "█"*70)
    print("  EXPERIMENT A  ·  ET₀ Estimation  (monthly 2001–2025)")
    print("█"*70)
    df = build_intelligence_blocks(monthly, is_daily=False)
    TARGET = 'EVAPOTRANSPIRATION (mm)'
    exclude = ['Year','Month','Month_Num','Date','RAINFALL (mm)',TARGET,
               'CSI','FRI','RPS','IDD','WHI']
    feat_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feat_cols+[TARGET]).reset_index(drop=True)

    train_mask = df['Year'] <= 2020
    X_tr = df.loc[train_mask, feat_cols].values
    X_te = df.loc[~train_mask, feat_cols].values
    y_tr = df.loc[train_mask, TARGET].values
    y_te = df.loc[~train_mask, TARGET].values
    dates_te = df.loc[~train_mask, 'Date'].values

    models = build_regressors()
    results, trained, preds = [], {}, {}
    print(f"\n  Features: {len(feat_cols)}  |  Train: {train_mask.sum()}  |  Test: {(~train_mask).sum()}")
    print(f"\n{'─'*70}")
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        p = mdl.predict(X_te)
        results.append(reg_metrics(y_te, p, name))
        trained[name]=mdl; preds[name]=p
    res_df = pd.DataFrame(results).sort_values('NSE',ascending=False)

    _plot_obs_pred(y_te, preds, dates_te, 'ET₀ (mm/month)', outdir, 'et0')
    _plot_radar(res_df, outdir, 'et0_radar')
    _plot_residuals(y_te, preds, outdir, 'et0')

    # SHAP for tree models
    for nm in ['RF','XGBoost','LightGBM']:
        m = trained[nm]
        if hasattr(m,'feature_importances_'):
            _plot_shap(m, X_tr, X_te, feat_cols, nm, outdir, f'shap_et0_{nm.lower()}')

    # Bootstrap CI
    best = res_df.iloc[0]['Model']
    lo,hi,pm = _bootstrap_ci(trained[best], X_tr, y_tr, X_te, n_boot=150)
    _plot_uncertainty(y_te, pm, lo, hi, dates_te, best, 'ET₀ (mm/month)', outdir, 'et0_ci')

    _export_latex(res_df, outdir/'et0_results.tex',
                  "ET$_0$ estimation model comparison — Ibadan 2021–2025",
                  "tab:et0")
    res_df.to_csv(outdir/'et0_results.csv', index=False)
    return res_df, trained, feat_cols


# 6. EXPERIMENT B — RAIN PREDICTION CLASSIFIER  (daily)
def experiment_rain_classifier(daily, outdir):
    print("\n" + "█"*70)
    print("  EXPERIMENT B  ·  Next-Day Rain Prediction  (daily 2021–2025)")
    print("█"*70)
    df = build_intelligence_blocks(daily, is_daily=True)
    df['rain_tomorrow'] = (df['RAINFALL (mm)'].shift(-1) > 1.0).astype(int)

    TARGET = 'rain_tomorrow'
    exclude_base = ['Year','Month','Month_Num','Date','doy',
                    'RAINFALL (mm)','log_rain','rain_binary',
                    'EVAPOTRANSPIRATION (mm)','ET_lag1','ET_lag2','ET_roll3','ET_roll6',
                    TARGET]
    feat_cols = [c for c in df.columns if c not in exclude_base
                 and not c.startswith('ET_')]
    df = df.dropna(subset=feat_cols+[TARGET]).reset_index(drop=True)

    split = int(len(df)*0.80)
    X_tr, X_te = df[feat_cols].values[:split], df[feat_cols].values[split:]
    y_tr, y_te = df[TARGET].values[:split], df[TARGET].values[split:]
    dates_te   = df['Date'].values[split:]

    models = build_classifiers()
    results, trained, probs_all = [], {}, {}
    print(f"\n  Features: {len(feat_cols)}  |  Train: {split}  |  Test: {len(X_te)}")
    print(f"  Rain prevalence (test): {y_te.mean()*100:.1f}%\n{'─'*70}")
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        prob = mdl.predict_proba(X_te)[:,1]
        pred = (prob > 0.35).astype(int)
        results.append(clf_metrics(y_te, prob, pred, name))
        trained[name]=mdl; probs_all[name]=prob
    res_df = pd.DataFrame(results).sort_values('AUC',ascending=False)

    # Best model detail
    best = res_df.iloc[0]['Model']
    best_prob = probs_all[best]
    best_pred = (best_prob > 0.35).astype(int)
    print(f"\nBest model: {best}")
    print(classification_report(y_te, best_pred, target_names=['No Rain','Rain']))

    _plot_rain_dashboard(y_te, best_prob, best_pred, dates_te,
                         trained[best], X_tr, feat_cols, best, outdir)
    _plot_clf_radar(res_df, outdir, 'rain_radar')
    res_df.to_csv(outdir/'rain_results.csv', index=False)
    return res_df, trained


# 7. EXPERIMENT C — CROP STRESS INDEX FORECASTING  (daily, 3-day ahead)
def experiment_csi_forecast(daily, outdir):
    print("\n" + "█"*70)
    print("  EXPERIMENT C  ·  Crop Stress Index Forecast  (daily, 3-day ahead)")
    print("█"*70)
    df = build_intelligence_blocks(daily, is_daily=True)

    TARGET = 'CSI'
    # Target is 3 days ahead
    df['CSI_3d'] = df['CSI'].shift(-3)
    df['CSI_lag1'] = df['CSI'].shift(1)
    df['CSI_lag2'] = df['CSI'].shift(2)
    df['CSI_lag3'] = df['CSI'].shift(3)
    df['CSI_roll5'] = df['CSI'].shift(1).rolling(5, min_periods=1).mean()

    exclude = ['Year','Month','Month_Num','Date','doy',
               'RAINFALL (mm)','log_rain','rain_binary',
               'EVAPOTRANSPIRATION (mm)','CSI','FRI','RPS','IDD','WHI','CSI_3d']
    feat_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feat_cols+['CSI_3d']).reset_index(drop=True)

    split = int(len(df)*0.80)
    X_tr, X_te = df[feat_cols].values[:split], df[feat_cols].values[split:]
    y_tr, y_te = df['CSI_3d'].values[:split], df['CSI_3d'].values[split:]
    dates_te   = df['Date'].values[split:]

    models = build_regressors()
    results, trained, preds = [], {}, {}
    print(f"\n  Features: {len(feat_cols)}  |  Train: {split}  |  Test: {len(X_te)}")
    print(f"  CSI range: {df['CSI_3d'].min():.1f}–{df['CSI_3d'].max():.1f}\n{'─'*70}")
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        p = mdl.predict(X_te)
        results.append(reg_metrics(y_te, p, name))
        trained[name]=mdl; preds[name]=p
    res_df = pd.DataFrame(results).sort_values('R2', ascending=False)

    _plot_obs_pred(y_te, preds, dates_te, 'Crop Stress Index (0–100)', outdir, 'csi')
    res_df.to_csv(outdir/'csi_results.csv', index=False)
    return res_df, trained


# 8. EXPERIMENT D — MULTI-TARGET FORECASTING  (joint ET₀ + CSI + FRI)
def experiment_multitarget(daily, outdir):
    print("\n" + "█"*70)
    print("  EXPERIMENT D  ·  Multi-Target: ET₀ + CSI + FRI  (monthly)")
    print("█"*70)
    monthly = daily.groupby(['Year',pd.Grouper(key='Date',freq='ME')]).agg({
        col: 'mean' for col in daily.select_dtypes(include=np.number).columns
        if col not in ['Day', 'Year']
    }).reset_index()
    monthly['Month_Num'] = monthly['Date'].dt.month

    df = build_intelligence_blocks(monthly, is_daily=False)
    TARGETS = ['EVAPOTRANSPIRATION (mm)', 'CSI', 'FRI']
    exclude = ['Year','Date','Month_Num','RAINFALL (mm)'] + TARGETS + \
              ['IDD','RPS','WHI','Month']
    feat_cols = [c for c in df.columns if c not in exclude and
                 not c.startswith('ET_')]
    df = df.dropna(subset=feat_cols+TARGETS).reset_index(drop=True)
    split = int(len(df)*0.80)

    X_tr, X_te = df[feat_cols].values[:split], df[feat_cols].values[split:]
    Y_tr = df[TARGETS].values[:split]
    Y_te = df[TARGETS].values[split:]

    base = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              random_state=42, verbosity=0)
    mo = MultiOutputRegressor(base, n_jobs=-1)
    mo.fit(X_tr, Y_tr)
    Y_pred = mo.predict(X_te)

    print(f"\n  Train: {split}  |  Test: {len(X_te)}")
    for i, t in enumerate(TARGETS):
        reg_metrics(Y_te[:,i], Y_pred[:,i], t.split('(')[0].strip()[:15])

    _plot_multitarget(Y_te, Y_pred, TARGETS, df['Date'].values[split:], outdir)
    return mo


# 9. DECISION ENGINE VISUALISATION
def visualise_decision_engine(daily, outdir):
    print("\nRunning Level-1 Decision Engine…")
    df = build_intelligence_blocks(daily, is_daily=True)
    alerts = run_decision_engine(df)
    print(f"  Total alerts generated: {len(alerts)}")
    print(alerts['Alert'].value_counts().to_string())
    _plot_alert_dashboard(alerts, df, outdir)
    alerts.to_csv(outdir/'alert_log.csv', index=False)
    return alerts


# 10. INTELLIGENCE INDEX TIME-SERIES DASHBOARD
def plot_index_dashboard(daily, outdir):
    df = build_intelligence_blocks(daily, is_daily=True)
    # Monthly aggregate
    df_m = df.groupby(pd.Grouper(key='Date', freq='ME'))[
        ['CSI','FRI','RPS','IDD','WHI']].mean().reset_index()

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.patch.set_facecolor(BG)
    titles   = ['Crop Stress Index (CSI)', 'Fire Risk Index (FRI)',
                'Rain Prediction Score (RPS)', 'Irrigation Demand Driver (IDD)',
                'Weather Hazard Index (WHI)']
    cols_    = [RED, ORANGE, BLUE, TEAL, PURPLE]
    thresholds = [60, 65, 60, 65, 50]

    for i, (col, title, color, thresh) in enumerate(zip(
            ['CSI','FRI','RPS','IDD','WHI'], titles, cols_, thresholds)):
        ax = fig.add_subplot(5, 1, i+1)
        ax.set_facecolor(BG)
        values = df_m[col].values
        dates  = df_m['Date'].values

        # colour fill above/below threshold
        ax.fill_between(dates, values, thresh,
                         where=values>=thresh, color=color, alpha=0.40,
                         label=f'Above alert threshold ({thresh})')
        ax.fill_between(dates, values, thresh,
                         where=values<thresh,  color=color, alpha=0.15)
        ax.plot(dates, values, color=color, lw=1.8, zorder=4)
        ax.axhline(thresh, color='#555', lw=1.2, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 100)
        ax.set_ylabel(col, fontsize=9, color=DARK)
        ax.set_title(title, fontsize=10, fontweight='bold',
                      color=DARK, loc='left', pad=3)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=8)
        if i < 4: ax.set_xticklabels([])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # % time above threshold
        pct = (values >= thresh).mean() * 100
        ax.text(0.99, 0.85, f'{pct:.0f}% above threshold',
                 transform=ax.transAxes, ha='right', fontsize=8,
                 color=color, fontweight='bold')

    fig.suptitle('Micro-Climate Intelligence Indices  ·  Ibadan 2021–2025\n'
                  '(Monthly averages of daily-computed indices; dashed line = alert threshold)',
                  fontsize=13, fontweight='bold', color=DARK, y=1.01)
    plt.tight_layout()
    out = outdir/'index_dashboard.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


# 11. PLOT HELPERS (internal)
def _plot_obs_pred(y_te, preds, dates, ylabel, outdir, prefix):
    n = len(preds)
    fig = plt.figure(figsize=(16, 4.5*n), facecolor=BG)
    fig.patch.set_facecolor(BG)
    outer = gridspec.GridSpec(n, 2, hspace=0.55, wspace=0.30, width_ratios=[3,1])
    colors = plt.cm.tab10(np.linspace(0,1,n))
    y_arr = np.array(y_te)

    for i, (name, pred) in enumerate(preds.items()):
        col = colors[i]; pred_arr = np.array(pred)
        ax_ts = fig.add_subplot(outer[i,0])
        ax_ts.plot(dates, y_arr,    color=DARK, lw=2.0, label='Observed')
        ax_ts.plot(dates, pred_arr, color=col,  lw=1.6, linestyle='--',
                   label=name, alpha=0.88)
        r2_=r2_score(y_arr,pred_arr); nse_=nse(y_arr,pred_arr)
        ax_ts.set_title(f'{name}  |  R²={r2_:.3f}  NSE={nse_:.3f}',
                         fontsize=10, fontweight='bold', color=DARK, loc='left')
        ax_ts.set_ylabel(ylabel, fontsize=9)
        ax_ts.legend(fontsize=8, loc='upper right')
        ax_ts.grid(linestyle='--', alpha=0.4); ax_ts.set_facecolor(BG)
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_ts.tick_params(axis='x', rotation=45, labelsize=8)

        ax_sc = fig.add_subplot(outer[i,1])
        ax_sc.scatter(y_arr, pred_arr, color=col, alpha=0.65, s=22, edgecolors='none')
        lim=[min(y_arr.min(),pred_arr.min())-0.5, max(y_arr.max(),pred_arr.max())+0.5]
        ax_sc.plot(lim,lim,color='#555',lw=1.5,linestyle='--')
        ax_sc.set_xlim(lim); ax_sc.set_ylim(lim)
        ax_sc.set_xlabel('Observed',fontsize=9); ax_sc.set_ylabel('Predicted',fontsize=9)
        ax_sc.set_title('1:1 Plot',fontsize=10,fontweight='bold',color=DARK)
        ax_sc.grid(linestyle='--',alpha=0.4); ax_sc.set_facecolor(BG)

    fig.suptitle(f'Observed vs Predicted {ylabel}  ·  Ibadan\n'
                  f'Chronological test set (all models)',
                  fontsize=13, fontweight='bold', color=DARK, y=1.005)
    out = outdir/f'{prefix}_obs_pred.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def _plot_radar(res_df, outdir, fname):
    metrics_cols = ['RMSE','MAE','R2','NSE','KGE']
    labels = ['RMSE\n(↓better)','MAE\n(↓better)','R²','NSE','KGE']
    df = res_df[['Model']+metrics_cols].copy()
    df['RMSE'] = 1-(df['RMSE']-df['RMSE'].min())/(df['RMSE'].max()-df['RMSE'].min()+1e-9)
    df['MAE']  = 1-(df['MAE'] -df['MAE'].min()) /(df['MAE'].max() -df['MAE'].min() +1e-9)
    for c in ['R2','NSE','KGE']:
        mn,mx=df[c].min(),df[c].max()
        df[c]=(df[c]-mn)/(mx-mn+1e-9)
    N=5; angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(7,7),subplot_kw={'polar':True},facecolor='#0D1B2A')
    ax.set_facecolor('#0D1B2A')
    colors=plt.cm.Set2(np.linspace(0,1,len(df)))
    for (_,row),col in zip(df.iterrows(),colors):
        vals=row[metrics_cols].values.tolist(); vals+=vals[:1]
        ax.plot(angles,vals,lw=2.2,color=col,label=row['Model'])
        ax.fill(angles,vals,color=col,alpha=0.12)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels,color='white',fontsize=9)
    ax.set_yticklabels([]); ax.set_ylim(0,1)
    for r in np.arange(0.2,1.2,0.2): ax.plot(angles,[r]*(N+1),color='white',lw=0.3,alpha=0.2)
    ax.spines['polar'].set_color('#ffffff20'); ax.grid(color='white',lw=0.3,alpha=0.15)
    legend=ax.legend(loc='lower left',bbox_to_anchor=(-0.28,-0.05),
                      fontsize=9,framealpha=0.3,facecolor='#1a2a3a',labelcolor='white')
    fig.text(0.5,0.97,'Model Comparison Radar',ha='center',color='white',
             fontsize=12,fontweight='bold')
    out=outdir/f'{fname}.png'
    plt.savefig(out,dpi=200,bbox_inches='tight',facecolor='#0D1B2A')
    plt.close(); print(f"  Saved → {out}")


def _plot_clf_radar(res_df, outdir, fname):
    metrics_cols = ['AUC','F1','Accuracy']
    labels = ['AUC-ROC','F1-Score','Accuracy']
    df = res_df[['Model']+metrics_cols].copy()
    for c in metrics_cols:
        mn,mx=df[c].min(),df[c].max(); df[c]=(df[c]-mn)/(mx-mn+1e-9)
    N=3; angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(7,7),subplot_kw={'polar':True},facecolor='#0D1B2A')
    ax.set_facecolor('#0D1B2A')
    colors=plt.cm.Set2(np.linspace(0,1,len(df)))
    for (_,row),col in zip(df.iterrows(),colors):
        vals=row[metrics_cols].values.tolist(); vals+=vals[:1]
        ax.plot(angles,vals,lw=2.2,color=col,label=row['Model'])
        ax.fill(angles,vals,color=col,alpha=0.15)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels,color='white',fontsize=10)
    ax.set_yticklabels([]); ax.set_ylim(0,1)
    for r in np.arange(0.2,1.2,0.2): ax.plot(angles,[r]*(N+1),color='white',lw=0.3,alpha=0.2)
    ax.spines['polar'].set_color('#ffffff20'); ax.grid(color='white',lw=0.3,alpha=0.15)
    legend=ax.legend(loc='lower left',bbox_to_anchor=(-0.28,-0.05),
                      fontsize=9,framealpha=0.3,facecolor='#1a2a3a',labelcolor='white')
    fig.text(0.5,0.97,'Rain Classifier Comparison',ha='center',color='white',
             fontsize=12,fontweight='bold')
    out=outdir/f'{fname}.png'
    plt.savefig(out,dpi=200,bbox_inches='tight',facecolor='#0D1B2A')
    plt.close(); print(f"  Saved → {out}")


def _plot_residuals(y_te, preds, outdir, prefix):
    best = max(preds, key=lambda k: r2_score(y_te, preds[k]))
    resid = np.array(y_te) - np.array(preds[best])
    fig, axes = plt.subplots(1,3,figsize=(14,4.5),facecolor=BG)
    fig.patch.set_facecolor(BG)
    pred_arr=np.array(preds[best])
    axes[0].scatter(pred_arr, resid, color=BLUE, alpha=0.65, s=20, edgecolors='none')
    axes[0].axhline(0,color=RED,lw=1.5,linestyle='--')
    axes[0].set_xlabel('Fitted values',fontsize=10); axes[0].set_ylabel('Residuals',fontsize=10)
    axes[0].set_title('Residuals vs Fitted',fontweight='bold',fontsize=11)
    axes[0].grid(linestyle='--',alpha=0.4); axes[0].set_facecolor(BG)
    mu,sigma=resid.mean(),resid.std()
    axes[1].hist(resid, bins=20, color=BLUE, edgecolor='white', alpha=0.8, density=True)
    x=np.linspace(resid.min()-0.5,resid.max()+0.5,200)
    axes[1].plot(x,stats.norm.pdf(x,mu,sigma),color=RED,lw=2)
    axes[1].set_xlabel('Residual',fontsize=10); axes[1].set_ylabel('Density',fontsize=10)
    axes[1].set_title('Residual Distribution',fontweight='bold',fontsize=11)
    axes[1].grid(linestyle='--',alpha=0.4); axes[1].set_facecolor(BG)
    (osm,osr),(sl,ic,_)=stats.probplot(resid,dist='norm')
    axes[2].scatter(osm,osr,color=BLUE,alpha=0.7,s=20,edgecolors='none')
    axes[2].plot(osm,sl*np.array(osm)+ic,color=RED,lw=2,linestyle='--')
    axes[2].set_xlabel('Theoretical quantiles',fontsize=10)
    axes[2].set_ylabel('Sample quantiles',fontsize=10)
    axes[2].set_title('Normal Q–Q Plot',fontweight='bold',fontsize=11)
    axes[2].grid(linestyle='--',alpha=0.4); axes[2].set_facecolor(BG)
    fig.suptitle(f'Residual Diagnostics  ·  Best Model: {best}',
                  fontsize=12,fontweight='bold',color=DARK,y=1.01)
    plt.tight_layout()
    out=outdir/f'{prefix}_residuals.png'
    plt.savefig(out,dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def _plot_shap(model, X_tr, X_te, feat_names, model_name, outdir, fname):
    print(f"  Computing SHAP for {model_name}…", end=' ')
    exp = shap.TreeExplainer(model)
    sv  = exp.shap_values(X_te)
    fig, axes = plt.subplots(1,2,figsize=(14,6),facecolor=BG)
    fig.patch.set_facecolor(BG)
    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:12]
    axes[0].barh(range(len(order)),
                  mean_abs[order][::-1],
                  color=[plt.cm.viridis(v) for v in np.linspace(0.2,0.9,len(order))],
                  edgecolor='none', alpha=0.88)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([feat_names[i] for i in order[::-1]], fontsize=8)
    axes[0].set_xlabel('Mean |SHAP value|',fontsize=10)
    axes[0].set_title(f'Feature Importance ({model_name})',fontsize=11,
                       fontweight='bold',color=DARK)
    axes[0].grid(axis='x',linestyle='--',alpha=0.4); axes[0].set_facecolor(BG)
    top=order[:10][::-1]
    for i,fi in enumerate(top):
        sv_=sv[:,fi]; fv_=X_te[:,fi]
        axes[1].scatter(sv_, np.ones(len(sv_))*i+np.random.normal(0,0.08,len(sv_)),
                         c=fv_, cmap='RdBu_r', alpha=0.55, s=18,
                         norm=plt.Normalize(fv_.min(),fv_.max()))
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels([feat_names[i] for i in top],fontsize=8)
    axes[1].axvline(0,color='#999',lw=1)
    axes[1].set_xlabel('SHAP value',fontsize=10)
    axes[1].set_title('SHAP Beeswarm',fontsize=11,fontweight='bold',color=DARK)
    axes[1].grid(axis='x',linestyle='--',alpha=0.3); axes[1].set_facecolor(BG)
    plt.tight_layout()
    out=outdir/f'{fname}.png'
    plt.savefig(out,dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print(f"→ {out}")


def _bootstrap_ci(model, X_tr, y_tr, X_te, n_boot=150, ci=95):
    preds = np.zeros((n_boot, len(X_te)))
    for b in range(n_boot):
        idx = np.random.choice(len(X_tr), len(X_tr), replace=True)
        m   = deepcopy(model)
        m.fit(X_tr[idx] if isinstance(X_tr,np.ndarray) else X_tr.iloc[idx],
              y_tr[idx] if isinstance(y_tr,np.ndarray) else y_tr.iloc[idx])
        preds[b] = m.predict(X_te)
    lo=np.percentile(preds,(100-ci)/2,axis=0)
    hi=np.percentile(preds,100-(100-ci)/2,axis=0)
    return lo, hi, preds.mean(axis=0)


def _plot_uncertainty(y_te, pm, lo, hi, dates, model_name, ylabel, outdir, fname):
    fig, ax = plt.subplots(figsize=(13,5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.fill_between(dates, lo, hi, color=BLUE, alpha=0.22, label='95% Bootstrap CI')
    ax.plot(dates, pm,   color=BLUE, lw=2.0, label='Bootstrap mean', zorder=4)
    ax.plot(dates, y_te, color=RED,  lw=2.0, marker='o', ms=5,
             label='Observed', zorder=5, alpha=0.85)
    ax.set_ylabel(ylabel,fontsize=11,color=DARK)
    ax.set_title(f'Prediction Uncertainty  ·  {model_name}  ·  95% Bootstrap CI',
                  fontsize=12,fontweight='bold',color=DARK)
    ax.legend(fontsize=9); ax.grid(linestyle='--',alpha=0.4); ax.set_facecolor(BG)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.tick_params(axis='x',rotation=45,labelsize=8)
    coverage = np.mean((np.array(y_te)>=lo)&(np.array(y_te)<=hi))*100
    ax.text(0.02,0.97,f'CI coverage = {coverage:.1f}%',transform=ax.transAxes,
             fontsize=9,va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.85))
    plt.tight_layout()
    out=outdir/f'{fname}.png'
    plt.savefig(out,dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def _plot_rain_dashboard(y_te, prob, pred, dates, model, X_tr, feat_names, name, outdir):
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── top-left: predicted probability time-series ──
    ax1 = fig.add_subplot(gs[0,:])
    rain_dates  = dates[y_te==1]
    norain_dates= dates[y_te==0]
    ax1.plot(dates, prob, color=BLUE, lw=1.5, zorder=3, alpha=0.8,
              label='Predicted rain probability')
    ax1.axhline(0.35, color='#555', lw=1.2, linestyle='--',
                 label='Decision threshold (0.35)')
    ax1.fill_between(dates, prob, 0.35,
                      where=prob>=0.35, color=BLUE, alpha=0.25,
                      label='Predicted rain')
    # mark actual rain days
    ax1.scatter(rain_dates, np.ones(len(rain_dates))*0.95,
                 marker='|', color=RED, s=50, lw=1.5, zorder=5,
                 label='Actual rain day')
    ax1.set_ylim(0,1.05); ax1.set_ylabel('Rain Probability',fontsize=10)
    ax1.set_title(f'Next-Day Rain Probability  ·  {name}  |  AUC = {roc_auc_score(y_te,prob):.3f}',
                   fontsize=12, fontweight='bold', color=DARK, loc='left')
    ax1.legend(fontsize=8.5, loc='upper right', ncol=2)
    ax1.grid(linestyle='--', alpha=0.35); ax1.set_facecolor(BG)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    # ── bottom-left: ROC curve ──
    from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
    fpr,tpr,_ = roc_curve(y_te, prob)
    auc = roc_auc_score(y_te, prob)
    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(fpr, tpr, color=BLUE, lw=2.5, label=f'ROC (AUC={auc:.3f})')
    ax2.plot([0,1],[0,1], color='#AAA', lw=1.2, linestyle='--', label='Random')
    ax2.fill_between(fpr, tpr, alpha=0.15, color=BLUE)
    ax2.set_xlabel('False Positive Rate',fontsize=10)
    ax2.set_ylabel('True Positive Rate',fontsize=10)
    ax2.set_title('ROC Curve',fontsize=11,fontweight='bold',color=DARK)
    ax2.legend(fontsize=9); ax2.grid(linestyle='--',alpha=0.4); ax2.set_facecolor(BG)

    # ── bottom-right: Precision-Recall ──
    prec,rec,_ = precision_recall_curve(y_te, prob)
    ap = average_precision_score(y_te, prob)
    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(rec, prec, color=RED, lw=2.5, label=f'P-R (AP={ap:.3f})')
    ax3.fill_between(rec, prec, alpha=0.15, color=RED)
    ax3.set_xlabel('Recall',fontsize=10); ax3.set_ylabel('Precision',fontsize=10)
    ax3.set_title('Precision–Recall Curve',fontsize=11,fontweight='bold',color=DARK)
    ax3.legend(fontsize=9); ax3.grid(linestyle='--',alpha=0.4); ax3.set_facecolor(BG)

    fig.suptitle('Rain Prediction Classifier Dashboard  ·  Ibadan 2021–2025',
                  fontsize=14, fontweight='bold', color=DARK, y=1.01)
    out=outdir/'rain_dashboard.png'
    plt.savefig(out,dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def _plot_multitarget(Y_te, Y_pred, targets, dates, outdir):
    fig, axes = plt.subplots(len(targets), 2, figsize=(14, 4*len(targets)), facecolor=BG)
    fig.patch.set_facecolor(BG)
    colors=[RED, ORANGE, BLUE]
    for i,(t,col) in enumerate(zip(targets,colors)):
        obs=Y_te[:,i]; pred=Y_pred[:,i]
        r2_=r2_score(obs,pred); mae_=mean_absolute_error(obs,pred)
        ax_ts=axes[i,0]
        ax_ts.plot(dates,obs,color=DARK,lw=1.8,label='Observed')
        ax_ts.plot(dates,pred,color=col,lw=1.5,linestyle='--',alpha=0.85,label='Predicted')
        ax_ts.set_title(f'{t.split("(")[0].strip()}  |  R²={r2_:.3f}  MAE={mae_:.3f}',
                         fontsize=10,fontweight='bold',color=DARK,loc='left')
        ax_ts.set_ylabel(t.split('(')[0][:15],fontsize=9)
        ax_ts.legend(fontsize=8); ax_ts.grid(linestyle='--',alpha=0.4); ax_ts.set_facecolor(BG)
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_ts.tick_params(axis='x',rotation=45,labelsize=8)

        ax_sc=axes[i,1]
        ax_sc.scatter(obs,pred,color=col,alpha=0.65,s=25,edgecolors='none')
        lim=[min(obs.min(),pred.min())-1,max(obs.max(),pred.max())+1]
        ax_sc.plot(lim,lim,color='#555',lw=1.5,linestyle='--')
        ax_sc.set_xlim(lim); ax_sc.set_ylim(lim)
        ax_sc.set_xlabel('Observed',fontsize=9); ax_sc.set_ylabel('Predicted',fontsize=9)
        ax_sc.set_title('1:1',fontsize=10,fontweight='bold',color=DARK)
        ax_sc.grid(linestyle='--',alpha=0.4); ax_sc.set_facecolor(BG)

    fig.suptitle('Multi-Target Forecasting: ET₀ + Crop Stress + Fire Risk  ·  Ibadan',
                  fontsize=13,fontweight='bold',color=DARK,y=1.005)
    plt.tight_layout()
    out=outdir/'multitarget.png'
    plt.savefig(out,dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def _plot_alert_dashboard(alerts, df, outdir):
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── alert frequency bar ──
    ax1 = fig.add_subplot(gs[0, 0])
    counts = alerts['Alert'].value_counts()
    bar_colors = [SEVERITY_COLOR.get(
        ALERT_RULES[a]['severity'] if a in ALERT_RULES else 'INFO', BLUE)
        for a in counts.index]
    ax1.barh(range(len(counts)), counts.values[::-1],
              color=bar_colors[::-1], alpha=0.85, edgecolor='none')
    ax1.set_yticks(range(len(counts)))
    ax1.set_yticklabels(counts.index[::-1], fontsize=8.5)
    ax1.set_xlabel('Number of Days Triggered', fontsize=10)
    ax1.set_title('Alert Frequency (2021–2025)',
                   fontsize=11, fontweight='bold', color=DARK)
    ax1.grid(axis='x', linestyle='--', alpha=0.4); ax1.set_facecolor(BG)

    # ── alert timeline (monthly) ──
    ax2 = fig.add_subplot(gs[0, 1])
    alerts['Date'] = pd.to_datetime(alerts['Date'])
    alerts['Month'] = alerts['Date'].dt.to_period('M')
    monthly_counts = alerts.groupby('Month').size().reset_index(name='count')
    monthly_counts['Date'] = monthly_counts['Month'].dt.to_timestamp()
    ax2.bar(monthly_counts['Date'], monthly_counts['count'],
             color=RED, alpha=0.7, width=20, edgecolor='none')
    roll = monthly_counts['count'].rolling(3, center=True, min_periods=1).mean()
    ax2.plot(monthly_counts['Date'], roll, color=DARK, lw=2.5,
              label='3-month average')
    ax2.set_ylabel('Total Alerts/Month', fontsize=10)
    ax2.set_title('Alert Activity Timeline',
                   fontsize=11, fontweight='bold', color=DARK)
    ax2.legend(fontsize=9); ax2.grid(linestyle='--',alpha=0.4); ax2.set_facecolor(BG)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='x', rotation=45, labelsize=8)

    # ── severity breakdown ──
    ax3 = fig.add_subplot(gs[1, 0])
    sev_counts = alerts['Severity'].value_counts()
    sev_colors = [SEVERITY_COLOR.get(s, BLUE) for s in sev_counts.index]
    wedges, texts, pcts = ax3.pie(sev_counts.values, labels=sev_counts.index,
                                    colors=sev_colors, autopct='%1.1f%%',
                                    startangle=90, pctdistance=0.75,
                                    textprops={'fontsize':9})
    ax3.set_title('Alert Severity Distribution',
                   fontsize=11, fontweight='bold', color=DARK)

    # ── indices seasonal pattern ──
    ax4 = fig.add_subplot(gs[1, 1])
    df_idx = build_intelligence_blocks(df, is_daily=True)
    monthly_idx = df_idx.groupby('Month_Num')[['CSI','FRI','IDD']].mean()
    months = ['J','F','M','A','M','J','J','A','S','O','N','D']
    x = np.arange(12)
    ax4.bar(x-0.25, monthly_idx['CSI'].values,  width=0.25, color=RED,    alpha=0.8, label='CSI')
    ax4.bar(x,      monthly_idx['FRI'].values,  width=0.25, color=ORANGE, alpha=0.8, label='FRI')
    ax4.bar(x+0.25, monthly_idx['IDD'].values,  width=0.25, color=TEAL,   alpha=0.8, label='IDD')
    ax4.set_xticks(x); ax4.set_xticklabels(months, fontsize=9)
    ax4.set_ylabel('Index Score (0–100)', fontsize=10)
    ax4.set_title('Seasonal Intelligence Index Pattern',
                   fontsize=11, fontweight='bold', color=DARK)
    ax4.legend(fontsize=9); ax4.grid(axis='y',linestyle='--',alpha=0.4); ax4.set_facecolor(BG)

    fig.suptitle('Level-1 Decision Engine Summary  ·  Ibadan 2021–2025',
                  fontsize=14, fontweight='bold', color=DARK, y=1.01)
    out=outdir/'alert_dashboard.png'
    plt.savefig(out,dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def _export_latex(res_df, path, caption, label):
    cols_order = ['Model','RMSE','MAE','R2','NSE','KGE','MBE']
    df = res_df[cols_order].copy()
    df.columns=['Model','RMSE (mm)','MAE (mm)','R²','NSE','KGE','MBE (mm)']
    best_idx={'RMSE (mm)':df['RMSE (mm)'].idxmin(),'MAE (mm)':df['MAE (mm)'].idxmin(),
              'R²':df['R²'].idxmax(),'NSE':df['NSE'].idxmax(),'KGE':df['KGE'].idxmax()}
    rows=[]
    for idx,row in df.iterrows():
        cells=[row['Model']]
        for c in ['RMSE (mm)','MAE (mm)','R²','NSE','KGE','MBE (mm)']:
            val=f"{row[c]:.4f}"
            if c in best_idx and best_idx[c]==idx:
                val=r"\textbf{"+val+"}"
            cells.append(val)
        rows.append(" & ".join(cells)+r" \\")
    tbl=(r"\begin{table}[htbp]"+"\n"+r"  \centering"+"\n"+
         f"  \\caption{{{caption}}}\n"+f"  \\label{{{label}}}\n"+
         r"  \begin{tabular}{lcccccc}"+"\n"+r"    \toprule"+"\n"+
         r"    Model & RMSE & MAE & R$^2$ & NSE & KGE & MBE \\" + "\n" +
         r"    \midrule"+"\n")
    for r in rows: tbl+="    "+r+"\n"
    tbl+=(r"    \bottomrule"+"\n"+r"  \end{tabular}"+"\n"+r"\end{table}"+"\n")
    Path(path).write_text(tbl)
    print(f"  LaTeX → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    DAILY_PATH   = "data/processed/Cleaned_Daily_Final.csv"
    MONTHLY_PATH = "data/processed/Cleaned_Monthly_Final.csv"
    OUT = Path("results/mcis")
    OUT.mkdir(parents=True, exist_ok=True)

    daily, monthly = load_data(DAILY_PATH, MONTHLY_PATH)

    # ── Intelligence indices dashboard ──────────────────────────────────
    plot_index_dashboard(daily, OUT)

    # ── Level-1 Decision Engine ──────────────────────────────────────────
    alerts = visualise_decision_engine(daily, OUT)

    # ── Experiment A: ET₀ estimation ────────────────────────────────────
    et0_res, et0_mdl, et0_feats = experiment_et0(daily, monthly, OUT)

    # ── Experiment B: Rain prediction ───────────────────────────────────
    rain_res, rain_mdl = experiment_rain_classifier(daily, OUT)

    # ── Experiment C: Crop Stress Index forecast ─────────────────────────
    csi_res, csi_mdl  = experiment_csi_forecast(daily, OUT)

    # ── Experiment D: Multi-target ───────────────────────────────────────
    mt_mdl = experiment_multitarget(daily, OUT)

    print("\n" + "="*60)
    print("ALL MCIS EXPERIMENTS COMPLETE")
    print(f"Outputs → {OUT}")
    print("="*60)