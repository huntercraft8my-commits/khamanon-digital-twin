# -*- coding: utf-8 -*-
"""
Khamanon Block Digital Twin - Soil-Satellite Correlation Engine
Resolves Crop-Type Mixing and Boundary Noise via cKDTree Spatial Alignment
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for headless server environments
matplotlib.use('Agg')

# Define Analysis Tokens
SOIL_COLS = ['pH', 'OC', 'EC', 'K2O', 'available_P', 'available_N', 'CEC', 'bulk_density', 'CaCO3']
SPECTRAL_COLS = ['NDVI', 'NDBI', 'SAVI', 'BSI']
CROP_CLASSES = ['Wheat', 'Spring_Maize', 'Agroforestry']
SIG_R = 0.25
SIG_P = 0.05

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, '..', 'data')
maps_dir = os.path.join(base, '..', 'maps')

os.makedirs(maps_dir, exist_ok=True)

print("=" * 60)
print("🚀 RUNNING OFFLINE SOIL-SATELLITE CORRELATION ENGINE")
print("=" * 60)

# -------------------------------------------------------------------------
# STEP 1: Load and Clean Master Dataset
# -------------------------------------------------------------------------
master_path = os.path.join(data_dir, 'master_training_data.csv')
if not os.path.exists(master_path):
    raise FileNotFoundError(f"Missing crucial master dataset at: {master_path}")

df = pd.read_csv(master_path)
print(f"[Step 1] Initial master dataset shape: {df.shape}")

# Purge any record missing matching lab metrics or satellite captures
target_cols = SOIL_COLS + SPECTRAL_COLS
df = df.dropna(subset=target_cols).reset_index(drop=True)
print(f"[Step 1] Records remaining after purging NaNs: {len(df)}")

# -------------------------------------------------------------------------
# STEP 2: LULC Spatial Join & Boundary Filtering (Loophole 1 & 2 Fixes)
# -------------------------------------------------------------------------
lulc_path = os.path.join(data_dir, 'lulc_rabi_map.csv')
if not os.path.exists(lulc_path):
    raise FileNotFoundError(f"Missing spatial validation asset at: {lulc_path}")

lulc_df = pd.read_csv(lulc_path)

# Mandatory Check 4: Align coordinate vectors exactly in [lat, lon] alignment
lulc_coords = lulc_df[['lat', 'lon']].values
tree = cKDTree(lulc_coords)

sample_coords = df[['latitude', 'longitude']].values
distances, indices = tree.query(sample_coords)

# Inject nearest neighbor attributes back to the primary frame
df['assigned_lulc'] = lulc_df['class_name'].iloc[indices].values
print("\n[Step 2] Nearest LULC neighbor frequency breakdown:")
print(df['assigned_lulc'].value_counts())

# Filter contaminated mixed-pixels landing outside active crop classifications
initial_count = len(df)
df_filtered = df[df['assigned_lulc'].isin(CROP_CLASSES)].copy()
dropped_count = initial_count - len(df_filtered)
print(f"[Step 2] Mixed-pixel buffer dropped: {dropped_count} points. Retained: {len(df_filtered)} points.")

filtered_out_path = os.path.join(data_dir, 'soil_spectral_filtered.csv')
df_filtered.to_csv(filtered_out_path, index=False)
print(f"[Step 2] Filtered core saved securely -> {filtered_out_path}")

# Isolate Wheat monoculture points to avoid phenological blinding
wheat_df = df_filtered[df_filtered['assigned_lulc'] == 'Wheat'].copy().reset_index(drop=True)
print(f"[Step 2] Monoculture Wheat points isolated for analytical processing: {len(wheat_df)}")

if len(wheat_df) == 0:
    print("⚠️ CRITICAL WARNING: No rows classified as Wheat. Using filtered block as structural fallback.")
    wheat_df = df_filtered

# -------------------------------------------------------------------------
# STEP 3: Spearman Correlation Matrix Processing
# -------------------------------------------------------------------------
corr_r = pd.DataFrame(index=SOIL_COLS, columns=SPECTRAL_COLS, dtype=float)
corr_p = pd.DataFrame(index=SOIL_COLS, columns=SPECTRAL_COLS, dtype=float)

for s_col in SOIL_COLS:
    for sp_col in SPECTRAL_COLS:
        r, p = spearmanr(wheat_df[s_col], wheat_df[sp_col])
        corr_r.loc[s_col, sp_col] = r
        corr_p.loc[s_col, sp_col] = p

corr_r.to_csv(os.path.join(data_dir, 'soil_spectral_corr_r.csv'))
corr_p.to_csv(os.path.join(data_dir, 'soil_spectral_corr_p.csv'))

print("\n[Step 3] Computed Spearman Correlation Matrix (R Coefficients):")
print(corr_r.round(3))

sig_pairs = []
for s_col in SOIL_COLS:
    for sp_col in SPECTRAL_COLS:
        r_val = corr_r.loc[s_col, sp_col]
        p_val = corr_p.loc[s_col, sp_col]
        if abs(r_val) >= SIG_R and p_val < SIG_P:
            sig_pairs.append({
                'soil_col': s_col,
                'spectral_col': sp_col,
                'spearman_r': r_val,
                'p_value': p_val
            })

sig_df = pd.DataFrame(sig_pairs)
sig_out_path = os.path.join(data_dir, 'soil_spectral_significant_pairs.csv')
sig_df.to_csv(sig_out_path, index=False)
print(f"[Step 3] Identified {len(sig_df)} statistically verified significant interaction chains.")

# -------------------------------------------------------------------------
# STEP 4: OLS Multiple Regression & Structural Residual Engine
# -------------------------------------------------------------------------
ndvi_predictors = [pair['soil_col'] for pair in sig_pairs if pair['spectral_col'] == 'NDVI']
ndvi_predictors = list(set(ndvi_predictors))

if len(ndvi_predictors) < 2:
    print("[Step 4] Significant soil features < 2. Employing academic baseline fallback variables.")
    predictor_cols = ['OC', 'pH', 'available_N']
else:
    predictor_cols = ndvi_predictors

print(f"[Step 4] Formulating multi-regression matrix using parameters: {predictor_cols}")

X = wheat_df[predictor_cols].values
y = wheat_df['NDVI'].values

reg_model = LinearRegression()
reg_model.fit(X, y)

fitted_ndvi = reg_model.predict(X)
residuals = y - fitted_ndvi
r2 = r2_score(y, fitted_ndvi)
print(f"[Step 4] Spatial regression performance metrics (R²): {r2:.3f}")

res_df = wheat_df.copy()
res_df['fitted_NDVI'] = fitted_ndvi
res_df['residual'] = residuals

# Backwards compatibility alignment for customdata unpacking signatures in Dash UI
res_df['actual_NDVI'] = wheat_df['NDVI']

residuals_out_path = os.path.join(data_dir, 'soil_spectral_residuals.csv')
res_df.to_csv(residuals_out_path, index=False)
print(f"[Step 4] Core spatial variance tracking matrix created -> {residuals_out_path}")

# -------------------------------------------------------------------------
# STEP 5: Static Visual Calibration Heatmap (Theme Synchronized)
# -------------------------------------------------------------------------
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#1e2535')
ax.set_facecolor('#1e2535')

# Draw master correlation matrix
sns.heatmap(corr_r, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={'label': 'Spearman Coefficient (R)'}, ax=ax, annot_kws={'size': 11, 'weight': 'bold'})

# Mute insignificant nodes with an alpha layer box overlay
for y_idx, s_col in enumerate(SOIL_COLS):
    for x_idx, sp_col in enumerate(SPECTRAL_COLS):
        if corr_p.loc[s_col, sp_col] >= SIG_P:
            rect = plt.Rectangle((x_idx, y_idx), 1, 1, fill=True, color='#2d3748', alpha=0.45, transform=ax.transData)
            ax.add_patch(rect)

plt.title("Spearman Correlation — Soil Chemistry vs Sentinel-2\n", color='#f1f5f9', fontsize=13, weight='bold')
ax.text(0.5, 1.02, f"Wheat-stratified · {len(wheat_df)} validation footprints · Khamanon Block Rabi 2025-26",
        transform=ax.transAxes, color='#94a3b8', fontsize=9, ha='center')

plt.xticks(color='#94a3b8', fontsize=10)
plt.yticks(color='#94a3b8', fontsize=10)
plt.tight_layout()

heatmap_png_path = os.path.join(maps_dir, 'soil_spectral_heatmap.png')
plt.savefig(heatmap_png_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"[Step 5] Matrix heatmap snapshot generated -> {heatmap_png_path}")

# -------------------------------------------------------------------------
# STEP 6: Core Validation Scatter Diagnostics
# -------------------------------------------------------------------------
count_scatters = 0
for pair in sig_pairs:
    s_c, sp_c = pair['soil_col'], pair['spectral_col']
    r_val, p_val = pair['spearman_r'], pair['p_value']
    
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#1e2535')
    ax.set_facecolor('#1e2535')
    
    x_vals = wheat_df[s_c].values
    y_vals = wheat_df[sp_c].values
    
    # Draw points
    ax.scatter(x_vals, y_vals, color='#00d4aa', alpha=0.7, edgecolors='#2d3748', linewidths=0.5, s=35, label='Samples')
    
    # Calculate fit line
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_range, m * x_range + b, color='#f59e0b', linestyle='--', linewidth=1.5, label='Trendline')
    
    title_str = f"{s_c.replace('_',' ').title()} vs {sp_c}"
    plt.title(title_str, color='#f1f5f9', fontsize=11, weight='bold', pad=12)
    
    # Clean text metrics card inside visualization footprint
    stat_box = f"Spearman R = {r_val:.3f}\np = {p_val:.4f}\nn = {len(wheat_df)}"
    ax.text(0.04, 0.93, stat_box, transform=ax.transAxes, color='#00d4aa', fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b26', edgecolor='#2d3748', alpha=0.9))
    
    ax.set_xlabel(s_c.replace('_',' ').title(), color='#94a3b8', fontsize=9)
    ax.set_ylabel(sp_c, color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.15, color='#ffffff')
    
    plt.tight_layout()
    scatter_path = os.path.join(maps_dir, f"scatter_{s_c}_{sp_c}.png")
    plt.savefig(scatter_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    count_scatters += 1

print(f"[Step 6] Dispatched {count_scatters} linear correlation reports to maps/ directory.")

# -------------------------------------------------------------------------
# STEP 7: Console Summary Document Dispatcher
# -------------------------------------------------------------------------
print("\n" + "="*60)
print("🏁 COMPILATION COMPLETED SUCCESSFULLY")
print("="*60)
print(f"  • Total points raw ingestion:          {initial_count}")
print(f"  • Post-purging data missing flags:     {len(df)}")
print(f"  • Monoculture Wheat testing arrays:    {len(wheat_df)}")
print(f"  • Statistically valid pairs discovered: {len(sig_pairs)}")
for p in sig_pairs:
    print(f"    -> [{p['soil_col']}] paired to [{p['spectral_col']}] -> R: {p['spearman_r']:.3f} (p={p['p_value']:.4e})")
print(f"  • Model variance definition fit (R²):  {r2:.3f}")
print("  • Core Predictors utilized:            " + ", ".join(predictor_cols))
print("="*60)
print("⚠️  DATA EXAMINER FOOTNOTE NOTICE:")
print("  Current iterations rely on synthetic practice matrices. Outputs are")
print("  structural placeholding records. Math equations automatically validate")
print("  upon replacement of real PAU physical laboratory spreadsheets.")
print("="*60 + "\n")