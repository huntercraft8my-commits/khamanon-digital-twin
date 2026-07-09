# ============================================
# DSS - KHAMANON BLOCK
# Script 4: Build Master Training Dataset
# Merge soil data + rasters + spectral indices
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 55)
print("  PHASE 1 — STEP 1.5: BUILD MASTER DATASET")
print("=" * 55)

# ============================================
# LOAD ALL THREE DATA SOURCES
# ============================================

# 1. Soil data with raster covariates
soil = pd.read_csv(
    os.path.join('..', 'data', 'soil_with_covariates.csv')
)
print(f"\nSoil + raster data : {soil.shape}")

# 2. Spectral indices from GEE
spectral = pd.read_csv(
    os.path.join('..', 'data', 'spectral_indices.csv')
)
print(f"Spectral indices   : {spectral.shape}")

# ============================================
# CLEAN SPECTRAL DATA
# Keep only first 208 rows (matching samples)
# Select only the columns we need
# ============================================

spectral_clean = spectral[['point_id', 'NDVI', 'NDBI',
                             'SAVI', 'BSI']].copy()

# Keep only rows matching our 208 sample points
spectral_clean = spectral_clean[
    spectral_clean['point_id'] < len(soil)
].reset_index(drop=True)

print(f"Spectral after trim: {spectral_clean.shape}")

# ============================================
# CHECK FOR MISSING SPECTRAL VALUES
# Can happen if cloud cover blocked some points
# ============================================

print("\nSpectral indices summary:")
print(spectral_clean[['NDVI','NDBI','SAVI','BSI']].describe().round(4).to_string())

missing_spectral = spectral_clean[['NDVI','NDBI','SAVI','BSI']].isnull().sum()
print(f"\nMissing spectral values: {missing_spectral.sum()}")

if missing_spectral.sum() > 0:
    for col in ['NDVI','NDBI','SAVI','BSI']:
        n = spectral_clean[col].isnull().sum()
        if n > 0:
            med = spectral_clean[col].median()
            spectral_clean[col] = spectral_clean[col].fillna(med)
            print(f"  Filled {n} missing {col} with median={med:.4f}")

# ============================================
# MERGE EVERYTHING TOGETHER
# soil already has raster values from Step 1.3
# Just add spectral indices by position
# Both are ordered 0 to 207
# ============================================

master = soil.copy()
master['NDVI'] = spectral_clean['NDVI'].values
master['NDBI'] = spectral_clean['NDBI'].values
master['SAVI'] = spectral_clean['SAVI'].values
master['BSI']  = spectral_clean['BSI'].values

print(f"\nMaster dataset shape: {master.shape}")
print(f"Total columns       : {len(master.columns)}")

# ============================================
# FINAL COLUMN STRUCTURE
# ============================================

print("\nAll columns in master dataset:")
print("-" * 40)

id_cols       = ['sample_id', 'easting_utm', 'northing_utm',
                 'longitude', 'latitude']
target_cols   = ['pH', 'OC', 'EC', 'K2O', 'available_P',
                 'available_N', 'CEC', 'bulk_density', 'CaCO3']
feature_cols  = ['dem', 'slope', 'aspect', 'lulc',
                 'lithology', 'geomorphology',
                 'NDVI', 'NDBI', 'SAVI', 'BSI']

print("ID columns (not used in model):")
for c in id_cols:
    print(f"  {c}")

print("\nTarget columns (what RF predicts):")
for c in target_cols:
    print(f"  {c}")

print("\nFeature columns (RF inputs):")
for c in feature_cols:
    print(f"  {c}")

# ============================================
# CORRELATION: FEATURES vs TARGETS
# Which spectral index predicts which
# soil property best?
# ============================================

print("\nFeature-Target Correlations (top relationships):")
print("-" * 55)

correlations = []
for feat in feature_cols:
    for targ in target_cols:
        r = master[feat].corr(master[targ])
        correlations.append({
            'feature': feat,
            'target' : targ,
            'r'      : round(r, 3),
            'abs_r'  : abs(r)
        })

corr_df = pd.DataFrame(correlations).sort_values(
    'abs_r', ascending=False
)

print(corr_df[['feature','target','r']].head(20).to_string(index=False))

# ============================================
# PLOT: NDVI vs OC (most important relationship)
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

pairs = [
    ('NDVI', 'OC',          '#2ecc71'),
    ('NDBI', 'pH',          '#e74c3c'),
    ('SAVI', 'available_N', '#3498db'),
    ('BSI',  'bulk_density','#e67e22')
]

for ax, (feat, targ, color) in zip(axes.flatten(), pairs):
    ax.scatter(master[feat], master[targ],
               alpha=0.5, color=color,
               edgecolors='gray', linewidth=0.3, s=40)
    r = master[feat].corr(master[targ])
    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel(targ.replace('_',' ').title(), fontsize=11)
    ax.set_title(f'{feat} vs {targ.replace("_"," ").title()}\nr = {r:.3f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(master[feat], master[targ], 1)
    p = np.poly1d(z)
    x_line = np.linspace(master[feat].min(),
                         master[feat].max(), 100)
    ax.plot(x_line, p(x_line), 'k--',
            linewidth=1.5, alpha=0.7)

plt.suptitle(
    'Spectral Index vs Soil Property Relationships\n'
    'Khamanon Block — Rabi Season 2025-2026',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps', 'feature_target_scatter.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("\nSaved: maps/feature_target_scatter.png")

# ============================================
# SAVE MASTER DATASET
# ============================================

out_path = os.path.join('..', 'data', 'master_training_data.csv')
master.to_csv(out_path, index=False)

print("\n" + "=" * 55)
print("  PHASE 1 COMPLETE — ALL STEPS DONE")
print("=" * 55)
print(f"\nMaster dataset saved: data/master_training_data.csv")
print(f"Samples  : {len(master)}")
print(f"Features : {len(feature_cols)}")
print(f"Targets  : {len(target_cols)}")
print("\nReady for Phase 2 — RF Model Training")
