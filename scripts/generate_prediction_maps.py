# ============================================
# DSS - KHAMANON BLOCK
# Script 3: Spatial Prediction Maps
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
import os

# ============================================
# CREATE PREDICTION GRID
# Fine grid covering all of Khamanon Block
# ============================================

print("=" * 50)
print("  SPATIAL PREDICTION MAPS - KHAMANON BLOCK")
print("=" * 50)

# Khamanon Block boundaries
lat_min, lat_max = 30.38, 30.52
lon_min, lon_max = 76.27, 76.42

# Create 50x50 grid = 2500 prediction points
n_grid = 50
lats   = np.linspace(lat_min, lat_max, n_grid)
lons   = np.linspace(lon_min, lon_max, n_grid)

# Create meshgrid
lon_grid, lat_grid = np.meshgrid(lons, lats)
lat_flat = lat_grid.flatten()
lon_flat = lon_grid.flatten()

n_points = len(lat_flat)
print(f"\nPrediction grid: {n_grid}x{n_grid} = {n_points} points")
print("Covering full Khamanon Block extent")

# ============================================
# SIMULATE REALISTIC SPATIAL PATTERNS
# This gives maps that look meaningful
# Real data: these come from GEE
# ============================================

np.random.seed(42)

# Simulate spatial gradient in NDVI
# (northern part more vegetated, south more built-up)
# This mimics real landscape patterns
lat_norm = (lat_flat - lat_min) / (lat_max - lat_min)
lon_norm = (lon_flat - lon_min) / (lon_max - lon_min)

NDVI = (0.35 + 0.30 * lat_norm
        - 0.10 * lon_norm
        + np.random.normal(0, 0.05, n_points))
NDVI = np.clip(NDVI, 0.05, 0.85)

NDBI = (0.25 - 0.20 * lat_norm
        + 0.15 * lon_norm
        + np.random.normal(0, 0.04, n_points))
NDBI = np.clip(NDBI, -0.30, 0.55)

SAVI = NDVI * 0.88 + np.random.normal(0, 0.02, n_points)
SAVI = np.clip(SAVI, 0.05, 0.80)

BSI  = (0.20 - 0.15 * lat_norm
        + np.random.normal(0, 0.04, n_points))
BSI  = np.clip(BSI, -0.10, 0.50)

# Terrain (flat alluvial plain with slight gradient)
elevation = (250 + 10 * lat_norm
             - 5  * lon_norm
             + np.random.normal(0, 2, n_points))
slope     = np.abs(np.random.normal(1.5, 0.8, n_points))
slope     = np.clip(slope, 0.1, 3.5)
aspect    = np.random.uniform(0, 360, n_points)

# LULC encoded (0=Built-up, 1=Cropland,
#               2=Fallow, 3=Vegetation, 4=Water body)
# Urban zone in southeast, cropland dominant
lulc_encoded = np.where(
    (lon_norm > 0.7) & (lat_norm < 0.4), 0,  # built-up southeast
    np.where(NDVI > 0.55, 3,                  # high NDVI = vegetation
    np.where(NDVI < 0.20, 2,                  # low NDVI  = fallow
    1))                                        # rest = cropland
)

# ============================================
# ASSEMBLE FEATURE GRID FOR PREDICTION
# ============================================

grid_df = pd.DataFrame({
    'NDVI'        : np.round(NDVI, 4),
    'NDBI'        : np.round(NDBI, 4),
    'SAVI'        : np.round(SAVI, 4),
    'BSI'         : np.round(BSI,  4),
    'elevation'   : np.round(elevation, 1),
    'slope'       : np.round(slope, 2),
    'aspect'      : np.round(aspect, 1),
    'LULC_encoded': lulc_encoded
})

# ============================================
# LOAD TRAINED MODELS AND PREDICT
# ============================================

targets = [
    'pH', 'organic_carbon', 'EC',
    'available_N', 'available_P',
    'available_K', 'CEC'
]

predictions = {}

print("\nGenerating predictions...")
for target in targets:
    model = joblib.load(
        os.path.join('..', 'models', f'rf_{target}.pkl')
    )
    pred = model.predict(grid_df)
    predictions[target] = pred.reshape(n_grid, n_grid)
    print(f"  {target:<20} done")

# ============================================
# PLOT AND SAVE MAPS
# One map per soil property
# ============================================

# Color schemes matching soil science convention
colormaps = {
    'pH'              : 'RdYlGn_r',  # red=high pH, green=low
    'organic_carbon'  : 'YlGn',      # yellow=low OC, green=high
    'EC'              : 'OrRd',      # orange-red = high salinity
    'available_N'     : 'Blues',
    'available_P'     : 'Purples',
    'available_K'     : 'BuGn',
    'CEC'             : 'PuBu'
}

units = {
    'pH'              : 'pH units',
    'organic_carbon'  : '%',
    'EC'              : 'dS/m',
    'available_N'     : 'kg/ha',
    'available_P'     : 'kg/ha',
    'available_K'     : 'kg/ha',
    'CEC'             : 'meq/100g'
}

print("\nSaving maps...")

for target in targets:

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        predictions[target],
        extent   = [lon_min, lon_max, lat_min, lat_max],
        origin   = 'lower',
        cmap     = colormaps[target],
        aspect   = 'auto'
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{target} ({units[target]})", fontsize=11)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude',  fontsize=11)
    ax.set_title(
        f'Predicted {target.replace("_"," ").title()}\n'
        f'Khamanon Block, Fatehgarh Sahib (Fake Data)',
        fontsize=12, fontweight='bold'
    )

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.4, color='white')

    plt.tight_layout()
    save_path = os.path.join('..', 'maps', f'map_{target}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: maps/map_{target}.png")

# ============================================
# SAVE FULL PREDICTION GRID AS CSV
# (used by dashboard later)
# ============================================

grid_df['latitude']  = lat_flat
grid_df['longitude'] = lon_flat

for target in targets:
    grid_df[target] = predictions[target].flatten()

grid_df.to_csv(
    os.path.join('..', 'data', 'prediction_grid.csv'),
    index=False
)

print(f"\nFull prediction grid saved: data/prediction_grid.csv")
print(f"Grid size: {n_points} points")

print("\n" + "=" * 50)
print("  STAGE 3 COMPLETE")
print("=" * 50)
print("\nWhat was built:")
print("  - 7 soil property maps saved to maps/")
print("  - Full prediction grid saved to data/")
print("  - Ready for dashboard in Stage 5")
print("\nGo open maps/ folder and look at your maps.")