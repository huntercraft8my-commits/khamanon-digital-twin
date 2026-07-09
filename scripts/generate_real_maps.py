# ============================================
# DSS - KHAMANON BLOCK
# Script 6: Spatial Prediction Maps
# Using real rasters + trained RF models
# ============================================

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  PHASE 3 — SPATIAL PREDICTION MAPS")
print("  Khamanon Block — Real Rasters")
print("=" * 55)

# ============================================
# LOAD ONE RASTER AS REFERENCE GRID
# DEM defines our prediction grid extent
# All rasters are same size and CRS
# ============================================

raster_dir = os.path.join('..', 'rasters')

ref_path = os.path.join(raster_dir, 'dem.tif')
with rasterio.open(ref_path) as src:
    ref_meta    = src.meta.copy()
    ref_shape   = (src.height, src.width)
    ref_bounds  = src.bounds
    ref_crs     = src.crs
    dem_data    = src.read(1).astype(float)
    nodata_val  = src.nodata
    transform   = src.transform

print(f"\nReference grid:")
print(f"  Size   : {ref_shape[1]} x {ref_shape[0]} pixels")
print(f"  Bounds : {ref_bounds}")
print(f"  CRS    : {ref_crs}")

# Mask nodata
if nodata_val is not None:
    dem_data[dem_data == nodata_val] = np.nan

# ============================================
# LOAD ALL RASTERS INTO MEMORY
# Each becomes a 2D array same shape as DEM
# ============================================

print("\nLoading rasters...")

def load_raster(path, nodata_override=None):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nd   = src.nodata if nodata_override is None \
               else nodata_override
        if nd is not None:
            data[data == nd] = np.nan
        data[data == -9999]  = np.nan
        data[data == -32768] = np.nan
    return data

raster_arrays = {
    'dem'          : dem_data,
    'slope'        : load_raster(
                       os.path.join(raster_dir,'slope.tif'),
                       nodata_override=127),
    'aspect'       : load_raster(
                       os.path.join(raster_dir,'aspect.tif')),
    'lulc'         : load_raster(
                       os.path.join(raster_dir,'lulc.tif'),
                       nodata_override=2147483647),
    'lithology'    : load_raster(
                       os.path.join(raster_dir,'lithology.tif'),
                       nodata_override=0),
    'geomorphology': load_raster(
                       os.path.join(raster_dir,
                                    'geomorphology.tif'),
                       nodata_override=0),
}

for name, arr in raster_arrays.items():
    valid = np.sum(~np.isnan(arr))
    print(f"  {name:<15}: {arr.shape} | "
          f"valid pixels={valid:,}")

# ============================================
# ADD SENTINEL-2 SPECTRAL INDICES
# We load real spectral data from our CSV
# and interpolate to full raster grid
# For real data: use GEE exported rasters
# For practice: simulate spatial patterns
# ============================================

print("\nGenerating spectral index grids...")

# Create coordinate grids matching raster
rows = np.arange(ref_shape[0])
cols = np.arange(ref_shape[1])
col_grid, row_grid = np.meshgrid(cols, rows)

# Convert pixel indices to UTM coordinates
easting  = (ref_bounds.left +
            col_grid * (ref_bounds.right -
                        ref_bounds.left) / ref_shape[1])
northing = (ref_bounds.top  -
            row_grid * (ref_bounds.top   -
                        ref_bounds.bottom) / ref_shape[0])

# Normalise 0-1 for spatial gradients
e_norm = (easting  - easting.min()) / (easting.max()  - easting.min())
n_norm = (northing - northing.min()) / (northing.max() - northing.min())

# Create mask from DEM (valid land area only)
land_mask = ~np.isnan(dem_data)

np.random.seed(42)

# NDVI — higher in north (more cropland)
NDVI_grid = (0.30 + 0.25 * n_norm
             - 0.08 * e_norm
             + np.random.normal(0, 0.04, ref_shape))
NDVI_grid = np.clip(NDVI_grid, 0.10, 0.80)

# NDBI — higher in southeast (built-up)
NDBI_grid = (-0.20 + 0.15 * e_norm
             - 0.10 * n_norm
             + np.random.normal(0, 0.03, ref_shape))
NDBI_grid = np.clip(NDBI_grid, -0.45, 0.25)

# SAVI — derived from NDVI
SAVI_grid = NDVI_grid * 0.88 + np.random.normal(
    0, 0.02, ref_shape)
SAVI_grid = np.clip(SAVI_grid, 0.05, 0.55)

# BSI — higher where bare soil / fallow
BSI_grid  = (-0.05 - 0.10 * n_norm
             + np.random.normal(0, 0.04, ref_shape))
BSI_grid  = np.clip(BSI_grid, -0.35, 0.25)

# Apply land mask
for arr in [NDVI_grid, NDBI_grid,
            SAVI_grid, BSI_grid]:
    arr[~land_mask] = np.nan

raster_arrays['NDVI'] = NDVI_grid
raster_arrays['NDBI'] = NDBI_grid
raster_arrays['SAVI'] = SAVI_grid
raster_arrays['BSI']  = BSI_grid

print("  Spectral grids created.")

# ============================================
# BUILD FLAT FEATURE ARRAY
# Shape: (n_valid_pixels, n_features)
# Only predict on valid land pixels
# ============================================

feature_cols = [
    'dem', 'slope', 'aspect',
    'lulc', 'lithology', 'geomorphology',
    'NDVI', 'NDBI', 'SAVI', 'BSI'
]

# Create validity mask — pixel valid in ALL rasters
valid_mask = land_mask.copy()
for name in feature_cols:
    valid_mask &= ~np.isnan(raster_arrays[name])

n_valid = valid_mask.sum()
print(f"\nValid pixels for prediction: {n_valid:,}")

# Build feature matrix
X_pred = np.column_stack([
    raster_arrays[f][valid_mask]
    for f in feature_cols
])

print(f"Feature matrix shape: {X_pred.shape}")

# ============================================
# PREDICT SOIL PROPERTIES ACROSS FULL GRID
# ============================================

target_cols = [
    'pH', 'OC', 'EC', 'K2O',
    'available_P', 'available_N',
    'CEC', 'bulk_density', 'CaCO3'
]

predictions = {}

print("\nGenerating predictions...")
print("-" * 40)

for target in target_cols:
    model_path = os.path.join(
        '..', 'models', f'rf_real_{target}.pkl'
    )
    model = joblib.load(model_path)
    pred  = model.predict(X_pred)

    # Reconstruct full raster grid
    pred_grid = np.full(ref_shape, np.nan)
    pred_grid[valid_mask] = pred
    predictions[target]   = pred_grid

    print(f"  {target:<15}: "
          f"min={np.nanmin(pred):.3f} | "
          f"max={np.nanmax(pred):.3f} | "
          f"mean={np.nanmean(pred):.3f}")

# ============================================
# SAVE PREDICTION GRIDS AS CSV
# (for dashboard)
# ============================================

# Sample every 10th pixel to keep CSV small
step = 10
rows_s = np.arange(0, ref_shape[0], step)
cols_s = np.arange(0, ref_shape[1], step)

grid_records = []
for r in rows_s:
    for c in cols_s:
        if valid_mask[r, c]:
            record = {
                'easting' : float(easting[r, c]),
                'northing': float(northing[r, c])
            }
            for target in target_cols:
                record[target] = float(
                    predictions[target][r, c]
                )
            grid_records.append(record)

grid_df = pd.DataFrame(grid_records)
grid_df.to_csv(
    os.path.join('..', 'data',
                 'real_prediction_grid.csv'),
    index=False
)
print(f"\nPrediction grid saved: {len(grid_df):,} points")

# ============================================
# PLOT AND SAVE MAPS
# ============================================

colormaps = {
    'pH'          : ('RdYlGn_r', 'pH units'),
    'OC'          : ('YlGn',     '%'),
    'EC'          : ('OrRd',     'dS/m'),
    'K2O'         : ('BuGn',     'kg/ha'),
    'available_P' : ('Purples',  'kg/ha'),
    'available_N' : ('Blues',    'kg/ha'),
    'CEC'         : ('PuBu',     'meq/100g'),
    'bulk_density': ('YlOrBr',   'g/L'),
    'CaCO3'       : ('RdPu',     '%')
}

extent = [
    ref_bounds.left,  ref_bounds.right,
    ref_bounds.bottom, ref_bounds.top
]

print("\nSaving soil maps...")

for target in target_cols:
    cmap_name, unit = colormaps[target]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        predictions[target],
        extent = extent,
        origin = 'upper',
        cmap   = cmap_name,
        aspect = 'auto'
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label(
        f"{target.replace('_',' ').title()} ({unit})",
        fontsize=10
    )

    ax.set_xlabel('Easting (m, UTM 43N)', fontsize=10)
    ax.set_ylabel('Northing (m, UTM 43N)', fontsize=10)
    ax.set_title(
        f"Predicted {target.replace('_',' ').title()}\n"
        f"Khamanon Block, Fatehgarh Sahib",
        fontsize=12, fontweight='bold'
    )
    ax.grid(True, linestyle='--',
            alpha=0.3, color='white')

    plt.tight_layout()
    save_path = os.path.join(
        '..', 'maps', f'realmap_{target}.png'
    )
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: maps/realmap_{target}.png")

print("\n" + "=" * 55)
print("  PHASE 3 COMPLETE")
print("=" * 55)
print("\nWhat was built:")
print(f"  9 soil prediction maps → maps/")
print(f"  Prediction grid CSV   → data/")
print("\nNext: Phase 4 — Crop Monitoring")
print("Then: Phase 5 — Dashboard with real maps")