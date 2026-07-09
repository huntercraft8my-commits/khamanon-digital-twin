# ============================================
# DSS - KHAMANON BLOCK
# Script 4: Crop Growth Monitoring
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

print("=" * 50)
print("  CROP MONITORING - KHAMANON BLOCK")
print("=" * 50)

# ============================================
# KHAMANON BLOCK CROP CALENDAR
# Based on Punjab rice-wheat system
# Exactly as described in your synopsis
# ============================================

# 12 months of the year
months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

# ============================================
# SIMULATE NDVI TIME SERIES
# For 4 zones in Khamanon Block
# Each zone has different crop health
# ============================================

# Zone 1: Healthy Cropland (north Khamanon)
# Strong rice-wheat rotation, good irrigation
ndvi_healthy = np.array([
    0.52,  # Jan  - wheat growing well
    0.65,  # Feb  - wheat peak
    0.72,  # Mar  - wheat peak
    0.58,  # Apr  - wheat maturity
    0.25,  # May  - post harvest, bare soil
    0.18,  # Jun  - tillage, bare
    0.35,  # Jul  - rice transplanting
    0.58,  # Aug  - rice growing
    0.72,  # Sep  - rice peak
    0.65,  # Oct  - rice maturity
    0.42,  # Nov  - rice harvest, wheat sowing
    0.48   # Dec  - wheat establishing
])

# Zone 2: Stressed Cropland (central Khamanon)
# Degraded soil, low OC, high pH
ndvi_stressed = np.array([
    0.38,  # Jan
    0.48,  # Feb
    0.52,  # Mar
    0.40,  # Apr
    0.18,  # May
    0.12,  # Jun
    0.22,  # Jul
    0.40,  # Aug
    0.52,  # Sep
    0.44,  # Oct
    0.28,  # Nov
    0.33   # Dec
])

# Zone 3: Peri-urban / Built-up (southeast)
# Low NDVI year round, urban expansion
ndvi_urban = np.array([
    0.18,  # Jan
    0.20,  # Feb
    0.22,  # Mar
    0.18,  # Apr
    0.12,  # May
    0.10,  # Jun
    0.14,  # Jul
    0.16,  # Aug
    0.18,  # Sep
    0.15,  # Oct
    0.14,  # Nov
    0.16   # Dec
])

# Zone 4: Vegetation / Scrubland (patches)
# Stable moderate NDVI year round
ndvi_vegetation = np.array([
    0.45,  # Jan
    0.48,  # Feb
    0.52,  # Mar
    0.55,  # Apr
    0.58,  # May
    0.60,  # Jun
    0.65,  # Jul
    0.68,  # Aug
    0.65,  # Sep
    0.60,  # Oct
    0.52,  # Nov
    0.47   # Dec
])

# ============================================
# PLOT 1: NDVI Time Series - All 4 Zones
# ============================================

fig, ax = plt.subplots(figsize=(12, 6))

x = range(12)

ax.plot(x, ndvi_healthy,
        'g-o', linewidth=2.5, markersize=7,
        label='Healthy Cropland (North)')
ax.plot(x, ndvi_stressed,
        'orange', linestyle='-', marker='s',
        linewidth=2.5, markersize=7,
        label='Stressed Cropland (Central)')
ax.plot(x, ndvi_urban,
        'r-^', linewidth=2.5, markersize=7,
        label='Peri-urban / Built-up (SE)')
ax.plot(x, ndvi_vegetation,
        'b-D', linewidth=2.5, markersize=7,
        label='Vegetation / Scrubland')

# Add crop season bands
ax.axvspan(-0.5, 3.5,  alpha=0.08, color='gold',
           label='Wheat Season')
ax.axvspan(3.5,  5.5,  alpha=0.08, color='gray',
           label='Fallow / Tillage')
ax.axvspan(5.5,  10.5, alpha=0.08, color='lightgreen',
           label='Rice Season')
ax.axvspan(10.5, 11.5, alpha=0.08, color='gold')

# Threshold line for crop stress
ax.axhline(y=0.40, color='red', linestyle='--',
           linewidth=1.5, alpha=0.7,
           label='Stress Threshold (NDVI=0.40)')

ax.set_xticks(x)
ax.set_xticklabels(months, fontsize=11)
ax.set_ylabel('NDVI', fontsize=12)
ax.set_xlabel('Month', fontsize=12)
ax.set_title(
    'Crop Growth Monitoring - Khamanon Block\n'
    'NDVI Time Series by Zone (2025-2026)',
    fontsize=13, fontweight='bold'
)
ax.set_ylim(0, 0.85)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps', 'crop_ndvi_timeseries.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("\nSaved: maps/crop_ndvi_timeseries.png")

# ============================================
# PLOT 2: Seasonal NDVI Maps
# Simulate spatial NDVI for 4 key months
# ============================================

lat_min, lat_max = 30.38, 30.52
lon_min, lon_max = 76.27, 76.42
n_grid = 50

lats = np.linspace(lat_min, lat_max, n_grid)
lons = np.linspace(lon_min, lon_max, n_grid)
lon_grid, lat_grid = np.meshgrid(lons, lats)

lat_norm = (lat_grid - lat_min) / (lat_max - lat_min)
lon_norm = (lon_grid - lon_min) / (lon_max - lon_min)

# 4 key seasons to map
seasons = {
    'March (Wheat Peak)'    : 0.72,
    'June (Fallow/Bare)'    : 0.18,
    'September (Rice Peak)' : 0.72,
    'November (Harvest)'    : 0.42
}

season_modifiers = {
    'March (Wheat Peak)'    :  0.15,
    'June (Fallow/Bare)'    : -0.05,
    'September (Rice Peak)' :  0.12,
    'November (Harvest)'    :  0.08
}

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for idx, (season_name, base_ndvi) in enumerate(seasons.items()):

    np.random.seed(idx * 10)

    # Spatial NDVI with realistic gradients
    spatial_ndvi = (
        base_ndvi
        + season_modifiers[season_name] * lat_norm
        - 0.08 * lon_norm
        + np.random.normal(0, 0.04, (n_grid, n_grid))
    )

    # Urban zone = always low NDVI
    urban_mask = (lon_norm > 0.70) & (lat_norm < 0.35)
    spatial_ndvi[urban_mask] = np.random.uniform(
        0.08, 0.20, spatial_ndvi[urban_mask].shape
    )

    spatial_ndvi = np.clip(spatial_ndvi, 0.05, 0.85)

    im = axes[idx].imshow(
        spatial_ndvi,
        extent  = [lon_min, lon_max, lat_min, lat_max],
        origin  = 'lower',
        cmap    = 'RdYlGn',
        vmin    = 0.05,
        vmax    = 0.85,
        aspect  = 'auto'
    )

    plt.colorbar(im, ax=axes[idx], shrink=0.8, label='NDVI')

    axes[idx].set_title(season_name, fontsize=11,
                        fontweight='bold')
    axes[idx].set_xlabel('Longitude', fontsize=9)
    axes[idx].set_ylabel('Latitude',  fontsize=9)
    axes[idx].grid(True, linestyle='--',
                   alpha=0.3, color='white')

fig.suptitle(
    'Seasonal NDVI Maps - Khamanon Block\n'
    'Fatehgarh Sahib, Punjab (Fake Data)',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps', 'crop_seasonal_maps.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/crop_seasonal_maps.png")

# ============================================
# PLOT 3: Stress Detection Map
# Flags zones below NDVI threshold
# During rice peak (September)
# ============================================

np.random.seed(99)

ndvi_sep = (
    0.68
    + 0.10 * lat_norm
    - 0.08 * lon_norm
    + np.random.normal(0, 0.06, (n_grid, n_grid))
)
ndvi_sep[(lon_norm > 0.70) & (lat_norm < 0.35)] = \
    np.random.uniform(0.08, 0.20,
    ndvi_sep[(lon_norm > 0.70) &
             (lat_norm < 0.35)].shape)
ndvi_sep = np.clip(ndvi_sep, 0.05, 0.85)

# Stress = NDVI below 0.40 during peak season
stress_map = np.where(ndvi_sep < 0.40, 1, 0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: NDVI map
im1 = axes[0].imshow(
    ndvi_sep,
    extent = [lon_min, lon_max, lat_min, lat_max],
    origin = 'lower',
    cmap   = 'RdYlGn',
    vmin   = 0.05, vmax=0.85,
    aspect = 'auto'
)
plt.colorbar(im1, ax=axes[0], shrink=0.8, label='NDVI')
axes[0].set_title(
    'September NDVI\n(Rice Peak Season)',
    fontsize=11, fontweight='bold'
)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

# Right: Stress detection
im2 = axes[1].imshow(
    stress_map,
    extent = [lon_min, lon_max, lat_min, lat_max],
    origin = 'lower',
    cmap   = 'RdYlGn_r',
    aspect = 'auto'
)
axes[1].set_title(
    'Crop Stress Detection Map\n(Red = NDVI < 0.40, Stress Zone)',
    fontsize=11, fontweight='bold'
)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')

stressed = mpatches.Patch(color='red',   label='Stressed / Urban')
healthy  = mpatches.Patch(color='green', label='Healthy Crop')
axes[1].legend(handles=[stressed, healthy],
               loc='lower right', fontsize=9)

fig.suptitle(
    'Crop Stress Detection - Khamanon Block\n'
    'Fatehgarh Sahib, Punjab (Fake Data)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps', 'crop_stress_detection.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/crop_stress_detection.png")

# ============================================
# SAVE NDVI TIME SERIES AS CSV
# (used by dashboard)
# ============================================

ndvi_df = pd.DataFrame({
    'month'          : months,
    'ndvi_healthy'   : ndvi_healthy,
    'ndvi_stressed'  : ndvi_stressed,
    'ndvi_urban'     : ndvi_urban,
    'ndvi_vegetation': ndvi_vegetation
})

ndvi_df.to_csv(
    os.path.join('..', 'data', 'ndvi_timeseries.csv'),
    index=False
)
print("Saved: data/ndvi_timeseries.csv")

print("\n" + "=" * 50)
print("  STAGE 4 COMPLETE")
print("=" * 50)
print("\nWhat was built:")
print("  - NDVI time series chart (12 months, 4 zones)")
print("  - Seasonal NDVI maps (4 key months)")
print("  - Crop stress detection map")
print("  - NDVI data saved for dashboard")
print("\nGo open maps/ folder to see your crop maps.")