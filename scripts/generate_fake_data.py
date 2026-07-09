# ============================================
# DSS - KHAMANON BLOCK
# Script 1: Fake Data Generator
# ============================================

import numpy as np
import pandas as pd
import os

# Set random seed so fake data is same every time you run it
np.random.seed(42)

# ============================================
# KHAMANON BLOCK BOUNDARIES (real coordinates)
# Latitude:  30.38 to 30.52
# Longitude: 76.27 to 76.42
# ============================================

n_samples = 50  # number of fake soil sample points

# Generate random coordinates within Khamanon Block
latitudes  = np.random.uniform(30.38, 30.52, n_samples)
longitudes = np.random.uniform(76.27, 76.42, n_samples)

# ============================================
# FAKE LAND USE CLASSES
# Based on your research area reality
# ============================================

lulc_classes = ['Cropland', 'Built-up', 'Fallow', 'Vegetation', 'Water body']
lulc_weights = [0.55, 0.20, 0.12, 0.08, 0.05]  # cropland dominates in Khamanon
lulc = np.random.choice(lulc_classes, size=n_samples, p=lulc_weights)

# ============================================
# FAKE SOIL PROPERTIES
# Ranges based on Punjab soil literature
# from your own synopsis references
# ============================================

pH = np.where(
    lulc == 'Built-up',
    np.random.uniform(8.0, 8.8, n_samples),   # built-up = higher pH
    np.where(
        lulc == 'Cropland',
        np.random.uniform(7.5, 8.2, n_samples), # cropland = moderate pH
        np.random.uniform(7.2, 8.0, n_samples)  # other = lower pH
    )
)

organic_carbon = np.where(
    lulc == 'Vegetation',
    np.random.uniform(0.55, 0.85, n_samples),  # vegetation = higher OC
    np.where(
        lulc == 'Built-up',
        np.random.uniform(0.15, 0.30, n_samples), # built-up = lowest OC
        np.random.uniform(0.30, 0.60, n_samples)  # others = moderate OC
    )
)

ec = np.where(
    lulc == 'Built-up',
    np.random.uniform(0.45, 0.90, n_samples),  # built-up = higher EC
    np.random.uniform(0.15, 0.50, n_samples)   # others = lower EC
)

available_N = np.random.uniform(150, 280, n_samples)  # kg/ha
available_P = np.random.uniform(10,  35,  n_samples)  # kg/ha
available_K = np.random.uniform(110, 220, n_samples)  # kg/ha
CEC         = np.random.uniform(8,   22,  n_samples)  # meq/100g

# ============================================
# FAKE SPECTRAL INDICES
# Simulated Sentinel-2 derived values
# ============================================

NDVI = np.where(
    lulc == 'Cropland',
    np.random.uniform(0.45, 0.75, n_samples),  # healthy crops = high NDVI
    np.where(
        lulc == 'Vegetation',
        np.random.uniform(0.55, 0.85, n_samples),
        np.where(
            lulc == 'Built-up',
            np.random.uniform(0.05, 0.20, n_samples), # built-up = low NDVI
            np.random.uniform(0.15, 0.45, n_samples)
        )
    )
)

NDBI = np.where(
    lulc == 'Built-up',
    np.random.uniform(0.20, 0.55, n_samples),  # built-up = high NDBI
    np.random.uniform(-0.30, 0.15, n_samples)  # others = low NDBI
)

SAVI = NDVI * 0.88 + np.random.uniform(-0.03, 0.03, n_samples)
BSI  = np.where(
    lulc == 'Fallow',
    np.random.uniform(0.20, 0.50, n_samples),  # fallow = high bare soil
    np.random.uniform(-0.10, 0.25, n_samples)
)

# ============================================
# FAKE TERRAIN (DEM derivatives)
# Khamanon is flat alluvial plain
# ============================================

elevation = np.random.uniform(235, 265, n_samples)  # metres above sea level
slope     = np.random.uniform(0.1, 3.5, n_samples)  # degrees, flat terrain
aspect    = np.random.uniform(0, 360, n_samples)     # degrees

# ============================================
# ASSEMBLE INTO A DATAFRAME
# ============================================

df = pd.DataFrame({
    'sample_id'       : range(1, n_samples + 1),
    'latitude'        : np.round(latitudes, 5),
    'longitude'       : np.round(longitudes, 5),
    'LULC'            : lulc,
    'pH'              : np.round(pH, 2),
    'organic_carbon'  : np.round(organic_carbon, 3),
    'EC'              : np.round(ec, 3),
    'available_N'     : np.round(available_N, 1),
    'available_P'     : np.round(available_P, 1),
    'available_K'     : np.round(available_K, 1),
    'CEC'             : np.round(CEC, 2),
    'NDVI'            : np.round(NDVI, 4),
    'NDBI'            : np.round(NDBI, 4),
    'SAVI'            : np.round(SAVI, 4),
    'BSI'             : np.round(BSI, 4),
    'elevation'       : np.round(elevation, 1),
    'slope'           : np.round(slope, 2),
    'aspect'          : np.round(aspect, 1)
})

# ============================================
# SAVE TO DATA FOLDER
# ============================================

output_path = os.path.join('..', 'data', 'soil_samples_fake.csv')
df.to_csv(output_path, index=False)

print("=" * 45)
print("  FAKE DATA GENERATED SUCCESSFULLY")
print("=" * 45)
print(f"  Total sample points : {n_samples}")
print(f"  Saved to            : data/soil_samples_fake.csv")
print("=" * 45)
print("\nLULC Distribution:")
print(df['LULC'].value_counts().to_string())
print("\nSoil Property Summary:")
print(df[['pH','organic_carbon','EC','available_N',
          'available_P','available_K']].describe().round(3).to_string())