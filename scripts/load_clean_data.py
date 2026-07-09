# ============================================
# DSS - KHAMANON BLOCK
# Real Data Pipeline
# Script 1: Load and Clean Soil Data
# ============================================

import pandas as pd
import numpy as np
import os

print("=" * 55)
print("  PHASE 1 — STEP 1.1: LOADING REAL SOIL DATA")
print("=" * 55)

# ============================================
# LOAD EXCEL — skip row 0 which is our
# real header, fix column names manually
# ============================================

df_raw = pd.read_excel(
    os.path.join('..', 'data', 'clhs_soil_clean.xlsx'),
    engine='openpyxl',
    header=None     # don't auto-detect header
)

print(f"\nRaw shape: {df_raw.shape}")

# Row 0 = column names, Row 1 onwards = data
# Extract column names from row 0
col_names = df_raw.iloc[0].tolist()
print(f"\nRaw column names detected:")
for i, c in enumerate(col_names):
    print(f"  Col {i}: {c}")

# ============================================
# CLEAN COLUMN NAMES
# ============================================

clean_columns = [
    'sample_id',
    'easting_utm',
    'northing_utm',
    'longitude',
    'latitude',
    'bulk_density',
    'pH',
    'EC',
    'K2O',
    'available_P',
    'OC',
    'CaCO3',
    'available_N',
    'CEC'
]

# Use data from row 1 onwards
df = df_raw.iloc[1:].copy()
df.columns = clean_columns
df = df.reset_index(drop=True)

print(f"\nData shape after cleaning header: {df.shape}")

# ============================================
# CONVERT COLUMNS TO CORRECT DATA TYPES
# All soil values should be numeric
# ============================================

numeric_cols = [
    'easting_utm', 'northing_utm',
    'longitude', 'latitude',
    'bulk_density', 'pH', 'EC',
    'K2O', 'available_P', 'OC',
    'CaCO3', 'available_N', 'CEC'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nData types after conversion:")
print(df.dtypes)

# ============================================
# CHECK MISSING VALUES
# ============================================

print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing)
total_missing = missing.sum()
print(f"\nTotal missing values: {total_missing}")

# ============================================
# HANDLE MISSING VALUES
# Strategy: fill with column median
# Median is robust to outliers
# Better than mean for skewed soil data
# ============================================

if total_missing > 0:
    print("\nFilling missing values with column median...")
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  {col}: filled {n_missing} values with median={median_val:.3f}")

# ============================================
# CHECK FOR ZERO VALUES IN SOIL DATA
# Zero nitrogen / zero CEC is unrealistic
# Flag and replace with median
# ============================================

print("\nChecking for unrealistic zero values...")

soil_cols = ['bulk_density', 'pH', 'EC', 'K2O',
             'available_P', 'OC', 'CaCO3',
             'available_N', 'CEC']

for col in soil_cols:
    n_zeros = (df[col] == 0).sum()
    if n_zeros > 0:
        median_val = df[df[col] > 0][col].median()
        df.loc[df[col] == 0, col] = median_val
        print(f"  {col}: replaced {n_zeros} zeros with median={median_val:.3f}")

# ============================================
# BASIC COORDINATE CHECK
# Make sure points are within Khamanon area
# ============================================

print("\nCoordinate range check:")
print(f"  Latitude  : {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
print(f"  Longitude : {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
print(f"  Easting   : {df['easting_utm'].min():.0f} to {df['easting_utm'].max():.0f}")
print(f"  Northing  : {df['northing_utm'].min():.0f} to {df['northing_utm'].max():.0f}")

# ============================================
# SUMMARY STATISTICS
# ============================================

print("\nSoil Property Summary:")
print("-" * 55)
summary = df[soil_cols].describe().round(3)
print(summary.to_string())

# ============================================
# SAVE CLEANED DATA
# ============================================

out_path = os.path.join('..', 'data', 'soil_data_clean.csv')
df.to_csv(out_path, index=False)

print("\n" + "=" * 55)
print("  STEP 1.1 COMPLETE")
print("=" * 55)
print(f"\nClean data saved: data/soil_data_clean.csv")
print(f"Total samples   : {len(df)}")
print(f"Total columns   : {len(df.columns)}")
print("\nColumns in clean file:")
for c in df.columns:
    print(f"  - {c}")