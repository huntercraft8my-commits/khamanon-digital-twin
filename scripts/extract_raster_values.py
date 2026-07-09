# ============================================
# DSS - KHAMANON BLOCK
# Script 3: Extract Raster Values at
#           Sample Point Locations
# ============================================

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
import os

print("=" * 55)
print("  PHASE 1 — STEP 1.3: EXTRACT RASTER VALUES")
print("=" * 55)

# ============================================
# LOAD VALIDATED SOIL DATA
# ============================================

df = pd.read_csv(os.path.join('..', 'data', 'soil_data_validated.csv'))
print(f"\nSample points loaded: {len(df)}")

# ============================================
# RASTER FILES TO EXTRACT FROM
# ============================================

raster_dir = os.path.join('..', 'rasters')

rasters = {
    'dem'          : os.path.join(raster_dir, 'dem.tif'),
    'slope'        : os.path.join(raster_dir, 'slope.tif'),
    'aspect'       : os.path.join(raster_dir, 'aspect.tif'),
    'lulc'         : os.path.join(raster_dir, 'lulc.tif'),
    'lithology'    : os.path.join(raster_dir, 'lithology.tif'),
    'geomorphology': os.path.join(raster_dir, 'geomorphology.tif'),
}

# ============================================
# CHECK RASTER CRS AND PROPERTIES
# ============================================

print("\nRaster file information:")
print("-" * 55)

for name, path in rasters.items():
    with rasterio.open(path) as src:
        print(f"\n  {name}:")
        print(f"    CRS     : {src.crs}")
        print(f"    Size    : {src.width} x {src.height} pixels")
        print(f"    Bounds  : {src.bounds}")
        print(f"    NoData  : {src.nodata}")
        print(f"    Dtype   : {src.dtypes[0]}")

# ============================================
# DETERMINE COORDINATE SYSTEM TO USE
# Your rasters are UTM Zone 43N
# Your points have both UTM and WGS84
# We use UTM coordinates for extraction
# ============================================

print("\n" + "-" * 55)
print("Using UTM Zone 43N coordinates for extraction")
print(f"Point Easting  range: {df['easting_utm'].min():.0f} to {df['easting_utm'].max():.0f}")
print(f"Point Northing range: {df['northing_utm'].min():.0f} to {df['northing_utm'].max():.0f}")

# ============================================
# EXTRACT FUNCTION
# For each sample point, find which pixel
# it falls in and read that pixel's value
# ============================================

def extract_values(raster_path, eastings, northings, name):
    values = []
    n_outside = 0
    n_nodata  = 0

    with rasterio.open(raster_path) as src:
        nodata = src.nodata

        for e, n in zip(eastings, northings):
            try:
                # Convert coordinates to row/col
                row, col = src.index(e, n)

                # Check within bounds
                if (0 <= row < src.height and
                    0 <= col < src.width):
                    val = src.read(1)[row, col]

                    # Check for nodata
                    if nodata is not None and val == nodata:
                        values.append(np.nan)
                        n_nodata += 1
                    elif val == -9999 or val == -32768:
                        values.append(np.nan)
                        n_nodata += 1
                    else:
                        values.append(float(val))
                else:
                    values.append(np.nan)
                    n_outside += 1

            except Exception:
                values.append(np.nan)
                n_outside += 1

    print(f"  {name:<15}: extracted {len(values)} values | "
          f"outside={n_outside} | nodata={n_nodata}")
    return values

# ============================================
# RUN EXTRACTION FOR ALL RASTERS
# ============================================

print("\nExtracting values...")
print("-" * 55)

eastings  = df['easting_utm'].values
northings = df['northing_utm'].values

for raster_name, raster_path in rasters.items():
    df[raster_name] = extract_values(
        raster_path,
        eastings,
        northings,
        raster_name
    )

# ============================================
# CHECK EXTRACTION RESULTS
# ============================================

print("\nExtracted column statistics:")
print("-" * 55)

extracted_cols = list(rasters.keys())

for col in extracted_cols:
    n_valid   = df[col].notna().sum()
    n_missing = df[col].isna().sum()
    if n_valid > 0:
        print(f"  {col:<15}: valid={n_valid} | "
              f"missing={n_missing} | "
              f"min={df[col].min():.2f} | "
              f"max={df[col].max():.2f} | "
              f"mean={df[col].mean():.2f}")
    else:
        print(f"  {col:<15}: NO VALUES EXTRACTED — check CRS")

# ============================================
# HANDLE ANY REMAINING MISSING VALUES
# Fill with median of extracted values
# ============================================

total_missing = df[extracted_cols].isnull().sum().sum()
if total_missing > 0:
    print(f"\nFilling {total_missing} missing extracted values with medians...")
    for col in extracted_cols:
        n = df[col].isnull().sum()
        if n > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"  {col}: filled {n} with median={med:.3f}")

# ============================================
# SAVE ENRICHED DATASET
# ============================================

out_path = os.path.join('..', 'data', 'soil_with_covariates.csv')
df.to_csv(out_path, index=False)

print("\n" + "=" * 55)
print("  STEP 1.3 COMPLETE")
print("=" * 55)
print(f"\nEnriched dataset saved: data/soil_with_covariates.csv")
print(f"Total columns now: {len(df.columns)}")
print("\nAll columns:")
for c in df.columns:
    print(f"  - {c}")