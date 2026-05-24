# ============================================
# DIGITAL TWIN - KHAMANON BLOCK
# Script 9: Real-Time Auto Updater
# Connects to GEE, gets latest Sentinel-2,
# reruns RF models, updates predictions
# ============================================

import ee
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime, timedelta

print("=" * 55)
print("  REAL-TIME UPDATER — KHAMANON BLOCK")
print("=" * 55)

# ============================================
# STEP 1: CONNECT TO GEE
# ============================================

print("\nConnecting to Google Earth Engine...")
ee.Initialize(project='optimum-archery-387602')
print("Connected.")

# ============================================
# STEP 2: DEFINE KHAMANON AREA AND POINTS
# ============================================

khamanon = ee.Geometry.Rectangle(
    [76.26, 30.71, 76.45, 30.88]
)

# Load your validated soil data for coordinates
base = os.path.dirname(os.path.abspath(__file__))

soil = pd.read_csv(
    os.path.join(base, '..', 'data',
                 'soil_data_validated.csv')
)

# Create GEE feature collection from coordinates
features = [
    ee.Feature(
        ee.Geometry.Point(
            [row['longitude'], row['latitude']]
        ),
        {'point_id': int(i)}
    )
    for i, row in soil.iterrows()
]
points = ee.FeatureCollection(features)

print(f"Sample points defined: {len(soil)}")

# ============================================
# STEP 3: FIND LATEST SENTINEL-2 IMAGE DATE
# ============================================

print("\nChecking latest Sentinel-2 imagery...")

# Get images from last 30 days
today     = datetime.now()
days_back = 30
start     = (today - timedelta(days=days_back)
             ).strftime('%Y-%m-%d')
end       = today.strftime('%Y-%m-%d')

s2_recent = ee.ImageCollection(
    'COPERNICUS/S2_SR_HARMONIZED'
).filterBounds(khamanon
).filterDate(start, end
).filter(
    ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)
)

# Get image count and most recent date
count = s2_recent.size().getInfo()
print(f"Images found (last {days_back} days): {count}")

if count == 0:
    print("No clear images in last 30 days.")
    print("Extending search to 60 days...")
    start = (today - timedelta(days=60)
             ).strftime('%Y-%m-%d')
    s2_recent = ee.ImageCollection(
        'COPERNICUS/S2_SR_HARMONIZED'
    ).filterBounds(khamanon
    ).filterDate(start, end
    ).filter(
        ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)
    )
    count = s2_recent.size().getInfo()
    print(f"Images found (last 60 days): {count}")

# Get date of most recent image
latest_img  = s2_recent.sort(
    'system:time_start', False
).first()
latest_date = datetime.fromtimestamp(
    latest_img.get('system:time_start'
                   ).getInfo() / 1000
).strftime('%Y-%m-%d')

print(f"Latest image date: {latest_date}")

# ============================================
# STEP 4: CHECK IF UPDATE IS NEEDED
# Compare with last update timestamp
# ============================================

timestamp_path = os.path.join(
    base, '..', 'data', 'last_update.json'
)

needs_update = True

if os.path.exists(timestamp_path):
    with open(timestamp_path, 'r') as f:
        last = json.load(f)
    last_date = last.get('sentinel2_date', '')
    last_run  = last.get('last_run', 'Never')
    print(f"\nLast update run  : {last_run}")
    print(f"Last S2 date used: {last_date}")

    if last_date == latest_date:
        print("\nNo new imagery since last update.")
        print("Forcing update anyway for demonstration...")
        needs_update = True
    else:
        print(f"\nNew imagery available: {latest_date}")
        needs_update = True
else:
    print("\nNo previous update found. Running first update.")

# ============================================
# STEP 5: EXTRACT FRESH SPECTRAL INDICES
# ============================================

if needs_update:
    print("\n" + "-" * 40)
    print("Extracting fresh spectral indices...")

    def mask_s2_clouds(image):
        qa  = image.select('QA60')
        mask = (
            qa.bitwiseAnd(1 << 10).eq(0)
            .And(qa.bitwiseAnd(1 << 11).eq(0))
        )
        return (image.updateMask(mask)
                .divide(10000)
                .copyProperties(
                    image,
                    ['system:time_start']
                ))

    # Use last 30 days composite
    s2 = (ee.ImageCollection(
              'COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(khamanon)
          .filterDate(start, end)
          .filter(ee.Filter.lt(
              'CLOUDY_PIXEL_PERCENTAGE', 30))
          .map(mask_s2_clouds)
          .median())

    # Compute spectral indices
    NDVI = s2.normalizedDifference(
        ['B8', 'B4']
    ).rename('NDVI')

    NDBI = s2.normalizedDifference(
        ['B11', 'B8']
    ).rename('NDBI')

    SAVI = s2.expression(
        '((NIR-RED)/(NIR+RED+0.5))*(1.5)',
        {
            'NIR': s2.select('B8'),
            'RED': s2.select('B4')
        }
    ).rename('SAVI')

    BSI = s2.expression(
        '((SWIR+RED)-(NIR+BLUE))'
        '/((SWIR+RED)+(NIR+BLUE))',
        {
            'SWIR': s2.select('B11'),
            'RED' : s2.select('B4'),
            'NIR' : s2.select('B8'),
            'BLUE': s2.select('B2')
        }
    ).rename('BSI')

    indices = (NDVI.addBands(NDBI)
                   .addBands(SAVI)
                   .addBands(BSI))

    # Extract at sample points
    print("Extracting at 208 sample points...")
    extracted = indices.reduceRegions(
        collection = points,
        reducer    = ee.Reducer.mean(),
        scale      = 10,
        crs        = 'EPSG:4326'
    )

    # Get results
    features_list = extracted.toList(
        extracted.size()
    ).getInfo()

    records = []
    for feat in features_list:
        props = feat['properties']
        records.append({
            'point_id': props.get('point_id', -1),
            'NDVI'    : props.get('NDVI',     np.nan),
            'NDBI'    : props.get('NDBI',     np.nan),
            'SAVI'    : props.get('SAVI',     np.nan),
            'BSI'     : props.get('BSI',      np.nan)
        })

    spectral_new = pd.DataFrame(records).sort_values(
        'point_id'
    ).reset_index(drop=True)

    # Fill any missing values
    for col in ['NDVI','NDBI','SAVI','BSI']:
        n = spectral_new[col].isnull().sum()
        if n > 0:
            med = spectral_new[col].median()
            spectral_new[col] = spectral_new[col].fillna(med)
            print(f"  Filled {n} missing {col}")

    print(f"Spectral extraction complete.")
    print(f"  NDVI mean: {spectral_new['NDVI'].mean():.4f}")
    print(f"  NDBI mean: {spectral_new['NDBI'].mean():.4f}")

    # ============================================
    # STEP 6: BUILD UPDATED FEATURE MATRIX
    # ============================================

    print("\nBuilding updated feature matrix...")

    raster_data = pd.read_csv(
        os.path.join(base, '..', 'data',
                     'soil_with_covariates.csv')
    )

    master_new = raster_data.copy()
    master_new['NDVI'] = spectral_new['NDVI'].values
    master_new['NDBI'] = spectral_new['NDBI'].values
    master_new['SAVI'] = spectral_new['SAVI'].values
    master_new['BSI']  = spectral_new['BSI'].values

    feature_cols = [
        'dem', 'slope', 'aspect',
        'lulc', 'lithology', 'geomorphology',
        'NDVI', 'NDBI', 'SAVI', 'BSI'
    ]

    X_new = master_new[feature_cols].fillna(
        master_new[feature_cols].median()
    )

    # ============================================
    # STEP 7: RERUN RF MODELS
    # ============================================

    print("Rerunning RF models with fresh indices...")

    target_cols = [
        'pH', 'OC', 'EC', 'K2O',
        'available_P', 'available_N',
        'CEC', 'bulk_density', 'CaCO3'
    ]

    grid_old = pd.read_csv(
        os.path.join(base, '..', 'data',
                     'real_prediction_grid.csv')
    )

    master_old = pd.read_csv(
        os.path.join(base, '..', 'data',
                     'master_training_data.csv')
    )

    # Update spectral stats for reporting
    changes = {}
    for col in ['NDVI','NDBI','SAVI','BSI']:
        old_mean = master_old[col].mean()
        new_mean = spectral_new[col].mean()
        changes[col] = {
            'old'  : round(old_mean, 4),
            'new'  : round(new_mean, 4),
            'delta': round(new_mean - old_mean, 4)
        }

    print("\nSpectral index changes:")
    for col, c in changes.items():
        direction = '↑' if c['delta'] > 0 else '↓'
        print(f"  {col}: {c['old']} → {c['new']} "
              f"({direction}{abs(c['delta']):.4f})")

    # ============================================
    # UPDATE SPECTRAL VALUES IN FULL GRID
    # Scale existing grid spectral values by the
    # ratio of new mean to old mean
    # This propagates the Sentinel-2 update
    # across all 19,290 grid points
    # ============================================

    grid_new = grid_old.copy()

    for col in ['NDVI','NDBI','SAVI','BSI']:
        old_mean = changes[col]['old']
        new_mean = changes[col]['new']
        if old_mean != 0:
            ratio = new_mean / old_mean
            grid_new[col] = grid_old[col] * ratio \
                if col in grid_old.columns \
                else new_mean
        else:
            grid_new[col] = new_mean

    # Build feature matrix for full grid
    grid_features = [
        'dem', 'slope', 'aspect',
        'lulc', 'lithology', 'geomorphology',
        'NDVI', 'NDBI', 'SAVI', 'BSI'
    ]

    # Check which feature cols exist in grid
    available = [c for c in grid_features
                 if c in grid_new.columns]
    missing   = [c for c in grid_features
                 if c not in grid_new.columns]

    if missing:
        print(f"\nNote: {missing} not in grid.")
        print("Using raster covariate means...")
        raster_means = master_new[
            feature_cols
        ].mean()
        for col in missing:
            grid_new[col] = raster_means.get(col, 0)

    X_grid = grid_new[grid_features].fillna(
        grid_new[grid_features].median()
    )

    # Predict soil properties on full grid
    print("\nUpdating soil predictions...")
    for target in target_cols:
        model_path = os.path.join(
            base, '..', 'models',
            f'rf_real_{target}.pkl'
        )
        model    = joblib.load(model_path)
        old_mean = grid_old[target].mean()
        grid_new[target] = model.predict(X_grid)
        new_mean = grid_new[target].mean()
        print(f"  {target:<15}: "
              f"{old_mean:.3f} → {new_mean:.3f}")