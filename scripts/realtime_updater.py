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
import os as _os
import json as _json

_gee_creds = _os.environ.get('GEE_CREDENTIALS')
if _gee_creds:
    # Running on Render Cloud Container — use Service Account JSON Key Environment Variable
    _key = _json.loads(_gee_creds)
    _credentials = ee.ServiceAccountCredentials(
        email=_key['client_email'],
        key_data=_key['private_key']
    )
    ee.Initialize(_credentials, project='optimum-archery-387602')
    print("  GEE: Authenticated via service account key (Render Cloud Environment)")
else:
    # Running locally on your desktop machine — fallback to your standard system user credentials profile
    ee.Initialize(project='optimum-archery-387602')
    print("  GEE: Authenticated via local system credentials profile")
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
# ── SATELLITE ACQUISITION LOG ─────────────────────────────────────
print("  Building satellite acquisition log...")

all_passes = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(khamanon)
              .filterDate(start, end)
              .sort('system:time_start'))

try:
    ts_list  = all_passes.aggregate_array('system:time_start').getInfo()
    cc_list  = all_passes.aggregate_array('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    sat_list = all_passes.aggregate_array('SPACECRAFT_NAME').getInfo()

    log_rows = []
    for ts, cc, sat in zip(ts_list, cc_list, sat_list):
        dt     = datetime.utcfromtimestamp(ts / 1000)
        date_s = dt.strftime('%Y-%m-%d')
        month  = dt.month
        cc     = round(cc or 0, 1)

        if 6 <= month <= 10:
            season, threshold = 'Kharif', 50
        else:
            season, threshold = 'Rabi', 25

        if cc <= threshold:
            status = 'ACCEPTED'
            action = f'Used in composite — NDVI extracted at 208 points'
        else:
            status = 'REJECTED'
            action = (f'Skipped — cloud cover {cc}% exceeds '
                      f'{threshold}% {season} QA threshold')

        log_rows.append({
            'date'      : date_s,
            'satellite' : sat or 'Sentinel-2',
            'cloud_pct' : cc,
            'status'    : status,
            'season'    : season,
            'threshold' : threshold,
            'action'    : action
        })

    log_df   = pd.DataFrame(log_rows)
    log_path = os.path.join(base, '..', 'data', 'satellite_log.csv')
    if os.path.exists(log_path):
        old_log = pd.read_csv(log_path)
        log_df  = pd.concat([old_log, log_df], ignore_index=True)
        log_df  = (log_df
                   .drop_duplicates(subset=['date', 'satellite', 'cloud_pct'])
                   .sort_values('date')
                   .reset_index(drop=True))
    log_df.to_csv(log_path, index=False)
    accepted = (log_df['status'] == 'ACCEPTED').sum()
    rejected = (log_df['status'] == 'REJECTED').sum()
    print(f"  Sat log: {len(log_df)} total passes — "
          f"{accepted} accepted, {rejected} rejected → satellite_log.csv")

except Exception as e:
    print(f"  Sat log failed: {e}")

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

# ── RISK SCORING FUNCTIONS ───────────────────────────────────────
# Ported from risk_assessment.py — no matplotlib, safe for headless Render

def _normalize(val, low, high):
    return np.clip((val - low) / (high - low), 0, 1)

def _soil_degradation_risk(pH, OC, bulk_density, CaCO3):
    return np.round(
        0.35 * _normalize(pH, 7.0, 9.5) * 100
        + 0.35 * (1 - _normalize(OC, 0.0, 0.75)) * 100
        + 0.20 * _normalize(bulk_density, 150, 350) * 100
        + 0.10 * _normalize(CaCO3, 0, 5) * 100, 1)

def _crop_failure_risk(OC, available_N, ndvi, temp, humidity):
    hum_risk = (
        _normalize(40 - humidity, 0, 40) * 100 if humidity < 40
        else _normalize(humidity - 80, 0, 20) * 100 if humidity > 80
        else 0
    )
    return np.round(
        0.35 * (1 - _normalize(ndvi, 0.1, 0.8)) * 100
        + 0.25 * (1 - _normalize(available_N, 50, 400)) * 100
        + 0.15 * (1 - _normalize(OC, 0.1, 0.75)) * 100
        + 0.15 * _normalize(temp, 25, 45) * 100
        + 0.10 * hum_risk, 1)

def _salinity_risk(EC, pH, CaCO3, OC):
    return np.round(np.clip(
        0.45 * _normalize(EC, 0, 1.0) * 100
        + 0.30 * _normalize(pH, 7.5, 9.5) * 100
        + 0.15 * _normalize(CaCO3, 0, 5) * 100
        - 0.10 * _normalize(OC, 0, 0.75) * 30,
        0, 100), 1)

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

    # ── SAVE UPDATED PREDICTION GRID ────────────────────────────
    grid_path = os.path.join(base, '..', 'data', 'real_prediction_grid.csv')
    grid_new.to_csv(grid_path, index=False)
    print(f"\nSaved updated grid → real_prediction_grid.csv "
          f"({len(grid_new):,} points)")

    # ── WRITE last_update.json ───────────────────────────────────
    accepted = int(
        pd.read_csv(os.path.join(base, '..', 'data', 'satellite_log.csv'))
        .query("status == 'ACCEPTED'")['status'].count()
    ) if os.path.exists(
        os.path.join(base, '..', 'data', 'satellite_log.csv')
    ) else count
    with open(timestamp_path, 'w') as f:
        json.dump({
            "last_run"      : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "sentinel2_date": latest_date,
            "images_used"   : accepted,
            "ndvi_mean"     : round(float(spectral_new['NDVI'].mean()), 4),
            "ndbi_mean"     : round(float(spectral_new['NDBI'].mean()), 4),
            "alerts"        : [],
            "status"        : "SUCCESS"
        }, f, indent=2)
    print(f"Updated last_update.json — sentinel2_date: {latest_date}")

    # ── APPEND TO NDVI ZONE TIME SERIES ─────────────────────────
    print("\nUpdating NDVI zone time series...")
    _month_label = today.strftime('%b-%Y')  # e.g. "Jun-2026"

    _zone_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([76.34, 30.84]),
                   {'zone': 'Healthy Cropland (North)'}),
        ee.Feature(ee.Geometry.Point([76.36, 30.79]),
                   {'zone': 'Stressed Cropland (Central)'}),
        ee.Feature(ee.Geometry.Point([76.41, 30.74]),
                   {'zone': 'Peri-urban SE'}),
        ee.Feature(ee.Geometry.Point([76.30, 30.80]),
                   {'zone': 'Vegetation West'}),
    ])

    _zone_result = NDVI.sampleRegions(
        collection=_zone_fc,
        scale=10,
        geometries=False
    ).getInfo()

    _zone_ndvi = {
        f['properties']['zone']: round(
            f['properties'].get('NDVI', float('nan')), 6
        )
        for f in _zone_result['features']
    }

    _ndvi_path = os.path.join(base, '..', 'data', 'ndvi_processed.csv')
    _ndvi_hist = (pd.read_csv(_ndvi_path)
                  if os.path.exists(_ndvi_path)
                  else pd.DataFrame())

    if _month_label not in _ndvi_hist.get(
            'month', pd.Series(dtype=str)).values:
        _new_row = pd.DataFrame([{
            'month'                      : _month_label,
            'Healthy Cropland (North)'   : _zone_ndvi.get(
                'Healthy Cropland (North)',    float('nan')),
            'Stressed Cropland (Central)': _zone_ndvi.get(
                'Stressed Cropland (Central)', float('nan')),
            'Peri-urban SE'              : _zone_ndvi.get(
                'Peri-urban SE',               float('nan')),
            'Vegetation West'            : _zone_ndvi.get(
                'Vegetation West',             float('nan')),
        }])
        _ndvi_hist = pd.concat(
            [_ndvi_hist, _new_row], ignore_index=True
        )
        _ndvi_hist.to_csv(_ndvi_path, index=False)
        print(f"  Appended {_month_label} → ndvi_processed.csv")
        for _z, _v in _zone_ndvi.items():
            print(f"    {_z:<32}: {_v:.4f}")
    else:
        print(f"  {_month_label} already in ndvi_processed.csv — skipping")

    # ── RECOMPUTE RISK SCORES ────────────────────────────────────
    print("\nRecomputing risk scores...")
    _wx_path = os.path.join(base, '..', 'data', 'current_weather.json')
    if os.path.exists(_wx_path):
        with open(_wx_path) as _f:
            _wx = json.load(_f)
        _temp     = _wx.get('temperature', 35)
        _humidity = _wx.get('humidity', 45)
    else:
        _temp, _humidity = 35, 45
    _ndvi_mean = float(spectral_new['NDVI'].mean())

    for _df in [soil, grid_new]:
        _df['degradation_risk'] = _soil_degradation_risk(
            _df['pH'].values, _df['OC'].values,
            _df['bulk_density'].values, _df['CaCO3'].values)
        _df['crop_failure_risk'] = _crop_failure_risk(
            _df['OC'].values, _df['available_N'].values,
            _ndvi_mean, _temp, _humidity)
        _df['salinity_risk'] = _salinity_risk(
            _df['EC'].values, _df['pH'].values,
            _df['CaCO3'].values, _df['OC'].values)
        _df['overall_risk'] = (
            0.40 * _df['degradation_risk']
            + 0.40 * _df['crop_failure_risk']
            + 0.20 * _df['salinity_risk'])

    def _risk_cat(s):
        return 'HIGH' if s >= 70 else 'MODERATE' if s >= 45 else 'LOW'
    for _col in ['degradation_risk', 'crop_failure_risk', 'salinity_risk']:
        soil[_col.replace('_risk', '_cat')] = soil[_col].apply(_risk_cat)

    soil[[
        'sample_id', 'latitude', 'longitude',
        'easting_utm', 'northing_utm',
        'degradation_risk', 'crop_failure_risk',
        'salinity_risk', 'overall_risk',
        'degradation_cat', 'crop_failure_cat', 'salinity_cat'
    ]].to_csv(
        os.path.join(base, '..', 'data', 'point_risk_scores.csv'),
        index=False)

    grid_new[[
        'easting', 'northing',
        'degradation_risk', 'crop_failure_risk',
        'salinity_risk', 'overall_risk'
    ]].to_csv(
        os.path.join(base, '..', 'data', 'grid_risk_scores.csv'),
        index=False)

    print(f"  Degradation  : mean={soil['degradation_risk'].mean():.1f}/100")
    print(f"  Crop failure : mean={soil['crop_failure_risk'].mean():.1f}/100")
    print(f"  Salinity     : mean={soil['salinity_risk'].mean():.1f}/100")
    print("  Saved: point_risk_scores.csv, grid_risk_scores.csv")

# ── WEATHER FETCH PIPELINE ──────────────────────────────────────
import requests as _req
import os as _os
import json as _json

def _fetch_weather():
    # Read from env (GitHub Secret / Render env var); fall back to local literal.
    # NOTE: the literal below is already public in git history — rotate it.
    OWM_KEY = _os.environ.get("OWM_KEY", "aad8b57ba2b85482fa8be2bbbed79489")
    
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat=30.795&lon=76.352&appid={OWM_KEY}&units=metric"
    )
    try:
        r = _req.get(url, timeout=10)
        d = r.json()
        
        # Check if the API key is active or unauthorized
        if d.get("cod") != 200:
            print(f"  Weather API Error: {d.get('message', 'Unknown Error')}")
            return
            
        wx = {
            "temperature" : round(d["main"]["temp"], 1),
            "humidity"    : d["main"]["humidity"],
            "description" : d["weather"][0]["description"],
            "wind_speed"  : d["wind"]["speed"],
            "rain_1h"     : d.get("rain", {}).get("1h", 0),
            "advisories"  : []
        }
        
        # Try to resolve folder path safely
        try:
            script_dir = _os.path.dirname(_os.path.abspath(__file__))
        except NameError:
            script_dir = _os.getcwd()
            
        wx_path = _os.path.join(script_dir, '..', 'data', 'current_weather.json')
        
        with open(wx_path, 'w') as f:
            _json.dump(wx, f)
            
        print(f"  Success! Weather: {wx['temperature']}°C | {wx['description']} | RH {wx['humidity']}%")
    except Exception as e:
        print(f"  Weather fetch failed: {e}")

_fetch_weather()
