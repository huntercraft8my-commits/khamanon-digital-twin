# -*- coding: utf-8 -*-
# ================================================================
# Khamanon Training Points — Seasonal Classifier
# Input  : data/khamanon_trainingpts_ndvi_timeseries.csv
# Output : data/training_points_classified.csv
#          data/training_points_diagnostic.csv
# Purpose: Determine the TRUE seasonal class of each of the 139
#          training points using NDVI temporal profiles.
#          Critical for separating Spring Maize from Kharif Maize.
# ================================================================

import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.abspath(__file__))

# ── 1. LOAD RAW NDVI TIME SERIES ──────────────────────────────
csv_path = os.path.join(base, '..', 'data',
                        'khamanon_trainingpts_ndvi_timeseries.csv')

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])

print("=" * 60)
print("KHAMANON TRAINING POINTS — SEASONAL CLASSIFIER")
print("=" * 60)
print(f"\nLoaded {len(df):,} NDVI records")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Unique points: {df.groupby(['longitude','latitude']).ngroups}")
print(f"\nOriginal class distribution (one row per record, not per point):")
print(df['class_name'].value_counts())

# ── 2. CREATE POINT ID ────────────────────────────────────────
# Each unique (lon, lat) pair = one field point
df['point_id'] = (df['longitude'].round(6).astype(str)
                  + '_'
                  + df['latitude'].round(6).astype(str))

# Drop rows where NDVI is NaN (cloudy pixels for that point/date)
df_clean = df.dropna(subset=['NDVI']).copy()
print(f"\nValid (non-cloud) NDVI records: {len(df_clean):,}")

# ── 3. COMPUTE SEASONAL NDVI METRICS PER POINT ────────────────
# Define seasonal windows based on Punjab cropping calendar

# Rabi 2025-26 (Wheat / Spring Maize)
RABI_PEAK_START   = '2026-02-01'      # Wheat peak vegetative
RABI_PEAK_END     = '2026-03-15'
SPRING_PEAK_START = '2026-03-25'      # Spring Maize peak (DISTINCT from wheat)
SPRING_PEAK_END   = '2026-05-05'

# Kharif 2025 (Rice / Kharif Maize)
KHARIF_PEAK_START = '2025-08-01'      # Rice/Maize peak vegetative
KHARIF_PEAK_END   = '2025-09-30'
KHARIF_EARLY      = '2025-06-15'      # Flood/transplanting window
KHARIF_EARLY_END  = '2025-07-31'

def season_stats(point_data, label, start, end):
    """Get max, mean, count of NDVI within a season window."""
    s = point_data[(point_data['date'] >= start) &
                   (point_data['date'] <= end)]['NDVI']
    if len(s) == 0:
        return np.nan, np.nan, 0
    return s.max(), s.mean(), len(s)


print("\nComputing seasonal NDVI metrics per point...")

records = []
for point_id, grp in df_clean.groupby('point_id'):
    lon = grp['longitude'].iloc[0]
    lat = grp['latitude'].iloc[0]
    orig_class      = grp['class'].iloc[0]
    orig_class_name = grp['class_name'].iloc[0]
    orig_landuse    = grp['landuse'].iloc[0]

    # Rabi (wheat) window
    rabi_max, rabi_mean, rabi_n = season_stats(
        grp, 'rabi', RABI_PEAK_START, RABI_PEAK_END)

    # Spring (spring maize) window
    spring_max, spring_mean, spring_n = season_stats(
        grp, 'spring', SPRING_PEAK_START, SPRING_PEAK_END)

    # Kharif peak window
    kharif_max, kharif_mean, kharif_n = season_stats(
        grp, 'kharif', KHARIF_PEAK_START, KHARIF_PEAK_END)

    # Kharif early (flooding) window
    early_max, early_mean, early_n = season_stats(
        grp, 'early', KHARIF_EARLY, KHARIF_EARLY_END)

    # Year-round stability metrics
    annual_mean = grp['NDVI'].mean()
    annual_std  = grp['NDVI'].std()
    annual_min  = grp['NDVI'].min()

    records.append({
        'point_id'        : point_id,
        'longitude'       : lon,
        'latitude'        : lat,
        'original_class'  : orig_class,
        'original_name'   : orig_class_name,
        'original_landuse': orig_landuse,
        'rabi_max'        : rabi_max,
        'rabi_mean'       : rabi_mean,
        'spring_max'      : spring_max,
        'spring_mean'     : spring_mean,
        'kharif_max'      : kharif_max,
        'kharif_mean'     : kharif_mean,
        'kharif_early_max': early_max,
        'kharif_early_mean': early_mean,
        'annual_mean'     : annual_mean,
        'annual_std'      : annual_std,
        'annual_min'      : annual_min,
        'n_observations'  : len(grp)
    })

prof = pd.DataFrame(records)
print(f"Computed metrics for {len(prof)} unique points")

# ── 4. APPLY SEASONAL CLASSIFICATION RULES ────────────────────
# Each point gets TWO seasonal labels: one for Rabi, one for Kharif

def classify_rabi(row):
    """Classify each point for Rabi 2025-26 (Wheat / Spring Maize / Other)."""
    # AGROFORESTRY first — perennial high stable NDVI
    if (row['annual_mean'] >= 0.45
            and row['annual_std'] <= 0.13
            and row['annual_min'] >= 0.30):
        return 'Agroforestry'

    # SPRING MAIZE — high NDVI in late March / April (UNIQUE signature)
    # No other Punjab crop has high NDVI in this window
    if (row['spring_max'] >= 0.65
            and row['spring_max'] > row['rabi_max']):
        return 'Spring_Maize'

    # WHEAT — high NDVI in Feb-March, then DROPS by April
    if (row['rabi_max'] >= 0.55
            and (row['spring_max'] is np.nan
                 or row['spring_max'] < row['rabi_max'] - 0.10)):
        return 'Wheat'

    return 'Other'


def classify_kharif(row):
    """Classify each point for Kharif 2025 (Rice / Kharif Maize / Other)."""
    # AGROFORESTRY first
    if (row['annual_mean'] >= 0.45
            and row['annual_std'] <= 0.13
            and row['annual_min'] >= 0.30):
        return 'Agroforestry'

    # RICE — high Aug-Sep NDVI AND was flooded in Jun-Jul
    # Flooding indicator: very low NDVI in early Kharif (bare/water)
    if (row['kharif_max'] >= 0.60
            and row['kharif_early_max'] <= 0.40):
        return 'Rice'

    # KHARIF MAIZE — high Aug-Sep NDVI but NOT flooded
    # Maize planted in June already shows green by early Kharif
    if (row['kharif_max'] >= 0.55
            and row['kharif_early_max'] > 0.40):
        return 'Kharif_Maize'

    return 'Other'


prof['class_rabi_2025_26']  = prof.apply(classify_rabi,  axis=1)
prof['class_kharif_2025']   = prof.apply(classify_kharif, axis=1)

# ── 5. SUMMARY — what we discovered ────────────────────────────
print("\n" + "=" * 60)
print("RABI 2025-26 CLASSIFICATION RESULT")
print("=" * 60)
print(prof['class_rabi_2025_26'].value_counts())

print("\n" + "=" * 60)
print("KHARIF 2025 CLASSIFICATION RESULT")
print("=" * 60)
print(prof['class_kharif_2025'].value_counts())

# ── 6. DEEP DIVE ON THE 19 "Maize_Tall" POINTS ─────────────────
maize_pts = prof[prof['original_name'] == 'Maize_Tall'].copy()
print("\n" + "=" * 60)
print(f"DEEP DIVE — Original 'Maize_Tall' points (n = {len(maize_pts)})")
print("=" * 60)
print("\nClass assignment by season:")
print(maize_pts.groupby(['class_rabi_2025_26',
                         'class_kharif_2025']).size().to_string())

print("\nDetailed per-point profile (sorted by spring_max):")
maize_view = maize_pts[['longitude', 'latitude',
                        'rabi_max', 'spring_max',
                        'kharif_max', 'kharif_early_max',
                        'class_rabi_2025_26',
                        'class_kharif_2025']].copy()
maize_view = maize_view.round(3).sort_values('spring_max', ascending=False)
print(maize_view.to_string(index=False))

# ── 7. SANITY CHECKS ───────────────────────────────────────────
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# Wheat points — do they classify as wheat in Rabi?
wheat_pts = prof[prof['original_name'] == 'Wheat']
if len(wheat_pts) > 0:
    wheat_correct = (wheat_pts['class_rabi_2025_26'] == 'Wheat').sum()
    pct = 100 * wheat_correct / len(wheat_pts)
    print(f"\nWheat points: {wheat_correct} / {len(wheat_pts)} "
          f"classified as Wheat in Rabi 2025-26 ({pct:.1f}%)")
    print("Misclassified wheat points fell into:")
    print(wheat_pts[wheat_pts['class_rabi_2025_26'] != 'Wheat']
          ['class_rabi_2025_26'].value_counts().to_string())

# Agroforestry points
agr_pts = prof[prof['original_name'] == 'Agroforestry']
if len(agr_pts) > 0:
    agr_correct = (agr_pts['class_rabi_2025_26'] == 'Agroforestry').sum()
    print(f"\nAgroforestry points: {agr_correct} / {len(agr_pts)} "
          f"classified as Agroforestry")

# Others — these were vegetables/mustard/etc
oth_pts = prof[prof['original_name'] == 'Others']
if len(oth_pts) > 0:
    print(f"\nOthers points: distribution by Rabi class:")
    print(oth_pts['class_rabi_2025_26'].value_counts().to_string())
    print(f"\nOthers points: distribution by Kharif class:")
    print(oth_pts['class_kharif_2025'].value_counts().to_string())

# ── 8. SAVE ────────────────────────────────────────────────────
out_path = os.path.join(base, '..', 'data',
                        'training_points_classified.csv')
prof.to_csv(out_path, index=False)
print(f"\n\nSaved: {out_path}")

# Diagnostic file for the 19 maize points only
maize_diag_path = os.path.join(base, '..', 'data',
                               'training_points_maize_diagnostic.csv')
maize_pts.to_csv(maize_diag_path, index=False)
print(f"Saved: {maize_diag_path}")

print("\n" + "=" * 60)
print("DONE — Review the output above carefully")
print("=" * 60)
print("\nWhat to look at:")
print("  1. Do the 77 Wheat points classify as Wheat? (>90% expected)")
print("  2. Do the 3 Agroforestry points classify as Agroforestry?")
print("  3. The 19 Maize_Tall points — how many are:")
print("     - Spring_Maize in Rabi window? (mature April plants)")
print("     - Kharif_Maize in Kharif window? (newly sown / observed in 2025)")
print("     - Rice in Kharif window? (likely some were mislabelled)")