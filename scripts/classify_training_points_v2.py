# -*- coding: utf-8 -*-
# ================================================================
# Khamanon Training Points — Seasonal Classifier v2
# Input  : data/khamanon_trainingpts_multiindex_timeseries.csv
# Output : data/training_points_classified_v2.csv
#          data/training_points_maize_diagnostic_v2.csv
# Purpose: Properly separate Rice vs Kharif Maize using NDWI
#          flooding signal during late June — mid July
# ================================================================

import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.abspath(__file__))

# ── 1. LOAD MULTI-INDEX TIME SERIES ───────────────────────────
csv_path = os.path.join(base, '..', 'data',
                        'khamanon_trainingpts_multiindex_timeseries.csv')

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])

print("=" * 70)
print("KHAMANON TRAINING POINTS — SEASONAL CLASSIFIER v2")
print("With NDWI-based rice flooding detection")
print("=" * 70)
print(f"\nLoaded {len(df):,} records")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Indices available: NDVI, NDWI, MNDWI")
print(f"\nOriginal class distribution (raw record count):")
print(df['class_name'].value_counts())

# ── 2. CREATE POINT ID & CLEAN ────────────────────────────────
df['point_id'] = (df['longitude'].round(6).astype(str)
                  + '_'
                  + df['latitude'].round(6).astype(str))

df_clean = df.dropna(subset=['NDVI', 'NDWI']).copy()
print(f"\nValid records after cloud removal: {len(df_clean):,}")
print(f"Unique points: {df_clean['point_id'].nunique()}")

# ── 3. SEASONAL WINDOWS ────────────────────────────────────────
# Tuned to actual Punjab cropping calendar
# Source: your field observations + PAU Package of Practices

# Rabi 2025-26 windows (Wheat / Spring Maize)
RABI_PEAK_START   = '2026-02-01'   # Wheat peak vegetative
RABI_PEAK_END     = '2026-03-15'
SPRING_PEAK_START = '2026-03-25'   # Spring Maize peak (DISTINCT)
SPRING_PEAK_END   = '2026-05-05'

# Kharif 2025 windows (Rice / Kharif Maize)
KHARIF_PRE_START  = '2025-05-15'   # Pre-Kharif bare ground (post wheat)
KHARIF_PRE_END    = '2025-06-15'
FLOOD_START       = '2025-06-20'   # Rice transplanting flood window
FLOOD_END         = '2025-07-20'
KHARIF_PEAK_START = '2025-08-10'   # Rice canopy closes / maize peaks
KHARIF_PEAK_END   = '2025-09-25'
KHARIF_END_START  = '2025-10-10'   # Rice harvest period
KHARIF_END_END    = '2025-11-05'

def window_stats(point_data, col, start, end):
    """Return (max, mean, min, count) of an index within a date window."""
    s = point_data[(point_data['date'] >= start) &
                   (point_data['date'] <= end)][col]
    if len(s) == 0:
        return np.nan, np.nan, np.nan, 0
    return s.max(), s.mean(), s.min(), len(s)


print("\nComputing seasonal NDVI + NDWI metrics per point...")

records = []
for pid, grp in df_clean.groupby('point_id'):
    lon = grp['longitude'].iloc[0]
    lat = grp['latitude'].iloc[0]
    oc  = grp['class'].iloc[0]
    ocn = grp['class_name'].iloc[0]
    ol  = grp['landuse'].iloc[0]

    # ── Rabi: Wheat peak ──
    rabi_max, _, _, _    = window_stats(grp, 'NDVI', RABI_PEAK_START, RABI_PEAK_END)
    # ── Rabi: Spring maize peak ──
    spring_max, _, _, _  = window_stats(grp, 'NDVI', SPRING_PEAK_START, SPRING_PEAK_END)

    # ── Kharif: Pre-Kharif (May-June) NDVI — should be low for rice fields ──
    pre_max, pre_mean, _, _  = window_stats(grp, 'NDVI', KHARIF_PRE_START, KHARIF_PRE_END)

    # ── Kharif: Flooding window — NDWI and MNDWI HIGH means standing water ──
    flood_ndwi_max,  _, _, _  = window_stats(grp, 'NDWI',  FLOOD_START, FLOOD_END)
    flood_mndwi_max, _, _, _  = window_stats(grp, 'MNDWI', FLOOD_START, FLOOD_END)
    flood_ndvi_mean, _, _, _  = window_stats(grp, 'NDVI',  FLOOD_START, FLOOD_END)

    # ── Kharif: Peak (Aug-Sep) NDVI ──
    kharif_max, kharif_mean, _, _ = window_stats(grp, 'NDVI', KHARIF_PEAK_START, KHARIF_PEAK_END)

    # ── Kharif: End (Oct-Nov) NDVI — should drop sharply for rice harvest ──
    end_max, end_mean, end_min, _ = window_stats(grp, 'NDVI', KHARIF_END_START, KHARIF_END_END)

    # ── Year-round stability ──
    annual_mean = grp['NDVI'].mean()
    annual_std  = grp['NDVI'].std()
    annual_min  = grp['NDVI'].min()
    annual_max  = grp['NDVI'].max()

    records.append({
        'point_id'        : pid,
        'longitude'       : lon,
        'latitude'        : lat,
        'original_class'  : oc,
        'original_name'   : ocn,
        'original_landuse': ol,
        # Rabi
        'rabi_max'        : rabi_max,
        'spring_max'      : spring_max,
        # Pre-Kharif
        'prekharif_max'   : pre_max,
        'prekharif_mean'  : pre_mean,
        # Flood window — KEY for rice detection
        'flood_ndwi_max'  : flood_ndwi_max,
        'flood_mndwi_max' : flood_mndwi_max,
        'flood_ndvi_mean' : flood_ndvi_mean,
        # Kharif peak
        'kharif_max'      : kharif_max,
        'kharif_mean'     : kharif_mean,
        # End-Kharif
        'end_max'         : end_max,
        'end_mean'        : end_mean,
        # Year-round
        'annual_mean'     : annual_mean,
        'annual_std'      : annual_std,
        'annual_min'      : annual_min,
        'annual_max'      : annual_max,
    })

prof = pd.DataFrame(records)
print(f"Computed metrics for {len(prof)} unique points")

# ── 4. CLASSIFICATION RULES — Rabi 2025-26 ────────────────────
def classify_rabi(row):
    """Rabi 2025-26 class: Wheat / Spring_Maize / Agroforestry / Other"""

    # AGROFORESTRY — stable high NDVI year-round
    # Loosened std threshold (0.13 → 0.18) per v1 sanity check
    if (row['annual_mean'] >= 0.45
            and row['annual_std'] <= 0.18
            and row['annual_min'] >= 0.25):
        return 'Agroforestry'

    # SPRING MAIZE — unique high NDVI in late March / April
    # No other Punjab crop peaks here
    if (pd.notna(row['spring_max'])
            and row['spring_max'] >= 0.65
            and (pd.isna(row['rabi_max'])
                 or row['spring_max'] > row['rabi_max'])):
        return 'Spring_Maize'

    # WHEAT — peaked in Feb-March, dropped by April-May
    if (pd.notna(row['rabi_max'])
            and row['rabi_max'] >= 0.55
            and (pd.isna(row['spring_max'])
                 or row['spring_max'] < row['rabi_max'] - 0.08)):
        return 'Wheat'

    return 'Other'


# ── 5. CLASSIFICATION RULES — Kharif 2025 (THE KEY FIX) ───────
def classify_kharif(row):
    """Kharif 2025 class: Rice / Kharif_Maize / Agroforestry / Other

    The CRITICAL difference vs v1: now uses NDWI flooding signal.
    Rice fields are FLOODED in late June - mid July.
    Maize fields are NEVER flooded.
    """

    # AGROFORESTRY first (same rule as Rabi)
    if (row['annual_mean'] >= 0.45
            and row['annual_std'] <= 0.18
            and row['annual_min'] >= 0.25):
        return 'Agroforestry'

    # RICE — three independent evidence requirements
    # (a) FLOODING: NDWI > 0 during late June - mid July (standing water)
    #     OR MNDWI > -0.10 (water using SWIR — even better in slightly turbid water)
    # (b) HIGH peak NDVI in Aug-Sep (canopy closed)
    # (c) LOW NDVI in pre-Kharif window (was harvested wheat / bare)
    flood_detected = (
        (pd.notna(row['flood_ndwi_max'])  and row['flood_ndwi_max']  > -0.05)
        or (pd.notna(row['flood_mndwi_max']) and row['flood_mndwi_max'] > -0.15)
    )
    has_canopy = (pd.notna(row['kharif_max']) and row['kharif_max'] >= 0.55)
    was_bare   = (pd.isna(row['prekharif_max']) or row['prekharif_max'] <= 0.35)

    if flood_detected and has_canopy and was_bare:
        return 'Rice'

    # KHARIF MAIZE — high Kharif peak BUT no flooding
    # Maize planted early to mid-June, green by July (no bare phase)
    if (has_canopy
            and not flood_detected
            and pd.notna(row['prekharif_max'])
            and row['prekharif_max'] > 0.30):
        return 'Kharif_Maize'

    # Edge case: high peak but no clear flood + no early growth
    # Could be late-sown maize or rice that missed flood capture
    # Conservative: call it Rice if pre-Kharif was bare (more likely in Punjab)
    if has_canopy and was_bare:
        return 'Rice_Probable'

    return 'Other'


prof['class_rabi_2025_26'] = prof.apply(classify_rabi,  axis=1)
prof['class_kharif_2025']  = prof.apply(classify_kharif, axis=1)

# ── 6. SUMMARY ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RABI 2025-26 CLASSIFICATION (v2)")
print("=" * 70)
print(prof['class_rabi_2025_26'].value_counts().to_string())

print("\n" + "=" * 70)
print("KHARIF 2025 CLASSIFICATION (v2)")
print("=" * 70)
print(prof['class_kharif_2025'].value_counts().to_string())

# ── 7. DEEP DIVE — 19 maize points ────────────────────────────
maize_pts = prof[prof['original_name'] == 'Maize_Tall'].copy()
print("\n" + "=" * 70)
print(f"DEEP DIVE — Original Maize_Tall points (n = {len(maize_pts)})")
print("=" * 70)

cols_to_show = ['longitude', 'latitude',
                'rabi_max', 'spring_max',
                'prekharif_max', 'flood_ndwi_max', 'flood_mndwi_max',
                'kharif_max',
                'class_rabi_2025_26', 'class_kharif_2025']

view = maize_pts[cols_to_show].copy().round(3).sort_values('spring_max', ascending=False)
print("\n" + view.to_string(index=False))

print("\nMaize_Tall points — cross-tab Rabi vs Kharif:")
print(maize_pts.groupby(['class_rabi_2025_26', 'class_kharif_2025']).size().to_string())

# ── 8. DEEP DIVE — Wheat points ───────────────────────────────
wheat_pts = prof[prof['original_name'] == 'Wheat'].copy()
print("\n" + "=" * 70)
print(f"WHEAT points sanity check (n = {len(wheat_pts)})")
print("=" * 70)

wheat_rabi = wheat_pts['class_rabi_2025_26'].value_counts()
print("\nRabi class assignment:")
print(wheat_rabi.to_string())
correct_pct = 100 * wheat_rabi.get('Wheat', 0) / len(wheat_pts)
print(f"\n  → {wheat_rabi.get('Wheat',0)} / {len(wheat_pts)} "
      f"correctly classified as Wheat ({correct_pct:.1f}%)")

print("\nWheat points — Kharif class (these fields were rice/maize in 2025):")
print(wheat_pts['class_kharif_2025'].value_counts().to_string())

# ── 9. DEEP DIVE — Agroforestry ───────────────────────────────
agr_pts = prof[prof['original_name'] == 'Agroforestry'].copy()
print("\n" + "=" * 70)
print(f"AGROFORESTRY points (n = {len(agr_pts)})")
print("=" * 70)

for _, r in agr_pts.iterrows():
    print(f"\n  Point ({r['longitude']:.4f}, {r['latitude']:.4f}):")
    print(f"    annual_mean = {r['annual_mean']:.3f}")
    print(f"    annual_std  = {r['annual_std']:.3f}")
    print(f"    annual_min  = {r['annual_min']:.3f}")
    print(f"    → Rabi:   {r['class_rabi_2025_26']}")
    print(f"    → Kharif: {r['class_kharif_2025']}")

# ── 10. DEEP DIVE — Others ────────────────────────────────────
oth_pts = prof[prof['original_name'] == 'Others'].copy()
print("\n" + "=" * 70)
print(f"OTHERS points (n = {len(oth_pts)}) — heterogeneous crops")
print("=" * 70)
print("\nRabi distribution:")
print(oth_pts['class_rabi_2025_26'].value_counts().to_string())
print("\nKharif distribution:")
print(oth_pts['class_kharif_2025'].value_counts().to_string())

# ── 11. SAVE OUTPUTS ──────────────────────────────────────────
out_path = os.path.join(base, '..', 'data', 'training_points_classified_v2.csv')
prof.to_csv(out_path, index=False)
print(f"\n\nSaved: {out_path}")

maize_out = os.path.join(base, '..', 'data', 'training_points_maize_diagnostic_v2.csv')
maize_pts.to_csv(maize_out, index=False)
print(f"Saved: {maize_out}")

print("\n" + "=" * 70)
print("DONE — Review carefully")
print("=" * 70)
print("\nWhat to check:")
print("  1. Kharif Rice count — should be MUCH higher than v1's 16")
print("     Punjab Kharif = 70-85% rice, so expect majority Rice")
print("  2. Wheat sanity — should be 90%+ accuracy in Rabi")
print("  3. Maize_Tall split — Spring_Maize vs Kharif_Maize vs both?")
print("  4. Agroforestry — should now classify correctly (3/3)")