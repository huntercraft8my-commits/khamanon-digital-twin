# -*- coding: utf-8 -*-
# ================================================================
# Khamanon LULC Points Labeller
# Input:  data/khamanon_rabi_lulc_points.csv    (from GEE)
#         data/khamanon_kharif_lulc_points.csv  (from GEE)
# Output: data/lulc_rabi_map.csv
#         data/lulc_kharif_map.csv
# Purpose: Add class names and colours to sampled GEE points
#          No rasterio or GDAL required
# ================================================================

import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))

# ── CLASS DEFINITIONS ─────────────────────────────────────────
# GEE class codes from your Random Forest training
# Code 1=Wheat, 2=SpringMaize, 3=Rice, 4=KharifMaize, 5=Agroforestry

RABI_CLASSES = {
    1: ('Wheat',        '#fbbf24'),
    2: ('Spring Maize', '#16a34a'),
    5: ('Agroforestry', '#064e3b'),
}

KHARIF_CLASSES = {
    3: ('Rice',          '#22c55e'),
    4: ('Kharif Maize',  '#fb923c'),
    5: ('Agroforestry',  '#064e3b'),
}

def label_points(csv_path, class_map, season_label):
    """Read GEE sampled CSV, add class_name and color columns."""

    print(f"\nProcessing: {os.path.basename(csv_path)}")

    if not os.path.exists(csv_path):
        print(f"  ERROR — file not found: {csv_path}")
        print(f"  Download from Google Drive → GEE_Exports folder")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} points")
    print(f"  Columns: {list(df.columns)}")

    # Rename coordinate columns if needed
    for old, new in [('.geo', None), ('longitude', 'lon'), ('latitude', 'lat')]:
        if old in df.columns and new:
            df = df.rename(columns={old: new})

    # Keep only needed columns
    keep = ['lon' if 'lon' in df.columns else 'longitude',
            'lat' if 'lat' in df.columns else 'latitude',
            'class_code']
    df = df[[c for c in keep if c in df.columns]].copy()

    # Standardise column names
    df.columns = [c.replace('longitude', 'lon').replace('latitude', 'lat')
                  for c in df.columns]

    # Drop rows with invalid class codes
    valid_codes = list(class_map.keys())
    before = len(df)
    df = df[df['class_code'].isin(valid_codes)].copy()
    if before != len(df):
        print(f"  Dropped {before - len(df)} rows with unknown class codes")

    # Add class name and colour
    df['class_name'] = df['class_code'].map(
        {k: v[0] for k, v in class_map.items()}
    )
    df['color'] = df['class_code'].map(
        {k: v[1] for k, v in class_map.items()}
    )
    df['season'] = season_label

    # Round coordinates
    df['lat'] = df['lat'].round(6)
    df['lon'] = df['lon'].round(6)

    print(f"  Valid points: {len(df):,}")
    print(f"  Class breakdown:")
    for cls, grp in df.groupby('class_name'):
        pct = 100 * len(grp) / len(df)
        print(f"    {cls:<20s}: {len(grp):>6,} pts  ({pct:.1f}%)")

    return df


# ── MAIN ──────────────────────────────────────────────────────
print("=" * 60)
print("KHAMANON LULC POINTS LABELLER")
print("No rasterio required")
print("=" * 60)

# Process Rabi
rabi_in  = os.path.join(base, '..', 'data',
                         'khamanon_rabi_lulc_points.csv')
rabi_out = os.path.join(base, '..', 'data', 'lulc_rabi_map.csv')
rabi_df  = label_points(rabi_in, RABI_CLASSES, 'Rabi 2025-26')
if not rabi_df.empty:
    rabi_df.to_csv(rabi_out, index=False)
    print(f"\n  Saved: data/lulc_rabi_map.csv")

# Process Kharif
kharif_in  = os.path.join(base, '..', 'data',
                           'khamanon_kharif_lulc_points.csv')
kharif_out = os.path.join(base, '..', 'data', 'lulc_kharif_map.csv')
kharif_df  = label_points(kharif_in, KHARIF_CLASSES, 'Kharif 2025')
if not kharif_df.empty:
    kharif_df.to_csv(kharif_out, index=False)
    print(f"\n  Saved: data/lulc_kharif_map.csv")

# ── SUMMARY ───────────────────────────────────────────────────
print("\n" + "=" * 60)
if rabi_df.empty or kharif_df.empty:
    print("INCOMPLETE — download the missing CSV from Google Drive")
    print("Folder: GEE_Exports")
    print("Files needed:")
    print("  khamanon_rabi_lulc_points.csv")
    print("  khamanon_kharif_lulc_points.csv")
else:
    total = len(rabi_df) + len(kharif_df)
    print(f"DONE — {total:,} total map points ready")
    print(f"  Rabi:   {len(rabi_df):,} points")
    print(f"  Kharif: {len(kharif_df):,} points")
    print("\nRestart dashboard to see the LULC map in Tab 8")
print("=" * 60)