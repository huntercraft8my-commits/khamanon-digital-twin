# ============================================
# DSS - KHAMANON BLOCK
# Script 7: Crop Monitoring — Real GEE Data
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

print("=" * 55)
print("  PHASE 4 — CROP MONITORING")
print("  Real Sentinel-2 NDVI — Khamanon Block")
print("=" * 55)

# ============================================
# LOAD REAL NDVI DATA FROM GEE
# ============================================

df = pd.read_csv(
    os.path.join('..', 'data', 'ndvi_monthly_zones.csv')
)

print(f"\nRecords loaded: {len(df)}")

# ============================================
# RESHAPE DATA
# From long format to wide format
# Rows = months, Columns = zones
# ============================================

df_wide = df.pivot_table(
    index   = 'month',
    columns = 'zone',
    values  = 'mean'
).reset_index()

# Define correct month order
month_order = [
    'Jan-2025','Feb-2025','Mar-2025',
    'Apr-2025','May-2025','Jun-2025',
    'Jul-2025','Aug-2025','Sep-2025',
    'Oct-2025','Nov-2025','Dec-2025',
    'Jan-2026','Feb-2026','Mar-2026'
]

df_wide['month'] = pd.Categorical(
    df_wide['month'],
    categories = month_order,
    ordered    = True
)
df_wide = df_wide.sort_values('month').reset_index(drop=True)

print("\nNDVI time series (reshaped):")
print(df_wide.to_string())

# Also get image counts per month
img_counts = df.groupby('month')['image_count'].first()

zones = [
    'Healthy Cropland (North)',
    'Stressed Cropland (Central)',
    'Peri-urban SE',
    'Vegetation West'
]

zone_colors = {
    'Healthy Cropland (North)'    : 'green',
    'Stressed Cropland (Central)' : 'orange',
    'Peri-urban SE'               : 'red',
    'Vegetation West'             : 'blue'
}

zone_markers = {
    'Healthy Cropland (North)'    : 'o',
    'Stressed Cropland (Central)' : 's',
    'Peri-urban SE'               : '^',
    'Vegetation West'             : 'D'
}

# ============================================
# PLOT 1: NDVI TIME SERIES — ALL ZONES
# ============================================

fig, ax = plt.subplots(figsize=(14, 7))

x = range(len(df_wide))

for zone in zones:
    if zone in df_wide.columns:
        ax.plot(
            x,
            df_wide[zone],
            color   = zone_colors[zone],
            marker  = zone_markers[zone],
            linewidth = 2.5,
            markersize = 8,
            label   = zone
        )

# Season bands
ax.axvspan(-0.5, 2.5,
           alpha=0.10, color='gold',
           label='Rabi 2024-25 (Wheat)')
ax.axvspan(4.5, 5.5,
           alpha=0.10, color='gray',
           label='Summer Fallow')
ax.axvspan(5.5, 9.5,
           alpha=0.10, color='lightgreen',
           label='Kharif 2025 (Rice)')
ax.axvspan(11.5, 14.5,
           alpha=0.10, color='gold')

# Cloud warning band
ax.axvspan(6.5, 7.5,
           alpha=0.15, color='lightblue',
           label='Heavy cloud (few images)')

# Stress threshold
ax.axhline(y=0.40, color='red',
           linestyle='--', linewidth=1.5,
           alpha=0.8, label='Stress threshold (0.40)')

ax.set_xticks(list(x))
ax.set_xticklabels(
    df_wide['month'].tolist(),
    rotation=45, ha='right', fontsize=10
)
ax.set_ylabel('NDVI', fontsize=12)
ax.set_xlabel('Month', fontsize=12)
ax.set_title(
    'Crop Growth Monitoring — Khamanon Block\n'
    'Real Sentinel-2 NDVI Time Series (2025–2026)',
    fontsize=13, fontweight='bold'
)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right', fontsize=9,
          framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.4)

# Add image count annotation
for i, month in enumerate(df_wide['month']):
    if month in img_counts.index:
        n = int(img_counts[month])
        ax.text(i, 0.03, f'n={n}',
                ha='center', fontsize=7,
                color='gray', rotation=90)

plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps',
                 'real_ndvi_timeseries.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("\nSaved: maps/real_ndvi_timeseries.png")

# ============================================
# PLOT 2: CROP STRESS DETECTION TABLE
# Month by month stress alert
# ============================================

stress_threshold = 0.40

fig, ax = plt.subplots(figsize=(14, 6))

table_data = []
col_labels = ['Month', 'Images'] + zones

for _, row in df_wide.iterrows():
    month = row['month']
    n_img = int(img_counts.get(month, 0))
    r = [month, n_img]
    for zone in zones:
        if zone in row.index:
            val = row[zone]
            r.append(f"{val:.3f}")
        else:
            r.append('N/A')
    table_data.append(r)

table = ax.table(
    cellText    = table_data,
    colLabels   = col_labels,
    cellLoc     = 'center',
    loc         = 'center'
)

# Color cells by stress level
for i, row_data in enumerate(table_data):
    for j, zone in enumerate(zones):
        cell_idx = j + 2
        try:
            val = float(row_data[cell_idx])
            if val < stress_threshold:
                table[i+1, cell_idx].set_facecolor('#ffcccc')
            elif val < 0.55:
                table[i+1, cell_idx].set_facecolor('#fff3cc')
            else:
                table[i+1, cell_idx].set_facecolor('#ccffcc')
        except:
            pass

# Header styling
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white',
                                fontweight='bold')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)

ax.axis('off')
ax.set_title(
    'Monthly NDVI Stress Detection — Khamanon Block\n'
    'Red = Stress (NDVI<0.40) | Yellow = Moderate | '
    'Green = Healthy',
    fontsize=12, fontweight='bold', pad=20
)

plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps',
                 'real_stress_table.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/real_stress_table.png")

# ============================================
# SAVE PROCESSED NDVI FOR DASHBOARD
# ============================================

ndvi_save = df_wide.copy()
ndvi_save.to_csv(
    os.path.join('..', 'data',
                 'ndvi_processed.csv'),
    index=False
)
print("Saved: data/ndvi_processed.csv")

# ============================================
# PRINT KEY FINDINGS
# ============================================

print("\n" + "=" * 55)
print("  PHASE 4 COMPLETE")
print("=" * 55)

print("\nKey Findings from Real Sentinel-2 Data:")
print("-" * 55)

for zone in zones:
    if zone in df_wide.columns:
        peak = df_wide[zone].max()
        peak_month = df_wide.loc[
            df_wide[zone].idxmax(), 'month'
        ]
        low  = df_wide[zone].min()
        low_month = df_wide.loc[
            df_wide[zone].idxmin(), 'month'
        ]
        stress_months = df_wide[
            df_wide[zone] < stress_threshold
        ]['month'].tolist()

        print(f"\n  {zone}:")
        print(f"    Peak NDVI : {peak:.3f} ({peak_month})")
        print(f"    Low  NDVI : {low:.3f} ({low_month})")
        print(f"    Stress months: {stress_months}")

print("\nNext: Phase 5 — Real Data Dashboard")