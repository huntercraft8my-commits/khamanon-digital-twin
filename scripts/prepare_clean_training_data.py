# -*- coding: utf-8 -*-
# ================================================================
# Khamanon — Prepare Clean Training Data for GEE
# Input  : data/training_points_classified_v2.csv
# Output : data/training_points_clean_for_gee.csv
#          data/training_points_summary.txt
#
# Purpose: Merge Rice_Probable into Rice. Build a clean dataset
#          ready to upload back to GEE as a seasonal training asset.
# ================================================================

import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))

# ── 1. LOAD CLASSIFIED POINTS ──────────────────────────────────
src = os.path.join(base, '..', 'data',
                   'training_points_classified_v2.csv')
df = pd.read_csv(src)

print("=" * 70)
print("KHAMANON — CLEAN TRAINING DATA FOR GEE LULC CLASSIFICATION")
print("=" * 70)
print(f"\nLoaded {len(df)} classified points")

# ── 2. MERGE RICE_PROBABLE → RICE ──────────────────────────────
df['class_kharif_2025'] = df['class_kharif_2025'].replace(
    'Rice_Probable', 'Rice'
)

print("\nAfter merging Rice_Probable into Rice:")
print(df['class_kharif_2025'].value_counts().to_string())

# ── 3. ASSIGN NUMERIC CLASS CODES (for GEE training) ──────────
# Class codes used for Rabi & Kharif (consistent across seasons)
#   1 = Wheat
#   2 = Spring_Maize
#   3 = Rice
#   4 = Kharif_Maize
#   5 = Agroforestry
#   6 = Other

CLASS_CODE = {
    'Wheat':         1,
    'Spring_Maize':  2,
    'Rice':          3,
    'Kharif_Maize':  4,
    'Agroforestry':  5,
    'Other':         6
}

df['rabi_class_code']   = df['class_rabi_2025_26'].map(CLASS_CODE)
df['kharif_class_code'] = df['class_kharif_2025'].map(CLASS_CODE)

# ── 4. BUILD CLEAN EXPORT FOR GEE ──────────────────────────────
# We only need: longitude, latitude, rabi class, kharif class
out = df[[
    'longitude', 'latitude',
    'original_name',
    'class_rabi_2025_26',  'rabi_class_code',
    'class_kharif_2025',   'kharif_class_code'
]].copy()

# Rename for clarity (no spaces, lowercase, GEE-friendly)
out.columns = [
    'longitude', 'latitude',
    'original_label',
    'rabi_class',   'rabi_code',
    'kharif_class', 'kharif_code'
]

out_path = os.path.join(base, '..', 'data',
                        'training_points_clean_for_gee.csv')
out.to_csv(out_path, index=False)

print(f"\nSaved clean training data:")
print(f"  {out_path}")

# ── 5. SUMMARY REPORT ──────────────────────────────────────────
summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("KHAMANON TRAINING POINTS — CLEAN VERSION FOR GEE")
summary_lines.append("=" * 70)
summary_lines.append("")
summary_lines.append(f"Total points: {len(out)}")
summary_lines.append("")
summary_lines.append("Class code legend:")
for name, code in CLASS_CODE.items():
    summary_lines.append(f"  {code} = {name}")
summary_lines.append("")
summary_lines.append("-" * 70)
summary_lines.append("RABI 2025-26 (Wheat / Spring Maize / Agroforestry / Other)")
summary_lines.append("-" * 70)
rabi_summary = out['rabi_class'].value_counts()
for cls, n in rabi_summary.items():
    pct = 100 * n / len(out)
    summary_lines.append(f"  {cls:<15s}: {n:>3} points  ({pct:>5.1f}%)")

summary_lines.append("")
summary_lines.append("-" * 70)
summary_lines.append("KHARIF 2025 (Rice / Kharif Maize / Agroforestry / Other)")
summary_lines.append("-" * 70)
kharif_summary = out['kharif_class'].value_counts()
for cls, n in kharif_summary.items():
    pct = 100 * n / len(out)
    summary_lines.append(f"  {cls:<15s}: {n:>3} points  ({pct:>5.1f}%)")

summary_lines.append("")
summary_lines.append("-" * 70)
summary_lines.append("CROSS-TAB — how each point changed across seasons")
summary_lines.append("-" * 70)
ct = pd.crosstab(out['rabi_class'], out['kharif_class'],
                 margins=True, margins_name='Total')
summary_lines.append(ct.to_string())

summary_lines.append("")
summary_lines.append("=" * 70)
summary_lines.append("READY TO UPLOAD TO GEE AS:")
summary_lines.append("  training_points_clean_for_gee.csv")
summary_lines.append("")
summary_lines.append("Use this as the training asset for two GEE scripts:")
summary_lines.append("  • Khamanon_Rabi_2025_26_LULC   → uses 'rabi_code' as label")
summary_lines.append("  • Khamanon_Kharif_2025_LULC    → uses 'kharif_code' as label")
summary_lines.append("=" * 70)

# Print and save
for line in summary_lines:
    print(line)

with open(os.path.join(base, '..', 'data',
                       'training_points_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print(f"\nSaved summary report:")
print(f"  data/training_points_summary.txt")