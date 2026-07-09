# ============================================
# DSS - KHAMANON BLOCK
# Script 12: PAU Recommendation Engine
# Source: PAU Package of Practices Kharif/Rabi 2025
#         Punjab Soil Health Card
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

print("=" * 55)
print("  PRIORITY 2 — PAU RECOMMENDATION ENGINE")
print("  Khamanon Block — Real Data")
print("=" * 55)

base = os.path.dirname(os.path.abspath(__file__))

# ============================================
# PAU ADVISORY RULE ENGINE
# Direct implementation of Section 12
# Master Rule Table from knowledge base
# ============================================

def generate_pau_advisories(
    pH, OC, EC, K2O, available_P,
    available_N, CEC, bulk_density, CaCO3,
    ndvi_mean, season='rabi'
):
    """
    Generate PAU-based soil advisories.
    Source: PAU Package of Practices 2025
            Punjab Soil Health Card
    Returns list of advisory dicts with:
    rule_id, type, severity, message, action
    """

    advisories = []

    # ------------------------------------------
    # R01 — Nitrogen — Low OC
    # ------------------------------------------
    if OC < 0.40:
        advisories.append({
            'rule_id' : 'R01',
            'type'    : 'Nitrogen Management',
            'severity': 'WARNING',
            'message' : '⚠️ Low Organic Carbon detected.',
            'action'  : (
                'Increase nitrogen dose by 25% over '
                'standard recommendation. '
                'Apply FYM (50-100 kg/acre) and green '
                'manures before sowing to improve '
                'soil health. OC detected: '
                f'{OC:.3f}% (threshold: 0.40%)'
            )
        })

    # R02 — Nitrogen — High OC
    elif OC > 0.75:
        advisories.append({
            'rule_id' : 'R02',
            'type'    : 'Nitrogen Management',
            'severity': 'INFO',
            'message' : '✅ High Organic Carbon detected.',
            'action'  : (
                'Reduce nitrogen fertilizer by 25% below '
                'standard dose. Soil is nitrogen-rich. '
                'Avoid over-fertilization to prevent '
                'lodging and waterway contamination. '
                f'OC detected: {OC:.3f}%'
            )
        })

    # ------------------------------------------
    # R03 — Phosphorus — Low P
    # ------------------------------------------
    if available_P < 5:
        advisories.append({
            'rule_id' : 'R03',
            'type'    : 'Phosphorus Management',
            'severity': 'WARNING',
            'message' : '⚠️ Low Available Phosphorus.',
            'action'  : (
                'Apply 25% more than standard phosphatic '
                'fertilizer dose. Drill full dose at sowing '
                'as basal — never split phosphorus. '
                f'P detected: {available_P:.1f} kg/ha '
                '(threshold: 5 kg/acre)'
            )
        })

    # R04 — Phosphorus — Very High P
    elif available_P > 20:
        advisories.append({
            'rule_id' : 'R04',
            'type'    : 'Phosphorus Management',
            'severity': 'INFO',
            'message' : '✅ Very High Available Phosphorus.',
            'action'  : (
                'Omit phosphatic fertilizer for 2-3 years. '
                'Retest soil before resuming application. '
                'Kharif crops after wheat with full P dose '
                'do not need additional phosphorus (R18). '
                f'P detected: {available_P:.1f} kg/ha'
            )
        })

    # R05 — Combined P × OC — Both High
    if OC > 0.60 and available_P > 9:
        advisories.append({
            'rule_id' : 'R05',
            'type'    : 'Phosphorus Management',
            'severity': 'INFO',
            'message' : '✅ No phosphorus application required.',
            'action'  : (
                'Both OC and available P are above threshold. '
                'Omit phosphatic fertilizer for this season. '
                f'OC: {OC:.3f}%, P: {available_P:.1f} kg/ha'
            )
        })

    # ------------------------------------------
    # R06 — Potassium Deficient
    # ------------------------------------------
    if K2O < 55:
        advisories.append({
            'rule_id' : 'R06',
            'type'    : 'Potassium Management',
            'severity': 'WARNING',
            'message' : '⚠️ Potassium Deficiency Detected.',
            'action'  : (
                'Apply Muriate of Potash (MOP) as per '
                'crop recommendation. Khamanon Block '
                'falls in Fatehgarh Sahib — verify with '
                'local soil testing lab. '
                'High-risk districts: Gurdaspur, '
                'Hoshiarpur, Nawanshahar, Jalandhar, Ropar. '
                f'K₂O detected: {K2O:.1f} kg/ha '
                '(threshold: 55 kg/acre)'
            )
        })

    # ------------------------------------------
    # R11 — Sodic / Alkali Soil
    # ------------------------------------------
    if pH > 9.3:
        advisories.append({
            'rule_id' : 'R11',
            'type'    : 'Salt-Affected Soil',
            'severity': 'CRITICAL',
            'message' : '🔴 Sodic Soil Alert.',
            'action'  : (
                'Apply gypsum on soil-test basis + organic '
                'amendments. Increase N dose by +25% above '
                'standard. Apply zinc sulphate at higher '
                'rates. DO NOT apply gypsum to saline soils. '
                'Avoid saline-sensitive crops. '
                f'pH detected: {pH:.2f} (sodic threshold: 9.3)'
            )
        })

    # pH Warning — approaching sodic
    elif pH > 8.5:
        advisories.append({
            'rule_id' : 'R11b',
            'type'    : 'Soil pH',
            'severity': 'WARNING',
            'message' : '⚠️ High pH — Approaching Sodic Range.',
            'action'  : (
                'pH is approaching problematic alkalinity. '
                'Monitor soil regularly. Apply organic manures '
                'to buffer pH. Watch for micronutrient '
                'deficiencies (Zn, Fe, Mn). '
                f'pH detected: {pH:.2f}'
            )
        })

    # ------------------------------------------
    # R12 — Saline Soil
    # ------------------------------------------
    if EC > 0.8:
        advisories.append({
            'rule_id' : 'R12',
            'type'    : 'Salt-Affected Soil',
            'severity': 'WARNING',
            'message' : '⚠️ Saline Soil Detected.',
            'action'  : (
                'Apply +25% nitrogen fertilizer above '
                'standard dose. Add organic manures and '
                'green manures. Ensure adequate drainage. '
                'DO NOT apply gypsum to saline soils — '
                'gypsum is only for sodic/alkali soils. '
                f'EC detected: {EC:.3f} dS/m '
                '(threshold: 0.8 mmhos/cm)'
            )
        })

    # ------------------------------------------
    # R17 — FYM General Recommendation
    # ------------------------------------------
    if OC < 0.50:
        advisories.append({
            'rule_id' : 'R17',
            'type'    : 'Organic Matter',
            'severity': 'INFO',
            'message' : 'ℹ️ Low Organic Matter — FYM Recommended.',
            'action'  : (
                'Apply 50-100 kg/acre well-rotted Farmyard '
                'Manure (FYM) before sowing. Combine with '
                'chemical fertilizers for best soil health. '
                'Consider incorporating rice straw/crop '
                'residue to improve OC over seasons. '
                f'OC detected: {OC:.3f}%'
            )
        })

    # ------------------------------------------
    # R20 — Crop Residue Retention
    # ------------------------------------------
    if OC < 0.40:
        advisories.append({
            'rule_id' : 'R20',
            'type'    : 'Organic Matter',
            'severity': 'INFO',
            'message' : 'ℹ️ Retain Crop Residue.',
            'action'  : (
                'Incorporate paddy and wheat residue '
                'continuously in rice-wheat system. '
                'Improves soil OC, water retention, and '
                'reduces micronutrient deficiencies. '
                'Target: wheat yield 22.68 q/acre, '
                'system productivity 50.88 q/acre.'
            )
        })

    # ------------------------------------------
    # R14 / R15 — NDVI Crop Stress
    # ------------------------------------------
    if ndvi_mean < 0.20:
        advisories.append({
            'rule_id' : 'R15',
            'type'    : 'Crop Stress',
            'severity': 'CRITICAL',
            'message' : '🔴 Severe Crop Stress Detected.',
            'action'  : (
                'Immediate field inspection required. '
                'Evaluate irrigation status, pest/disease '
                'pressure, and soil health urgently. '
                'Consider emergency irrigation if moisture '
                'deficit is the cause. '
                f'NDVI: {ndvi_mean:.3f} (critical: <0.20)'
            )
        })
    elif ndvi_mean < 0.35:
        advisories.append({
            'rule_id' : 'R14',
            'type'    : 'Crop Stress',
            'severity': 'WARNING',
            'message' : '⚠️ Crop Stress Detected via Satellite.',
            'action'  : (
                'Probable causes: moisture deficit, nutrient '
                'deficiency, or pest pressure. Check field '
                'conditions. Consider irrigation and '
                'supplementary soil testing. '
                f'NDVI: {ndvi_mean:.3f} (threshold: 0.35)'
            )
        })

    # ------------------------------------------
    # High CaCO3 Advisory
    # ------------------------------------------
    if CaCO3 > 2.0:
        advisories.append({
            'rule_id' : 'R16b',
            'type'    : 'Calcareous Soil',
            'severity': 'WARNING',
            'message' : '⚠️ High Calcium Carbonate.',
            'action'  : (
                'Apply organic manures at 8 t/acre FYM '
                'or green manure or wheat straw at '
                '2.5 t/acre per year. '
                'High CaCO3 fixes phosphorus and '
                'micronutrients — watch for Zn and Fe '
                'deficiency. '
                f'CaCO3 detected: {CaCO3:.2f}%'
            )
        })

    # ------------------------------------------
    # Bulk Density Advisory
    # ------------------------------------------
    if bulk_density > 260:
        advisories.append({
            'rule_id' : 'BD01',
            'type'    : 'Soil Structure',
            'severity': 'WARNING',
            'message' : '⚠️ High Bulk Density — Soil Compaction.',
            'action'  : (
                'Bulk density above 260 g/L indicates '
                'compaction from heavy machinery or '
                'tillage pan. Practice deep tillage '
                '(sub-soiling) once every 3-4 years. '
                'Add organic matter to improve '
                'soil structure. '
                f'Bulk density: {bulk_density:.1f} g/L'
            )
        })

    # ------------------------------------------
    # No advisories = healthy soil
    # ------------------------------------------
    if len(advisories) == 0:
        advisories.append({
            'rule_id' : 'OK',
            'type'    : 'General',
            'severity': 'OK',
            'message' : '✅ Soil Health Within Normal Range.',
            'action'  : (
                'All monitored parameters are within '
                'acceptable limits as per PAU recommendations. '
                'Continue standard agronomic practices. '
                'Retest soil annually.'
            )
        })

    return advisories


# ============================================
# APPLY ENGINE TO ALL 208 SAMPLE POINTS
# ============================================

print("\nLoading validated soil data...")

soil = pd.read_csv(
    os.path.join(base, '..', 'data',
                 'soil_data_validated.csv')
)

# Get current NDVI from last update
status_path = os.path.join(
    base, '..', 'data', 'last_update.json'
)
if os.path.exists(status_path):
    with open(status_path, 'r') as f:
        status = json.load(f)
    ndvi_current = status.get('ndvi_mean', 0.35)
else:
    ndvi_current = 0.35

print(f"Current NDVI    : {ndvi_current:.3f}")
print(f"Sample points   : {len(soil)}")

# Generate advisories for every sample point
print("\nGenerating PAU advisories for all points...")

all_point_advisories = []

for idx, row in soil.iterrows():
    advisories = generate_pau_advisories(
        pH            = row['pH'],
        OC            = row['OC'],
        EC            = row['EC'],
        K2O           = row['K2O'],
        available_P   = row['available_P'],
        available_N   = row['available_N'],
        CEC           = row['CEC'],
        bulk_density  = row['bulk_density'],
        CaCO3         = row['CaCO3'],
        ndvi_mean     = ndvi_current,
        season        = 'rabi'
    )

    for adv in advisories:
        all_point_advisories.append({
            'sample_id' : row['sample_id'],
            'latitude'  : row['latitude'],
            'longitude' : row['longitude'],
            'easting'   : row['easting_utm'],
            'northing'  : row['northing_utm'],
            **adv
        })

adv_df = pd.DataFrame(all_point_advisories)

print(f"Total advisories generated: {len(adv_df)}")
print(f"Unique rule IDs triggered:")
rule_counts = adv_df['rule_id'].value_counts()
for rule, count in rule_counts.items():
    print(f"  {rule:<8}: {count} points")

# ============================================
# BLOCK-LEVEL SUMMARY
# Generate one advisory for the whole block
# based on median soil values
# ============================================

print("\nGenerating block-level advisory summary...")

block_advisories = generate_pau_advisories(
    pH           = soil['pH'].median(),
    OC           = soil['OC'].median(),
    EC           = soil['EC'].median(),
    K2O          = soil['K2O'].median(),
    available_P  = soil['available_P'].median(),
    available_N  = soil['available_N'].median(),
    CEC          = soil['CEC'].median(),
    bulk_density = soil['bulk_density'].median(),
    CaCO3        = soil['CaCO3'].median(),
    ndvi_mean    = ndvi_current,
    season       = 'rabi'
)

print(f"\nBlock-level advisories ({len(block_advisories)}):")
for a in block_advisories:
    print(f"  [{a['severity']}] {a['rule_id']} — "
          f"{a['message']}")

# ============================================
# PLOT 1: Advisory Distribution Map
# Shows which rules fire at which locations
# ============================================

print("\nGenerating advisory maps...")

severity_colors = {
    'CRITICAL': '#e74c3c',
    'WARNING' : '#f39c12',
    'INFO'    : '#3498db',
    'OK'      : '#2ecc71'
}

# Most severe advisory per point
def get_severity_rank(sev):
    ranks = {'CRITICAL':4,'WARNING':3,
             'INFO':2,'OK':1}
    return ranks.get(sev, 0)

point_max_severity = (
    adv_df.groupby('sample_id')
    .apply(lambda g: g.loc[
        g['severity'].map(get_severity_rank).idxmax()
    ])
    .reset_index(drop=True)
)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Map of max severity per point
for sev, color in severity_colors.items():
    mask = point_max_severity['severity'] == sev
    if mask.sum() > 0:
        axes[0].scatter(
            point_max_severity.loc[mask, 'longitude'],
            point_max_severity.loc[mask, 'latitude'],
            c=color, s=60, alpha=0.8,
            label=f'{sev} ({mask.sum()})',
            edgecolors='gray', linewidth=0.3
        )

axes[0].set_xlabel('Longitude', fontsize=11)
axes[0].set_ylabel('Latitude', fontsize=11)
axes[0].set_title(
    'PAU Advisory Severity — Khamanon Block\n'
    '208 cLHS Sample Points',
    fontsize=12, fontweight='bold'
)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Right: Rule frequency bar chart
rule_counts_plot = rule_counts.head(10)
colors_bar = [
    '#e74c3c' if r in ['R11','R15','BD01']
    else '#f39c12' if r in ['R01','R06',
                            'R12','R11b',
                            'R14','R16b']
    else '#3498db'
    for r in rule_counts_plot.index
]

axes[1].barh(
    rule_counts_plot.index[::-1],
    rule_counts_plot.values[::-1],
    color=colors_bar[::-1],
    alpha=0.85,
    edgecolor='black', linewidth=0.5
)
axes[1].set_xlabel('Number of Points', fontsize=11)
axes[1].set_title(
    'PAU Advisory Rule Frequency\n'
    'Khamanon Block',
    fontsize=12, fontweight='bold'
)
axes[1].grid(True, alpha=0.3, axis='x')

for i, (rule, count) in enumerate(
        zip(rule_counts_plot.index[::-1],
            rule_counts_plot.values[::-1])):
    axes[1].text(
        count + 1, i, str(count),
        va='center', fontsize=10
    )

plt.suptitle(
    'PAU Soil Advisory Analysis — Khamanon Block\n'
    'Source: PAU Package of Practices Kharif/Rabi 2025',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(base, '..', 'maps',
                 'pau_advisory_map.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/pau_advisory_map.png")

# ============================================
# PLOT 2: Block Advisory Summary Card
# ============================================

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

y_start = 0.95
line_h  = 0.07

sev_colors_txt = {
    'CRITICAL': '#e74c3c',
    'WARNING' : '#e67e22',
    'INFO'    : '#2980b9',
    'OK'      : '#27ae60'
}

ax.text(
    0.5, y_start + 0.03,
    'KHAMANON BLOCK — PAU SOIL HEALTH ADVISORY',
    transform=ax.transAxes,
    fontsize=14, fontweight='bold',
    ha='center', va='top',
    color='#2c3e50'
)
ax.text(
    0.5, y_start - 0.03,
    'Source: PAU Package of Practices 2025 | '
    'Punjab Soil Health Card | '
    f'NDVI: {ndvi_current:.3f}',
    transform=ax.transAxes,
    fontsize=9, ha='center', va='top',
    color='#7f8c8d'
)

y = y_start - 0.10

for i, adv in enumerate(block_advisories):
    color = sev_colors_txt.get(
        adv['severity'], '#2c3e50'
    )

    ax.text(
        0.02, y,
        f"[{adv['rule_id']}] {adv['message']}",
        transform=ax.transAxes,
        fontsize=11, fontweight='bold',
        va='top', color=color
    )
    y -= 0.04

    # Wrap action text
    action = adv['action']
    words  = action.split()
    line   = ''
    lines  = []
    for w in words:
        if len(line + w) < 100:
            line += w + ' '
        else:
            lines.append(line.strip())
            line = w + ' '
    if line:
        lines.append(line.strip())

    for l in lines[:3]:
        ax.text(
            0.04, y,
            l,
            transform=ax.transAxes,
            fontsize=9, va='top',
            color='#2c3e50'
        )
        y -= 0.033

    y -= 0.01

    ax.axhline(
        y=y + 0.005,
        xmin=0.02, xmax=0.98,
        color='#ecf0f1', linewidth=1
    )

    if y < 0.05:
        break

plt.tight_layout()
plt.savefig(
    os.path.join(base, '..', 'maps',
                 'pau_advisory_card.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/pau_advisory_card.png")

# ============================================
# SAVE ALL ADVISORIES
# ============================================

adv_df.to_csv(
    os.path.join(base, '..', 'data',
                 'point_advisories.csv'),
    index=False
)

block_adv_df = pd.DataFrame(block_advisories)
block_adv_df.to_csv(
    os.path.join(base, '..', 'data',
                 'block_advisories.csv'),
    index=False
)

print("Saved: data/point_advisories.csv")
print("Saved: data/block_advisories.csv")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 55)
print("  PRIORITY 2 COMPLETE")
print("=" * 55)

print(f"\nKhamanon Block Advisory Summary:")
print(f"  Total sample points : {len(soil)}")
print(f"  Total advisories    : {len(adv_df)}")
print(f"  Unique rules fired  : {adv_df['rule_id'].nunique()}")

sev_summary = adv_df['severity'].value_counts()
print(f"\nSeverity breakdown:")
for sev, count in sev_summary.items():
    pct = count / len(adv_df) * 100
    print(f"  {sev:<10}: {count:>4} ({pct:.1f}%)")

print(f"\nBlock-level PAU recommendations:")
for a in block_advisories:
    print(f"  {a['rule_id']}: {a['message']}")

print("\nNext: Priority 3 — Weather API Integration")