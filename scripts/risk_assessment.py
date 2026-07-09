# ============================================
# DSS - KHAMANON BLOCK
# Script 14: Risk Assessment Maps
# Composite soil + crop + salinity risk
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os

print("=" * 55)
print("  PRIORITY 4 — RISK ASSESSMENT MAPS")
print("  Khamanon Block — Real Data")
print("=" * 55)

base = os.path.dirname(os.path.abspath(__file__))

# ============================================
# LOAD DATA
# ============================================

soil = pd.read_csv(
    os.path.join(base,'..','data',
                 'soil_data_validated.csv')
)

grid = pd.read_csv(
    os.path.join(base,'..','data',
                 'real_prediction_grid.csv')
)

# Load weather if available
weather_path = os.path.join(
    base,'..','data','current_weather.json'
)
if os.path.exists(weather_path):
    with open(weather_path,'r') as f:
        weather = json.load(f)
    temp     = weather.get('temperature', 35)
    humidity = weather.get('humidity', 45)
else:
    temp     = 35
    humidity = 45

# Load NDVI status
status_path = os.path.join(
    base,'..','data','last_update.json'
)
if os.path.exists(status_path):
    with open(status_path,'r') as f:
        status = json.load(f)
    ndvi_mean = status.get('ndvi_mean', 0.35)
else:
    ndvi_mean = 0.35

print(f"\nSample points : {len(soil)}")
print(f"Grid points   : {len(grid)}")
print(f"Temperature   : {temp}°C")
print(f"Humidity      : {humidity}%")
print(f"NDVI mean     : {ndvi_mean:.3f}")

# ============================================
# RISK SCORING FUNCTIONS
# Each returns 0-100 (0=safe, 100=high risk)
# ============================================

def normalize(val, low, high):
    """Normalize value to 0-1 range"""
    return np.clip((val - low) / (high - low), 0, 1)

def soil_degradation_risk(pH, OC, bulk_density, CaCO3):
    """
    Soil Degradation Risk Score (0-100)
    Based on PAU thresholds
    """
    # pH risk: high pH = high risk
    pH_risk = normalize(pH, 7.0, 9.5) * 100

    # OC risk: low OC = high risk (inverted)
    OC_risk = (1 - normalize(OC, 0.0, 0.75)) * 100

    # Bulk density risk: high = compaction
    BD_risk = normalize(bulk_density, 150, 350) * 100

    # CaCO3 risk: high = nutrient fixation
    CaCO3_risk = normalize(CaCO3, 0, 5) * 100

    # Weighted composite
    composite = (
        0.35 * pH_risk +
        0.35 * OC_risk +
        0.20 * BD_risk +
        0.10 * CaCO3_risk
    )
    return np.round(composite, 1)

def crop_failure_risk(OC, available_N,
                      ndvi, temp, humidity):
    """
    Crop Failure Risk Score (0-100)
    Combines soil nutrition + satellite + weather
    """
    # NDVI risk: low NDVI = high risk
    ndvi_risk = (1 - normalize(ndvi, 0.1, 0.8)) * 100

    # Nitrogen risk: low N = high risk
    N_risk = (1 - normalize(available_N,
                            50, 400)) * 100

    # OC as nutrient buffer
    OC_risk = (1 - normalize(OC, 0.1, 0.75)) * 100

    # Temperature risk
    temp_risk = normalize(temp, 25, 45) * 100

    # Humidity risk (very low or very high)
    if humidity < 40:
        hum_risk = normalize(40 - humidity, 0, 40) * 100
    elif humidity > 80:
        hum_risk = normalize(humidity - 80, 0, 20) * 100
    else:
        hum_risk = 0

    composite = (
        0.35 * ndvi_risk +
        0.25 * N_risk +
        0.15 * OC_risk +
        0.15 * temp_risk +
        0.10 * hum_risk
    )
    return np.round(composite, 1)

def salinity_risk(EC, pH, CaCO3, OC):
    """
    Salinity / Salt-Affected Soil Risk (0-100)
    Based on PAU EC and pH thresholds
    """
    # EC risk: high EC = saline
    EC_risk = normalize(EC, 0, 1.0) * 100

    # pH contribution to alkalinity
    pH_risk = normalize(pH, 7.5, 9.5) * 100

    # CaCO3 contribution
    CaCO3_risk = normalize(CaCO3, 0, 5) * 100

    # OC buffers salinity (inverted)
    OC_buffer = normalize(OC, 0, 0.75) * 30

    composite = (
        0.45 * EC_risk +
        0.30 * pH_risk +
        0.15 * CaCO3_risk -
        0.10 * OC_buffer
    )
    return np.round(np.clip(composite, 0, 100), 1)

# ============================================
# COMPUTE RISK SCORES AT SAMPLE POINTS
# ============================================

print("\nComputing risk scores at 208 sample points...")

soil['degradation_risk'] = soil_degradation_risk(
    soil['pH'].values,
    soil['OC'].values,
    soil['bulk_density'].values,
    soil['CaCO3'].values
)

soil['crop_failure_risk'] = crop_failure_risk(
    soil['OC'].values,
    soil['available_N'].values,
    ndvi_mean,
    temp,
    humidity
)

soil['salinity_risk'] = salinity_risk(
    soil['EC'].values,
    soil['pH'].values,
    soil['CaCO3'].values,
    soil['OC'].values
)

# Composite overall risk
soil['overall_risk'] = (
    0.40 * soil['degradation_risk'] +
    0.40 * soil['crop_failure_risk'] +
    0.20 * soil['salinity_risk']
)

print("\nRisk Score Summary (sample points):")
for risk in ['degradation_risk',
             'crop_failure_risk',
             'salinity_risk',
             'overall_risk']:
    print(f"  {risk:<22}: "
          f"mean={soil[risk].mean():.1f} | "
          f"min={soil[risk].min():.1f} | "
          f"max={soil[risk].max():.1f}")

# ============================================
# COMPUTE RISK SCORES AT GRID POINTS
# For full spatial maps
# ============================================

print("\nComputing risk scores across full grid...")

grid['degradation_risk'] = soil_degradation_risk(
    grid['pH'].values,
    grid['OC'].values,
    grid['bulk_density'].values,
    grid['CaCO3'].values
)

grid['crop_failure_risk'] = crop_failure_risk(
    grid['OC'].values,
    grid['available_N'].values,
    ndvi_mean,
    temp,
    humidity
)

grid['salinity_risk'] = salinity_risk(
    grid['EC'].values,
    grid['pH'].values,
    grid['CaCO3'].values,
    grid['OC'].values
)

grid['overall_risk'] = (
    0.40 * grid['degradation_risk'] +
    0.40 * grid['crop_failure_risk'] +
    0.20 * grid['salinity_risk']
)

# ============================================
# RISK CATEGORIES
# ============================================

def risk_category(score):
    if score >= 70:   return 'HIGH'
    elif score >= 45: return 'MODERATE'
    else:             return 'LOW'

for risk in ['degradation_risk',
             'crop_failure_risk',
             'salinity_risk',
             'overall_risk']:
    col = risk.replace('_risk','_cat')
    soil[col] = soil[risk].apply(risk_category)

print("\nRisk Category Distribution:")
for risk in ['degradation_risk',
             'crop_failure_risk',
             'salinity_risk']:
    cat_col = risk.replace('_risk','_cat')
    counts  = soil[cat_col].value_counts()
    print(f"\n  {risk}:")
    for cat, count in counts.items():
        pct = count / len(soil) * 100
        print(f"    {cat:<10}: "
              f"{count:>3} points ({pct:.1f}%)")

# ============================================
# PLOT 1: Four Risk Maps Side by Side
# ============================================

print("\nGenerating risk maps...")

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
axes = axes.flatten()

risk_configs = [
    ('overall_risk',     'Overall Risk Score',
     'RdYlGn_r'),
    ('degradation_risk', 'Soil Degradation Risk',
     'RdYlGn_r'),
    ('crop_failure_risk','Crop Failure Risk',
     'RdYlGn_r'),
    ('salinity_risk',    'Salinity Risk',
     'RdYlGn_r'),
]

for i, (col, title, cmap) in enumerate(
        risk_configs):

    sc = axes[i].scatter(
        grid['easting'],
        grid['northing'],
        c=grid[col],
        cmap=cmap,
        s=3,
        alpha=0.7,
        vmin=0,
        vmax=100
    )

    cb = plt.colorbar(sc, ax=axes[i], shrink=0.8)
    cb.set_label('Risk Score (0-100)', fontsize=9)

    # Add HIGH risk contour overlay
    high_mask = grid[col] >= 70
    if high_mask.sum() > 0:
        axes[i].scatter(
            grid.loc[high_mask, 'easting'],
            grid.loc[high_mask, 'northing'],
            c='red', s=8, alpha=0.4,
            marker='x', linewidths=0.5,
            label=f'HIGH risk '
                  f'({high_mask.sum()} pts)'
        )
        axes[i].legend(fontsize=8)

    axes[i].set_title(
        title, fontsize=11, fontweight='bold'
    )
    axes[i].set_xlabel('Easting (UTM m)', fontsize=9)
    axes[i].set_ylabel('Northing (UTM m)', fontsize=9)
    axes[i].grid(True, alpha=0.2)

plt.suptitle(
    'Composite Risk Assessment — Khamanon Block\n'
    f'Temperature: {temp}°C | Humidity: {humidity}% | '
    f'NDVI: {ndvi_mean:.3f}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(base,'..','maps',
                 'risk_assessment_maps.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/risk_assessment_maps.png")

# ============================================
# PLOT 2: Risk Score Distribution
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

risks = [
    ('degradation_risk', 'Soil Degradation',
     '#e74c3c'),
    ('crop_failure_risk','Crop Failure',
     '#f39c12'),
    ('salinity_risk',    'Salinity',
     '#9b59b6')
]

for ax, (col, title, color) in zip(axes, risks):
    ax.hist(
        soil[col], bins=20,
        color=color, alpha=0.8,
        edgecolor='white', linewidth=0.5
    )
    ax.axvline(
        x=70, color='red',
        linestyle='--', linewidth=2,
        label='High risk (70)'
    )
    ax.axvline(
        x=45, color='orange',
        linestyle='--', linewidth=2,
        label='Moderate risk (45)'
    )
    ax.axvline(
        x=soil[col].mean(),
        color='black',
        linestyle='-', linewidth=2,
        label=f'Mean ({soil[col].mean():.1f})'
    )
    ax.set_xlabel('Risk Score', fontsize=11)
    ax.set_ylabel('Number of Points', fontsize=11)
    ax.set_title(
        f'{title} Risk\nDistribution',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'Risk Score Distributions — 208 cLHS Points\n'
    'Khamanon Block, Fatehgarh Sahib',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(base,'..','maps',
                 'risk_distributions.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/risk_distributions.png")

# ============================================
# SAVE RISK DATA
# ============================================

# Sample point risks
soil[[
    'sample_id','latitude','longitude',
    'easting_utm','northing_utm',
    'degradation_risk','crop_failure_risk',
    'salinity_risk','overall_risk',
    'degradation_cat','crop_failure_cat',
    'salinity_cat'
]].to_csv(
    os.path.join(base,'..','data',
                 'point_risk_scores.csv'),
    index=False
)

# Grid risks
grid[[
    'easting','northing',
    'degradation_risk','crop_failure_risk',
    'salinity_risk','overall_risk'
]].to_csv(
    os.path.join(base,'..','data',
                 'grid_risk_scores.csv'),
    index=False
)

print("Saved: data/point_risk_scores.csv")
print("Saved: data/grid_risk_scores.csv")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 55)
print("  PRIORITY 4 COMPLETE")
print("=" * 55)

high_deg  = (soil['degradation_cat']  == 'HIGH').sum()
high_crop = (soil['crop_failure_cat'] == 'HIGH').sum()
high_sal  = (soil['salinity_cat']     == 'HIGH').sum()
high_over = (soil['overall_risk'] >= 70).sum()

print(f"\nKhamanon Block Risk Summary:")
print(f"  High degradation risk  : "
      f"{high_deg} points "
      f"({high_deg/len(soil)*100:.1f}%)")
print(f"  High crop failure risk : "
      f"{high_crop} points "
      f"({high_crop/len(soil)*100:.1f}%)")
print(f"  High salinity risk     : "
      f"{high_sal} points "
      f"({high_sal/len(soil)*100:.1f}%)")
print(f"  High overall risk      : "
      f"{high_over} points "
      f"({high_over/len(soil)*100:.1f}%)")

print(f"\nMean risk scores:")
print(f"  Degradation  : "
      f"{soil['degradation_risk'].mean():.1f}/100")
print(f"  Crop failure : "
      f"{soil['crop_failure_risk'].mean():.1f}/100")
print(f"  Salinity     : "
      f"{soil['salinity_risk'].mean():.1f}/100")
print(f"  Overall      : "
      f"{soil['overall_risk'].mean():.1f}/100")

print("\nNext: Priority 5 — Email Alert System")