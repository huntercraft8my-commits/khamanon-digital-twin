# ============================================
# DSS - KHAMANON BLOCK
# Script 11: SHAP Values + Uncertainty Maps
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
import shap
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  PRIORITY 1 — SHAP + UNCERTAINTY ANALYSIS")
print("  Khamanon Block — Real Data")
print("=" * 55)

# ============================================
# LOAD DATA
# ============================================

base = os.path.dirname(os.path.abspath(__file__))

master = pd.read_csv(
    os.path.join(base, '..', 'data',
                 'master_training_data.csv')
)

feature_cols = [
    'dem', 'slope', 'aspect',
    'lulc', 'lithology', 'geomorphology',
    'NDVI', 'NDBI', 'SAVI', 'BSI'
]

target_cols = [
    'pH', 'OC', 'EC', 'K2O',
    'available_P', 'available_N',
    'CEC', 'bulk_density', 'CaCO3'
]

X = master[feature_cols].fillna(
    master[feature_cols].median()
)

print(f"\nData loaded: {len(master)} samples")
print(f"Features   : {len(feature_cols)}")

# ============================================
# STEP 1: SHAP ANALYSIS
# For each soil property:
# - compute SHAP values
# - show which features drive predictions
# - save summary plots
# ============================================

print("\n" + "=" * 55)
print("STEP 1: Computing SHAP Values...")
print("=" * 55)

shap_results = {}

for target in target_cols:

    print(f"\n  Computing SHAP for {target}...")

    model_path = os.path.join(
        base, '..', 'models',
        f'rf_real_{target}.pkl'
    )
    model = joblib.load(model_path)

    # TreeExplainer is fast for RF models
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_results[target] = {
        'shap_values': shap_values,
        'explainer'  : explainer
    }

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        'Feature'   : feature_cols,
        'SHAP_Mean' : mean_shap
    }).sort_values('SHAP_Mean', ascending=False)

    print(f"    Top 3 drivers for {target}:")
    for _, row in importance.head(3).iterrows():
        print(f"      {row['Feature']:<15}: "
              f"{row['SHAP_Mean']:.4f}")

print("\nSHAP values computed for all 9 properties.")

# ============================================
# PLOT 1: SHAP Summary — pH (most important)
# Beeswarm plot showing feature impact
# ============================================

print("\nGenerating SHAP summary plots...")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

colors_list = [
    '#3498db','#2ecc71','#e74c3c','#9b59b6',
    '#f39c12','#1abc9c','#e67e22','#34495e',
    '#e91e63'
]

for i, target in enumerate(target_cols):

    sv   = shap_results[target]['shap_values']
    mean = np.abs(sv).mean(axis=0)

    imp = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP'   : mean
    }).sort_values('SHAP', ascending=True)

    axes[i].barh(
        imp['Feature'],
        imp['SHAP'],
        color=colors_list[i],
        alpha=0.85
    )
    axes[i].set_title(
        f"SHAP — {target.replace('_',' ').title()}",
        fontsize=11, fontweight='bold'
    )
    axes[i].set_xlabel(
        'Mean |SHAP| value', fontsize=9
    )
    axes[i].grid(True, alpha=0.3, axis='x')

plt.suptitle(
    'SHAP Feature Importance — All Soil Properties\n'
    'Khamanon Block, Fatehgarh Sahib',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(base, '..', 'maps',
                 'shap_importance_all.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/shap_importance_all.png")

# ============================================
# PLOT 2: SHAP Dependence — pH vs NDVI
# Shows HOW NDVI affects pH prediction
# ============================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

key_pairs = [
    ('pH',           'NDVI'),
    ('OC',           'NDVI'),
    ('EC',           'NDBI'),
    ('available_N',  'SAVI'),
    ('bulk_density', 'dem'),
    ('CaCO3',        'lithology')
]

for i, (target, feature) in enumerate(key_pairs):

    sv      = shap_results[target]['shap_values']
    feat_idx= feature_cols.index(feature)
    shap_v  = sv[:, feat_idx]
    feat_v  = X[feature].values

    sc = axes[i].scatter(
        feat_v, shap_v,
        c=feat_v, cmap='RdYlGn',
        alpha=0.6, s=40,
        edgecolors='gray', linewidth=0.3
    )
    plt.colorbar(sc, ax=axes[i],
                 shrink=0.8,
                 label=feature)
    axes[i].axhline(y=0, color='black',
                    linestyle='--',
                    linewidth=1, alpha=0.5)
    axes[i].set_xlabel(feature, fontsize=10)
    axes[i].set_ylabel(
        f'SHAP value for {target}',
        fontsize=10
    )
    axes[i].set_title(
        f'How {feature} affects '
        f'{target.replace("_"," ").title()}',
        fontsize=10, fontweight='bold'
    )
    axes[i].grid(True, alpha=0.3)

plt.suptitle(
    'SHAP Dependence Plots — Key Relationships\n'
    'Khamanon Block (How each feature drives predictions)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(base, '..', 'maps',
                 'shap_dependence.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/shap_dependence.png")

# ============================================
# STEP 2: UNCERTAINTY MAPS
# RF gives multiple tree predictions
# Std dev across trees = uncertainty
# ============================================

print("\n" + "=" * 55)
print("STEP 2: Computing Uncertainty Maps...")
print("=" * 55)

grid = pd.read_csv(
    os.path.join(base, '..', 'data',
                 'real_prediction_grid.csv')
)

feature_means = master[feature_cols].median()

grid_features = [
    'dem','slope','aspect',
    'lulc','lithology','geomorphology',
    'NDVI','NDBI','SAVI','BSI'
]

available = [c for c in grid_features
             if c in grid.columns]
missing   = [c for c in grid_features
             if c not in grid.columns]

grid_X = pd.DataFrame()
for col in grid_features:
    if col in grid.columns:
        grid_X[col] = grid[col]
    else:
        grid_X[col] = feature_means.get(col, 0)

grid_X = grid_X.fillna(grid_X.median())

# If grid is empty use master data for uncertainty
if len(grid_X) == 0:
    print("Grid missing raster cols — using master data")
    grid_X   = X.copy()
    grid_lats = master['northing_utm'].values
    grid_lons = master['easting_utm'].values
    use_master = True
else:
    grid_lats = grid['northing'].values
    grid_lons = grid['easting'].values
    use_master = False

print(f"\nGrid points: {len(grid_X):,}")
print(f"Computing uncertainty (std across trees)...")
uncertainty_results = {}
# ============================================
# PLOT 3: Uncertainty Maps — pH and OC
# Most important properties
# ============================================

print("\nGenerating uncertainty maps...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

plot_targets = ['pH', 'OC', 'EC']

for col_idx, target in enumerate(plot_targets):

    res  = uncertainty_results[target]
    lats = grid_lats
    lons = grid_lons

    # Prediction map
    sc1 = axes[0, col_idx].scatter(
        lons, lats,
        c=res['mean'],
        cmap='RdYlGn_r' if target == 'pH'
             else 'YlGn',
        s=3, alpha=0.6
    )
    plt.colorbar(sc1, ax=axes[0, col_idx],
                 shrink=0.8)
    axes[0, col_idx].set_title(
        f'Predicted {target}',
        fontsize=11, fontweight='bold'
    )
    axes[0, col_idx].set_xlabel(
        'Easting (m)', fontsize=9
    )
    axes[0, col_idx].set_ylabel(
        'Northing (m)', fontsize=9
    )

    # Uncertainty map
    sc2 = axes[1, col_idx].scatter(
        lons, lats,
        c=res['cv'],
        cmap='YlOrRd',
        s=3, alpha=0.6,
        vmin=0,
        vmax=np.percentile(res['cv'], 95)
    )
    plt.colorbar(sc2, ax=axes[1, col_idx],
                 shrink=0.8,
                 label='CV%')
    axes[1, col_idx].set_title(
        f'Uncertainty — {target} (CV%)',
        fontsize=11, fontweight='bold'
    )
    axes[1, col_idx].set_xlabel(
        'Easting (m)', fontsize=9
    )
    axes[1, col_idx].set_ylabel(
        'Northing (m)', fontsize=9
    )

plt.suptitle(
    'Prediction vs Uncertainty Maps\n'
    'Khamanon Block — Top row: Prediction | '
    'Bottom row: Uncertainty (CV%)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join(base, '..', 'maps',
                 'uncertainty_maps.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/uncertainty_maps.png")

# ============================================
# SAVE UNCERTAINTY DATA FOR DASHBOARD
# ============================================

uncertainty_df = pd.DataFrame({
    'easting' : grid_lons,
    'northing': grid_lats
})

for target in target_cols:
    uncertainty_df[f'{target}_std'] = (
        uncertainty_results[target]['std']
    )
    uncertainty_df[f'{target}_cv'] = (
        uncertainty_results[target]['cv']
    )

uncertainty_df.to_csv(
    os.path.join(base, '..', 'data',
                 'uncertainty_grid.csv'),
    index=False
)
print("Saved: data/uncertainty_grid.csv")

# ============================================
# SAVE SHAP IMPORTANCE SUMMARY
# ============================================

shap_summary = {}
for target in target_cols:
    sv   = shap_results[target]['shap_values']
    mean = np.abs(sv).mean(axis=0)
    shap_summary[target] = dict(
        zip(feature_cols, mean.round(4))
    )

shap_df = pd.DataFrame(shap_summary).T
shap_df.index.name = 'soil_property'
shap_df.to_csv(
    os.path.join(base, '..', 'data',
                 'shap_importance.csv')
)
print("Saved: data/shap_importance.csv")

# ============================================
# PRINT KEY FINDINGS
# ============================================

print("\n" + "=" * 55)
print("  PRIORITY 1 COMPLETE")
print("=" * 55)

print("\nKey SHAP Findings:")
print("-" * 55)
print("(Top driver for each soil property)\n")

for target in target_cols:
    sv   = shap_results[target]['shap_values']
    mean = np.abs(sv).mean(axis=0)
    top_idx     = mean.argmax()
    top_feature = feature_cols[top_idx]
    top_val     = mean[top_idx]
    print(f"  {target:<15}: driven by "
          f"{top_feature:<15} "
          f"(SHAP={top_val:.4f})")

print("\nKey Uncertainty Findings:")
print("-" * 55)
for target in ['pH','OC','EC']:
    cv_mean = uncertainty_results[target]['cv'].mean()
    cv_max  = uncertainty_results[target]['cv'].max()
    print(f"  {target:<15}: "
          f"mean CV={cv_mean:.1f}% | "
          f"max CV={cv_max:.1f}%")

print("\nFiles saved:")
print("  maps/shap_importance_all.png")
print("  maps/shap_dependence.png")
print("  maps/uncertainty_maps.png")
print("  data/uncertainty_grid.csv")
print("  data/shap_importance.csv")
print("\nNext: Priority 2 — Recommendation Engine")