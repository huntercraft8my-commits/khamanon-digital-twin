# ============================================
# DSS - KHAMANON BLOCK
# Script 5: Random Forest Model Training
# Real Data Pipeline — Phase 2
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble          import RandomForestRegressor
from sklearn.model_selection   import (train_test_split,
                                        cross_val_score,
                                        KFold)
from sklearn.metrics           import (r2_score,
                                        mean_squared_error,
                                        mean_absolute_error)
from sklearn.preprocessing     import LabelEncoder

print("=" * 55)
print("  PHASE 2 — RF MODEL TRAINING")
print("  Khamanon Block — Real Data")
print("=" * 55)

# ============================================
# LOAD MASTER DATASET
# ============================================

df = pd.read_csv(
    os.path.join('..', 'data', 'master_training_data.csv')
)
print(f"\nMaster dataset loaded: {df.shape}")

# ============================================
# DEFINE FEATURES AND TARGETS
# ============================================

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

X = df[feature_cols].copy()
print(f"\nFeatures : {len(feature_cols)}")
print(f"Targets  : {len(target_cols)}")
print(f"Samples  : {len(df)}")

# ============================================
# CHECK FOR ANY NaN IN FEATURES
# ============================================

nan_check = X.isnull().sum()
if nan_check.sum() > 0:
    print("\nFilling NaN in features with median...")
    for col in feature_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

# ============================================
# TRAIN RF MODELS
# One per soil property
# 70/30 split + 5-fold cross validation
# ============================================

results   = {}
models    = {}

print("\n" + "=" * 55)
print("Training models...")
print(f"{'Property':<15} {'R²':>6} {'RMSE':>8} "
      f"{'MAE':>8} {'CV R²':>8}")
print("-" * 55)

for target in target_cols:

    y = df[target].copy()

    # 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.30,
        random_state = 42
    )

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators = 200,
        max_depth    = 10,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features = 'sqrt',
        random_state = 42,
        n_jobs       = -1
    )
    rf.fit(X_train, y_train)

    # Test set metrics
    y_pred = rf.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)

    # 5-fold cross validation on full dataset
    kf     = KFold(n_splits=5, shuffle=True,
                   random_state=42)
    cv_r2  = cross_val_score(
        rf, X, y, cv=kf,
        scoring='r2'
    ).mean()

    results[target] = {
        'R2'   : round(r2,   3),
        'RMSE' : round(rmse, 4),
        'MAE'  : round(mae,  4),
        'CV_R2': round(cv_r2,3)
    }

    models[target] = rf

    print(f"{target:<15} {r2:>6.3f} {rmse:>8.4f} "
          f"{mae:>8.4f} {cv_r2:>8.3f}")

# ============================================
# SAVE ALL MODELS
# ============================================

os.makedirs(os.path.join('..', 'models'), exist_ok=True)

for target, model in models.items():
    path = os.path.join('..', 'models',
                        f'rf_real_{target}.pkl')
    joblib.dump(model, path)

print("-" * 55)
print("All 9 models saved to models/")

# ============================================
# SAVE VALIDATION RESULTS
# ============================================

results_df = pd.DataFrame(results).T
results_df.index.name = 'soil_property'
results_df.to_csv(
    os.path.join('..', 'data',
                 'model_validation_real.csv')
)

# ============================================
# PLOT 1: Actual vs Predicted — all 9 properties
# ============================================

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()

colors = ['#3498db','#2ecc71','#e74c3c','#9b59b6',
          '#f39c12','#1abc9c','#e67e22','#34495e','#e91e63']

units = {
    'pH'          : 'pH',
    'OC'          : '%',
    'EC'          : 'dS/m',
    'K2O'         : 'kg/ha',
    'available_P' : 'kg/ha',
    'available_N' : 'kg/ha',
    'CEC'         : 'meq/100g',
    'bulk_density': 'g/L',
    'CaCO3'       : '%'
}

for i, (target, color) in enumerate(
        zip(target_cols, colors)):

    y      = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    y_pred = models[target].predict(X_te)

    axes[i].scatter(
        y_te, y_pred,
        alpha=0.6, color=color,
        edgecolors='gray', linewidth=0.3, s=50
    )

    # Perfect prediction line
    mn = min(y_te.min(), y_pred.min())
    mx = max(y_te.max(), y_pred.max())
    axes[i].plot([mn, mx], [mn, mx],
                 'k--', linewidth=1.5,
                 label='1:1 line')

    r2 = results[target]['R2']
    axes[i].set_xlabel(
        f"Actual ({units[target]})", fontsize=9
    )
    axes[i].set_ylabel(
        f"Predicted ({units[target]})", fontsize=9
    )
    axes[i].set_title(
        f"{target.replace('_',' ').title()}\n"
        f"R²={r2:.3f}",
        fontsize=10, fontweight='bold'
    )
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.suptitle(
    'Random Forest — Actual vs Predicted\n'
    'Khamanon Block, Fatehgarh Sahib (Real Data)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps',
                 'rf_actual_vs_predicted.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/rf_actual_vs_predicted.png")

# ============================================
# PLOT 2: Variable Importance — all properties
# ============================================

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
axes = axes.flatten()

for i, (target, color) in enumerate(
        zip(target_cols, colors)):

    imp = pd.DataFrame({
        'Feature'   : feature_cols,
        'Importance': models[target].feature_importances_
    }).sort_values('Importance', ascending=True)

    bars = axes[i].barh(
        imp['Feature'],
        imp['Importance'],
        color=color, alpha=0.8
    )
    axes[i].set_title(
        f"Variable Importance\n"
        f"{target.replace('_',' ').title()}",
        fontsize=10, fontweight='bold'
    )
    axes[i].set_xlabel('Importance', fontsize=9)
    axes[i].grid(True, alpha=0.3, axis='x')

plt.suptitle(
    'Random Forest Variable Importance\n'
    'Khamanon Block, Fatehgarh Sahib (Real Data)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps',
                 'rf_variable_importance.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/rf_variable_importance.png")

# ============================================
# PLOT 3: Model Accuracy Summary Bar Chart
# ============================================

fig, ax = plt.subplots(figsize=(11, 5))

props  = list(results.keys())
r2vals = [results[p]['R2']    for p in props]
cvvals = [results[p]['CV_R2'] for p in props]

x     = np.arange(len(props))
width = 0.35

bars1 = ax.bar(x - width/2, r2vals, width,
               label='Test R²',
               color='#3498db', alpha=0.85,
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, cvvals, width,
               label='CV R² (5-fold)',
               color='#2ecc71', alpha=0.85,
               edgecolor='black', linewidth=0.5)

ax.axhline(y=0.5, color='orange',
           linestyle='--', linewidth=1.5,
           label='Moderate (0.5)')
ax.axhline(y=0.3, color='red',
           linestyle='--', linewidth=1.5,
           label='Weak (0.3)')

ax.set_ylabel('R² Score', fontsize=12)
ax.set_title(
    'Model Accuracy — All Soil Properties\n'
    'Khamanon Block (Real Data)',
    fontsize=13, fontweight='bold'
)
ax.set_xticks(x)
ax.set_xticklabels(
    [p.replace('_','\n') for p in props],
    fontsize=9
)
ax.set_ylim(-0.5, 1.0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars1, r2vals):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.02,
        f'{val:.2f}',
        ha='center', fontsize=8
    )
for bar, val in zip(bars2, cvvals):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.02,
        f'{val:.2f}',
        ha='center', fontsize=8
    )

plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps',
                 'rf_model_accuracy.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/rf_model_accuracy.png")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 55)
print("  PHASE 2 — STEP 2.1 COMPLETE")
print("=" * 55)
print("\nModel Performance Summary:")
print("-" * 55)
print(f"{'Property':<15} {'Test R²':>8} {'CV R²':>8} "
      f"{'RMSE':>10} {'MAE':>10}")
print("-" * 55)
for prop in target_cols:
    r = results[prop]
    print(f"{prop:<15} {r['R2']:>8.3f} "
          f"{r['CV_R2']:>8.3f} "
          f"{r['RMSE']:>10.4f} "
          f"{r['MAE']:>10.4f}")

print("\n3 charts saved to maps/")
print("9 models saved to models/")
print("\nNext: Phase 3 — Spatial Prediction Maps")