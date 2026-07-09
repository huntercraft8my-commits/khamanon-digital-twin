# ============================================
# DSS - KHAMANON BLOCK
# Script 2: Validate and Fix Data
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 55)
print("  PHASE 1 — STEP 1.2: VALIDATE AND FIX DATA")
print("=" * 55)

# Load clean data
df = pd.read_csv(os.path.join('..', 'data', 'soil_data_clean.csv'))
print(f"\nLoaded: {len(df)} samples")

# ============================================
# FIX IMPOSSIBLE VALUES
# ============================================

print("\nChecking for impossible values...")

soil_cols = {
    'bulk_density' : (100, 400),     # g/L realistic range
    'pH'           : (4.0, 10.0),    # soil pH range
    'EC'           : (0.01, 5.0),    # dS/m range
    'K2O'          : (0, 500),       # kg/ha
    'available_P'  : (0, 300),       # kg/ha
    'OC'           : (0, 5.0),       # % range
    'CaCO3'        : (0, 30.0),      # % range
    'available_N'  : (0, 600),       # kg/ha
    'CEC'          : (0, 50)         # meq/100g
}

for col, (low, high) in soil_cols.items():
    # Values below minimum
    n_low = (df[col] < low).sum()
    if n_low > 0:
        median_val = df[df[col] >= low][col].median()
        df.loc[df[col] < low, col] = median_val
        print(f"  {col}: fixed {n_low} values below {low} → replaced with {median_val:.3f}")

    # Values above maximum
    n_high = (df[col] > high).sum()
    if n_high > 0:
        print(f"  {col}: WARNING {n_high} values above {high} — check these")
        print(df[df[col] > high][[col]].to_string())

print("\nAll impossible values fixed.")

# ============================================
# PLOT DISTRIBUTIONS
# So you can see your real data visually
# ============================================

print("\nGenerating distribution plots...")

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

plot_cols = ['pH', 'OC', 'EC', 'K2O', 'available_P',
             'available_N', 'CEC', 'bulk_density', 'CaCO3']

colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6',
          '#f39c12', '#1abc9c', '#e67e22', '#34495e', '#e91e63']

units = {
    'pH'          : 'pH units',
    'OC'          : '%',
    'EC'          : 'dS/m',
    'K2O'         : 'kg/ha',
    'available_P' : 'kg/ha',
    'available_N' : 'kg/ha',
    'CEC'         : 'meq/100g',
    'bulk_density': 'g/L',
    'CaCO3'       : '%'
}

for i, (col, color) in enumerate(zip(plot_cols, colors)):
    axes[i].hist(df[col], bins=20, color=color,
                 edgecolor='white', linewidth=0.5)
    axes[i].set_title(
        f"{col.replace('_',' ').title()} ({units[col]})",
        fontsize=11, fontweight='bold'
    )
    axes[i].set_xlabel(units[col], fontsize=9)
    axes[i].set_ylabel('Count', fontsize=9)
    axes[i].axvline(df[col].mean(), color='red',
                    linestyle='--', linewidth=1.5,
                    label=f'Mean={df[col].mean():.2f}')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.suptitle(
    'Soil Property Distributions — Khamanon Block (Real Data)\n'
    f'n = {len(df)} cLHS Sample Points',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps', 'real_data_distributions.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/real_data_distributions.png")

# ============================================
# CORRELATION MATRIX
# Shows which soil properties are related
# ============================================

fig, ax = plt.subplots(figsize=(10, 8))

corr_cols = ['pH', 'OC', 'EC', 'K2O', 'available_P',
             'available_N', 'CEC', 'bulk_density', 'CaCO3']

corr = df[corr_cols].corr().round(2)

im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, shrink=0.8)

ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(
    [c.replace('_',' ').title() for c in corr_cols],
    rotation=45, ha='right', fontsize=9
)
ax.set_yticklabels(
    [c.replace('_',' ').title() for c in corr_cols],
    fontsize=9
)

for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, str(corr.iloc[i, j]),
                ha='center', va='center',
                fontsize=8,
                color='black' if abs(corr.iloc[i, j]) < 0.7
                else 'white')

ax.set_title(
    'Soil Property Correlation Matrix\nKhamanon Block (Real Data)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    os.path.join('..', 'maps', 'real_data_correlation.png'),
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: maps/real_data_correlation.png")

# ============================================
# SAVE FINAL VALIDATED DATA
# ============================================

df.to_csv(
    os.path.join('..', 'data', 'soil_data_validated.csv'),
    index=False
)

print("\n" + "=" * 55)
print("  STEP 1.2 COMPLETE")
print("=" * 55)
print(f"\nValidated data saved: data/soil_data_validated.csv")
print(f"Samples: {len(df)}")
print("\nFinal summary statistics:")
print(df[corr_cols].describe().round(3).to_string())