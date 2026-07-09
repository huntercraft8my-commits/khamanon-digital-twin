# ============================================
# DSS - KHAMANON BLOCK
# Script 16: Time-Series History Tracker
# Records each update run permanently
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

print("=" * 55)
print("  PRIORITY 7 — TIME-SERIES HISTORY")
print("  Khamanon Block DSS")
print("=" * 55)

base = os.path.dirname(os.path.abspath(__file__))

history_path = os.path.join(
    base, '..', 'data', 'update_history.csv'
)

# ============================================
# LOAD CURRENT STATE
# ============================================

status_path = os.path.join(
    base, '..', 'data', 'last_update.json'
)
if os.path.exists(status_path):
    with open(status_path, 'r') as f:
        status = json.load(f)
else:
    print("No update status found.")
    status = {}

grid = pd.read_csv(
    os.path.join(base,'..','data',
                 'real_prediction_grid.csv')
)

soil_cols = [
    'pH','OC','EC','K2O','available_P',
    'available_N','CEC','bulk_density','CaCO3'
]

# ============================================
# CREATE NEW HISTORY RECORD
# One row per update run
# ============================================

record = {
    'timestamp'      : datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S'),
    'sentinel2_date' : status.get(
                        'sentinel2_date','Unknown'),
    'images_used'    : status.get('images_used', 0),
    'ndvi_mean'      : status.get('ndvi_mean', 0),
    'ndbi_mean'      : status.get('ndbi_mean', 0),
}

for col in soil_cols:
    if col in grid.columns:
        record[col + '_mean'] = round(
            grid[col].mean(), 4)
        record[col + '_std']  = round(
            grid[col].std(), 4)
        record[col + '_min']  = round(
            grid[col].min(), 4)
        record[col + '_max']  = round(
            grid[col].max(), 4)

# ============================================
# APPEND TO HISTORY FILE
# Creates file on first run
# Appends on subsequent runs
# ============================================

new_row = pd.DataFrame([record])

if os.path.exists(history_path):
    history = pd.read_csv(history_path)
    history = pd.concat(
        [history, new_row],
        ignore_index=True
    )
    print(f"\nHistory file exists.")
    print(f"Previous records : {len(history)-1}")
    print(f"Adding new record: {record['timestamp']}")
else:
    history = new_row
    print(f"\nFirst history record created.")
    print(f"Timestamp: {record['timestamp']}")

history.to_csv(history_path, index=False)
print(f"Total records now: {len(history)}")

# ============================================
# SHOW HISTORY SUMMARY
# ============================================

print("\nHistory summary:")
print("-" * 40)
print(history[[
    'timestamp','sentinel2_date',
    'ndvi_mean','pH_mean','OC_mean'
]].to_string(index=False))

# ============================================
# PLOT HISTORY (if enough records)
# ============================================

if len(history) >= 2:
    print("\nGenerating temporal trend charts...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    plot_cols = [
        ('ndvi_mean',    'NDVI',           'green'),
        ('pH_mean',      'pH',             '#e74c3c'),
        ('OC_mean',      'Organic Carbon %','#2ecc71'),
        ('EC_mean',      'EC dS/m',        '#e67e22'),
        ('available_N_mean','Available N', '#3498db'),
        ('K2O_mean',     'K2O kg/ha',      '#9b59b6')
    ]

    for i, (col, label, color) in enumerate(plot_cols):
        if col in history.columns:
            axes[i].plot(
                range(len(history)),
                history[col],
                color=color,
                marker='o',
                linewidth=2.5,
                markersize=8
            )
            axes[i].set_title(
                label, fontsize=11,
                fontweight='bold'
            )
            axes[i].set_xlabel(
                'Update Number', fontsize=9
            )
            axes[i].set_ylabel(label, fontsize=9)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks(range(len(history)))
            axes[i].set_xticklabels(
                [h[:10] for h in
                 history['timestamp'].tolist()],
                rotation=45, fontsize=7
            )

    plt.suptitle(
        'Temporal Evolution — Khamanon Block\n'
        'Soil Properties Over Update Cycles',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(base,'..','maps',
                     'temporal_trends.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: maps/temporal_trends.png")

else:
    print("\nOnly 1 record so far.")
    print("Run realtime_updater.py + this script")
    print("again after next Sentinel-2 pass")
    print("to see temporal trends.")
    print("(Next S2 pass ~May 28)")

print("\n" + "=" * 55)
print("  PRIORITY 7 COMPLETE")
print("=" * 55)
print(f"\nHistory file: data/update_history.csv")
print(f"Records      : {len(history)}")
print("\nRun this script after every")
print("realtime_updater.py to build")
print("your temporal soil record.")
print("\nAll 7 Priorities Complete.")
print("Next: Visual Redesign of Dashboard")