# -*- coding: utf-8 -*-
# ================================================================
# Khamanon LULC Event Tagger
# Input:  data/field_events.csv
# Output: data/lulc_summary.json
#         data/field_events_tagged.csv
# Purpose: Link each field operation event to its crop class
#          and area from the two seasonal LULC maps
# ================================================================

import pandas as pd
import json
import os
from datetime import datetime

base = os.path.dirname(os.path.abspath(__file__))

# ── 1. LULC AREA STATISTICS ────────────────────────────────────
# These values come directly from GEE Console output (verified)

BLOCK_AREA_HA = 19616.02

RABI_STATS = {
    'season'          : 'Rabi 2025-26',
    'window'          : 'Nov 2025 - May 2026',
    'method'          : 'Supervised RF · Sentinel-2',
    'model_accuracy'  : 98.35,
    'images_used'     : 89,
    'classes': {
        'Wheat': {
            'area_ha': 17241.76,
            'pct'    : 87.9,
            'color'  : '#fbbf24',
            'label'  : 'Wheat (Rabi)'
        },
        'Spring_Maize': {
            'area_ha': 2157.79,
            'pct'    : 11.0,
            'color'  : '#16a34a',
            'label'  : 'Spring Maize'
        },
        'Agroforestry': {
            'area_ha': 216.47,
            'pct'    : 1.1,
            'color'  : '#064e3b',
            'label'  : 'Agroforestry'
        }
    }
}

KHARIF_STATS = {
    'season'          : 'Kharif 2025',
    'window'          : 'Jun 2025 - Nov 2025',
    'method'          : 'Supervised RF · Sentinel-2 + SAR Fusion',
    'model_accuracy'  : 91.94,
    's2_images_used'  : 89,
    'sar_images_used' : 32,
    'classes': {
        'Rice': {
            'area_ha': 14736.42,
            'pct'    : 75.1,
            'color'  : '#22c55e',
            'label'  : 'Rice (Paddy)'
        },
        'Kharif_Maize': {
            'area_ha': 4877.71,
            'pct'    : 24.9,
            'color'  : '#fb923c',
            'label'  : 'Kharif Maize'
        },
        'Agroforestry': {
            'area_ha': 1.88,
            'pct'    : 0.0,
            'color'  : '#064e3b',
            'label'  : 'Agroforestry'
        }
    }
}

# Cropping rotation — from training points cross-tab analysis
ROTATION_STATS = [
    {
        'rotation'  : 'Wheat → Rice',
        'points'    : 75,
        'pct'       : 54.3,
        'area_est'  : 10654,
        'color'     : '#22c55e',
        'note'      : 'Dominant Punjab rotation'
    },
    {
        'rotation'  : 'Wheat → Kharif Maize',
        'points'    : 17,
        'pct'       : 12.3,
        'area_est'  : 2413,
        'color'     : '#fb923c',
        'note'      : 'Water-scarce or dairy-farm plots'
    },
    {
        'rotation'  : 'Spring Maize → Kharif Maize',
        'points'    : 12,
        'pct'       : 8.7,
        'area_est'  : 1707,
        'color'     : '#f59e0b',
        'note'      : 'Continuous maize - silage/dairy'
    },
    {
        'rotation'  : 'Spring Maize → Rice',
        'points'    : 7,
        'pct'       : 5.1,
        'area_est'  : 994,
        'color'     : '#3b82f6',
        'note'      : 'Intensive cropping'
    },
    {
        'rotation'  : 'Other / Mixed',
        'points'    : 27,
        'pct'       : 19.6,
        'area_est'  : 3849,
        'color'     : '#94a3b8',
        'note'      : 'Vegetables, fodder, settlements'
    }
]

# ── 2. EVENT-TO-CLASS MAPPING RULES ────────────────────────────
# Based on event timing and Punjab crop calendar
# Each event is assigned to the crop class that was
# physically present in those fields at the time of detection

EVENT_CLASS_MAP = {
    # Rabi events — wheat (and small Spring Maize component)
    ('HARVESTING',      'Rabi'  ): {
        'crop_class' : 'Wheat',
        'area_ha'    : 17241.76,
        'season_lulc': 'Rabi 2025-26',
        'note'       : 'Wheat harvested from 17,242 ha. '
                       'Minor Spring Maize contribution (~2,158 ha).'
    },
    ('STUBBLE_BURNING', 'Rabi'  ): {
        'crop_class' : 'Wheat (post-harvest)',
        'area_ha'    : 17241.76,
        'season_lulc': 'Rabi 2025-26',
        'note'       : 'Residue burning on harvested wheat fields. '
                       'NBR crossed from positive to negative.'
    },
    # Kharif events — rice (and Kharif Maize component)
    ('RICE_TRANSPLANTING', 'Kharif'): {
        'crop_class' : 'Rice (transplanted)',
        'area_ha'    : 14736.42,
        'season_lulc': 'Kharif 2025',
        'note'       : 'Rice transplanted onto 14,736 ha of '
                       'formerly harvested wheat ground.'
    },
    ('FIELD_FLOODING',  'Kharif'): {
        'crop_class' : 'Rice (puddling)',
        'area_ha'    : 14736.42,
        'season_lulc': 'Kharif 2025',
        'note'       : 'Field puddling for rice transplanting. '
                       'Water signal from Sentinel-2 and SAR.'
    },
    ('HARVESTING',      'Kharif'): {
        'crop_class' : 'Rice',
        'area_ha'    : 14736.42,
        'season_lulc': 'Kharif 2025',
        'note'       : 'Rice harvested from 14,736 ha. '
                       'Kharif Maize (4,878 ha) harvested earlier.'
    },
    ('STUBBLE_BURNING', 'Kharif'): {
        'crop_class' : 'Rice (post-harvest)',
        'area_ha'    : 14736.42,
        'season_lulc': 'Kharif 2025',
        'note'       : 'Post-rice stubble burning. '
                       'NBR sign change after rice harvest.'
    },
    ('PLOUGHING',       'Rabi'  ): {
        'crop_class' : 'Wheat (field prep)',
        'area_ha'    : 17241.76,
        'season_lulc': 'Rabi 2025-26',
        'note'       : 'Field tillage before wheat sowing.'
    },
}

def get_season(month):
    """Determine season from event month number."""
    if month in [3, 4, 5]:
        return 'Rabi'
    elif month in [6, 7, 8, 9]:
        return 'Kharif'
    elif month in [10, 11, 12]:
        return 'Kharif'
    else:
        return 'Rabi'


# ── 3. LOAD AND TAG EVENTS ─────────────────────────────────────
ev_path = os.path.join(base, '..', 'data', 'field_events.csv')
ev = pd.read_csv(ev_path)
ev['date'] = pd.to_datetime(ev['date'])
ev['month'] = ev['date'].dt.month

print("=" * 60)
print("KHAMANON LULC EVENT TAGGER")
print("=" * 60)
print(f"\nLoaded {len(ev)} events from field_events.csv")

tagged_rows = []
for _, row in ev.iterrows():
    season = get_season(row['month'])
    key    = (row['event'], season)
    info   = EVENT_CLASS_MAP.get(key, {
        'crop_class' : 'Unknown',
        'area_ha'    : 0,
        'season_lulc': 'Unknown',
        'note'       : 'Event-class mapping not defined'
    })

    tagged_rows.append({
        'date'        : row['date'].strftime('%Y-%m-%d'),
        'event'       : row['event'],
        'confidence'  : row['confidence'],
        'signal'      : row['signal'],
        'season'      : season,
        'crop_class'  : info['crop_class'],
        'area_ha'     : info['area_ha'],
        'season_lulc' : info['season_lulc'],
        'note'        : row.get('note', ''),
        'class_note'  : info['note']
    })

tagged = pd.DataFrame(tagged_rows)

# Save tagged events
out_ev = os.path.join(base, '..', 'data', 'field_events_tagged.csv')
tagged.to_csv(out_ev, index=False)

print(f"\nTagged events:")
print(tagged[['date', 'event', 'season', 'crop_class', 'area_ha']].to_string(index=False))

# ── 4. BUILD LULC SUMMARY JSON ─────────────────────────────────
summary = {
    'generated'         : datetime.now().strftime('%Y-%m-%d %H:%M'),
    'block_area_ha'     : BLOCK_AREA_HA,
    'seasons_mapped'    : 2,
    'rabi_2025_26'      : RABI_STATS,
    'kharif_2025'       : KHARIF_STATS,
    'cropping_rotation' : ROTATION_STATS,
    'events_tagged'     : len(tagged),
    'feature_importance_top5': [
        {'rank': 1, 'feature': 's2_prekharif_ndvi',    'importance': 36.74,
         'meaning': 'Pre-Kharif bare soil — rice fields bare, maize already growing'},
        {'rank': 2, 'feature': 's2_flood_ndvi',        'importance': 20.42,
         'meaning': 'NDVI during flood window — rice transplanting period'},
        {'rank': 3, 'feature': 's2_flood_ndwi',        'importance': 17.61,
         'meaning': 'Water index during flooding — standing water in rice fields'},
        {'rank': 4, 'feature': 'sar_peak_vhvv_ratio',  'importance': 14.28,
         'meaning': 'SAR cross-pol ratio at peak — rice stem structure vs maize leaves'},
        {'rank': 5, 'feature': 's2_flood_mndwi',       'importance': 13.43,
         'meaning': 'SWIR water index — confirms flooded field condition'},
    ]
}

out_json = os.path.join(base, '..', 'data', 'lulc_summary.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

# ── 5. PRINT FINAL SUMMARY ─────────────────────────────────────
print(f"\n{'=' * 60}")
print("SEASONAL LULC SUMMARY")
print(f"{'=' * 60}")

print(f"\nRABI 2025-26 (Optical RF · {RABI_STATS['model_accuracy']}% accuracy)")
for cls, data in RABI_STATS['classes'].items():
    print(f"  {cls:<15s}: {data['area_ha']:>10,.1f} ha  ({data['pct']:>5.1f}%)")

print(f"\nKHARIF 2025 (SAR+Optical Fusion · {KHARIF_STATS['model_accuracy']}% accuracy)")
for cls, data in KHARIF_STATS['classes'].items():
    print(f"  {cls:<15s}: {data['area_ha']:>10,.1f} ha  ({data['pct']:>5.1f}%)")

print(f"\n{'=' * 60}")
print("CROPPING ROTATION PATTERNS")
print(f"{'=' * 60}")
for r in ROTATION_STATS:
    print(f"  {r['rotation']:<30s}: {r['points']:>3} points  "
          f"~{r['area_est']:>6,} ha  ({r['pct']:>5.1f}%)")

print(f"\n{'=' * 60}")
print("OUTPUTS SAVED")
print(f"{'=' * 60}")
print(f"  data/lulc_summary.json")
print(f"  data/field_events_tagged.csv")
print(f"\nDone. Ready to build dashboard Tab 8.")