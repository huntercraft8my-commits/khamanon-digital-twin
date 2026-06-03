# ================================================================
# Khamanon Field Operations Detector  v4  — with event consolidation
# Input:  data/khamanon_multiindex_timeseries.csv
# Output: data/field_events.csv
#         data/multiindex_timeseries_clean.csv
#         data/field_ops_status.json
# ================================================================

import pandas as pd
import numpy as np
import json
from datetime import datetime

# ── 1. LOAD & CLEAN ─────────────────────────────────────────────
df = pd.read_csv('../data/khamanon_multiindex_timeseries.csv')

df = df[['date', 'NDVI', 'NDWI', 'NBR', 'MNDWI', 'BSI', 'NDTI']].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Average same-day duplicate images
df = df.groupby('date')[['NDVI','NDWI','NBR','MNDWI','BSI','NDTI']].mean().reset_index()
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['NDVI', 'NDWI', 'NBR'], how='all')

for col in ['NDVI','NDWI','NBR','MNDWI','BSI','NDTI']:
    df[col] = df[col].round(4)

df['month_num'] = df['date'].dt.month

print(f"Loaded {len(df)} image dates")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
df.to_csv('../data/multiindex_timeseries_clean.csv', index=False)

# ── 2. COMPUTE DELTA ────────────────────────────────────────────
df['NDVI_prev']  = df['NDVI'].shift(1)
df['NDWI_prev']  = df['NDWI'].shift(1)
df['NBR_prev']   = df['NBR'].shift(1)
df['BSI_prev']   = df['BSI'].shift(1)
df['MNDWI_prev'] = df['MNDWI'].shift(1)

df['dNDVI']  = df['NDVI']  - df['NDVI_prev']
df['dNDWI']  = df['NDWI']  - df['NDWI_prev']
df['dNBR']   = df['NBR']   - df['NBR_prev']
df['dBSI']   = df['BSI']   - df['BSI_prev']
df['dMNDWI'] = df['MNDWI'] - df['MNDWI_prev']

# ── 3. ROLLING PEAK & SIGN CHANGE ───────────────────────────────
df['NDVI_30d_peak'] = df['NDVI'].rolling(window=8, min_periods=2).max().shift(1)
df['NDVI_drop']     = df['NDVI'] - df['NDVI_30d_peak']
df['BSI_30d_min']   = df['BSI'].rolling(window=8, min_periods=2).min().shift(1)
df['BSI_rise']      = df['BSI'] - df['BSI_30d_min']
df['NBR_sign']      = np.sign(df['NBR'])
df['NBR_sign_prev'] = df['NBR_sign'].shift(1)

# ── 4. EVENT DETECTION ──────────────────────────────────────────
events = []

for i, row in df.iterrows():
    if pd.isna(row['dNDVI']):
        continue

    date_str = row['date'].strftime('%Y-%m-%d')
    detected = []

    # HARVESTING — 30-day rolling peak drop + BSI rise
    if (not pd.isna(row['NDVI_drop'])
            and row['NDVI_drop'] < -0.20
            and row['BSI_rise'] > 0.05
            and row['NDVI'] < 0.45
            and row['month_num'] in [3, 4, 5, 10, 11]):
        detected.append({
            'date': date_str,
            'event': 'HARVESTING',
            'confidence': 'HIGH' if row['NDVI_drop'] < -0.30 else 'MEDIUM',
            'signal': (f"NDVI={row['NDVI']:.3f} "
                       f"(30d peak={row['NDVI_30d_peak']:.3f}, "
                       f"drop={row['NDVI_drop']:.3f})"),
            'NDVI_after': row['NDVI'],
            'note': 'Crop cut — NDVI fell below 30-day peak'
        })

    # STUBBLE BURNING — NBR crosses from positive to near-zero or negative
    if (row['NBR'] < 0.03
            and row['NBR_sign_prev'] > 0
            and row['NDVI'] < 0.30
            and row['month_num'] in [4, 5, 6, 10, 11, 12]):
        detected.append({
            'date': date_str,
            'event': 'STUBBLE_BURNING',
            'confidence': 'HIGH' if row['NBR'] < 0.0 else 'MEDIUM',
            'signal': (f"NBR={row['NBR']:.3f} "
                       f"(was {row['NBR_prev']:.3f}) — crossed zero"),
            'NDVI_after': row['NDVI'],
            'note': 'Burn scar — NBR crossed from positive to negative'
        })

    # FIELD FLOODING — NDWI and MNDWI both jump (rice season only)
    if (row['dNDWI'] > 0.18
            and row['dMNDWI'] > 0.10
            and row['month_num'] in [5, 6, 7, 8, 9]):
        detected.append({
            'date': date_str,
            'event': 'FIELD_FLOODING',
            'confidence': 'HIGH' if row['dNDWI'] > 0.25 else 'MEDIUM',
            'signal': f"dNDWI=+{row['dNDWI']:.3f} | dMNDWI=+{row['dMNDWI']:.3f}",
            'NDVI_after': row['NDVI'],
            'note': 'Standing water — rice puddling or field irrigation'
        })

    # PLOUGHING — BSI rises while NDVI stays low
    if (row['dBSI'] > 0.10
            and row['NDVI'] < 0.25
            and abs(row['dNDVI']) < 0.10
            and row['month_num'] in [4, 5, 6, 11]):
        detected.append({
            'date': date_str,
            'event': 'PLOUGHING',
            'confidence': 'MEDIUM',
            'signal': f"dBSI=+{row['dBSI']:.3f} | NDVI={row['NDVI']:.3f}",
            'NDVI_after': row['NDVI'],
            'note': 'Probable tillage — soil texture shift, no crop change'
        })

    # RICE TRANSPLANTING — NDVI rises from near-zero in Kharif season
    if (row['dNDVI'] > 0.10
            and row['NDVI_prev'] < 0.30
            and row['month_num'] in [6, 7, 8, 9]):
        detected.append({
            'date': date_str,
            'event': 'RICE_TRANSPLANTING',
            'confidence': 'HIGH' if row['dNDVI'] > 0.20 else 'MEDIUM',
            'signal': f"dNDVI=+{row['dNDVI']:.3f} | NDWI={row['NDWI']:.3f}",
            'NDVI_after': row['NDVI'],
            'note': 'Green shoots over fields — rice establishment confirmed'
        })

    events.extend(detected)

# ── 5. CONSOLIDATE — keep first detection per event per 25-30 days
def consolidate_events(events_df, cooldown):
    """
    After a harvest or burning event fires, suppress the same
    event type for N days. This turns 8 daily harvest detections
    into 1 clean event — the first day of the harvest window.
    """
    if len(events_df) == 0:
        return events_df

    consolidated = []
    last_date = {}   # last kept date per event type

    for _, row in events_df.iterrows():
        etype     = row['event']
        edate     = pd.to_datetime(row['date'])
        n_days    = cooldown.get(etype, 20)

        if etype not in last_date:
            consolidated.append(row)
            last_date[etype] = edate
        else:
            if (edate - last_date[etype]).days > n_days:
                consolidated.append(row)
                last_date[etype] = edate

    return pd.DataFrame(consolidated).reset_index(drop=True)


# Cooldown windows (days) — how long after one event before
# the same type can fire again
COOLDOWN = {
    'HARVESTING':        25,
    'STUBBLE_BURNING':   30,
    'FIELD_FLOODING':    15,
    'PLOUGHING':         10,
    'RICE_TRANSPLANTING':30,
}

# ── 6. SAVE EVENTS ──────────────────────────────────────────────
raw_df = pd.DataFrame(events)

if len(raw_df) > 0:
    raw_df = raw_df.sort_values('date').reset_index(drop=True)
    events_df = consolidate_events(raw_df, COOLDOWN)

    events_df.to_csv('../data/field_events.csv', index=False)

    print(f"\n{'='*50}")
    print(f"RAW DETECTIONS  : {len(raw_df)}")
    print(f"AFTER COOLDOWN  : {len(events_df)}  (these are the real events)")
    print(f"{'='*50}")
    print(events_df[['date','event','confidence','signal']].to_string())

    print(f"\n{'='*50}")
    print("EVENTS BY TYPE")
    print(f"{'='*50}")
    print(events_df['event'].value_counts().to_string())

else:
    print("\nNo events detected")
    events_df = pd.DataFrame(
        columns=['date','event','confidence','signal','NDVI_after','note'])
    events_df.to_csv('../data/field_events.csv', index=False)

# ── 7. SEASONAL PROFILE ─────────────────────────────────────────
print(f"\n{'='*50}")
print("SEASONAL NDVI PROFILE (monthly mean)")
print(f"{'='*50}")
df['month'] = df['date'].dt.to_period('M')
monthly = df.groupby('month')[['NDVI','NDWI','NBR','BSI']].mean().round(3)
print(monthly.to_string())

# ── 8. SAVE STATUS FOR DASHBOARD ────────────────────────────────
latest  = df.iloc[-1]
status  = {
    'last_processed' : datetime.now().strftime('%Y-%m-%d %H:%M'),
    'latest_date'    : latest['date'].strftime('%Y-%m-%d'),
    'latest_NDVI'    : float(latest['NDVI']),
    'latest_NDWI'    : float(latest['NDWI']),
    'latest_NBR'     : float(latest['NBR']),
    'latest_BSI'     : float(latest['BSI']),
    'total_images'   : len(df),
    'total_events'   : len(events_df),
    'current_phase'  : 'POST_HARVEST_FALLOW'
}
with open('../data/field_ops_status.json', 'w') as f:
    json.dump(status, f, indent=2)

print(f"\nStatus saved to field_ops_status.json")
print(f"Done.")