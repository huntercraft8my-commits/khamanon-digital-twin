# -*- coding: utf-8 -*-
# ============================================
# DIGITAL TWIN - KHAMANON BLOCK
# Dashboard v2.0 — Professional Redesign
# ============================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback
import json
import os
import math
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import timedelta
import subprocess
import atexit
import sys
from functools import lru_cache
import time

print("=" * 55)
print("  DASHBOARD v2.0 — PROFESSIONAL REDESIGN")
print("  Khamanon Block Digital Twin")
print("=" * 55)

base = os.path.dirname(os.path.abspath(__file__))

# ============================================
# DESIGN TOKENS
# ============================================

COLORS = {
    'primary'    : '#0e0e24',
    'secondary'  : '#16163a',
    'card'       : '#16163a',
    'accent'     : '#00d4aa',
    'accent2'    : '#e91e8c',
    'warning'    : '#ff8c42',
    'danger'     : '#ef4444',
    'success'    : '#10b981',
    'text'       : '#ffffff',
    'text_muted' : '#5a6a8a',
    'border'     : 'rgba(0, 212, 170, 0.12)',
    'grid'       : 'rgba(0, 212, 170, 0.04)',
    'chart_bg'   : 'rgba(0,0,0,0)',
}

CARD_STYLE = {
    'backgroundColor' : 'rgba(22, 22, 58, 0.78)',
    'padding'         : '28px',
    'position'        : 'relative',
    'border'          : '1px solid rgba(0, 212, 170, 0.12)',
    'borderRadius'    : '12px',
    'backdropFilter'  : 'blur(14px)',
}

# Chart colour palette (Image 4 ref: hot-pink → teal gradient palette)
CHART_COLORS = ['#00d4aa', '#e91e8c', '#ff8c42', '#a78bfa', '#fbbf24', '#38bdf8']

TEXT_STYLE = {
    'color'     : COLORS['text'],
    'fontFamily': 'Inter, -apple-system, Segoe UI, sans-serif'
}

def _tab_bg(img):
    """Return a backgroundImage style that layers a dark veil over img."""
    return {
        'backgroundImage' : f"linear-gradient(rgba(4,7,22,0.70),rgba(4,7,22,0.70)),url('/assets/{img}')",
        'backgroundSize'  : 'cover',
        'backgroundPosition': 'center',
        'backgroundAttachment': 'fixed',
        'backgroundRepeat': 'no-repeat',
    }

# ============================================
# FIELD OPS CONSTANTS
# ============================================

EV_COLOR = {
    'HARVESTING'        : '#f59e0b',
    'STUBBLE_BURNING'   : '#ef4444',
    'RICE_TRANSPLANTING': '#00d4aa',
    'FIELD_FLOODING'    : '#3b82f6',
    'PLOUGHING'         : '#f97316',
}

EV_LABEL = {
    'HARVESTING'        : 'Harvest',
    'STUBBLE_BURNING'   : 'Burning',
    'RICE_TRANSPLANTING': 'Rice Transplant',
    'FIELD_FLOODING'    : 'Flooding',
    'PLOUGHING'         : 'Ploughing',
}

# ============================================
# FIELD OPS ICONS & CONFIG
# ============================================

EV_ICON = {
    'HARVESTING'        : '🌾',
    'STUBBLE_BURNING'   : '🔥',
    'RICE_TRANSPLANTING': '🌱',
    'FIELD_FLOODING'    : '💧',
    'PLOUGHING'         : '🚜',
}

EV_YPOS = {
    'HARVESTING'        : 3,
    'STUBBLE_BURNING'   : 2,
    'RICE_TRANSPLANTING': 1,
    'FIELD_FLOODING'    : 0,
    'PLOUGHING'         : 0,
}

# ============================================
# DATA LOADER — returns 14 items
# 60-second TTL cache. All callbacks within a 60s window share one read.
# Reduces 70 file reads per 30s down to 14 file reads per 60s — 10x speedup.
# ============================================

_CACHE_TTL_SECONDS = 60

def _ttl_hash():
    """Returns an integer that changes every _CACHE_TTL_SECONDS. Used as cache key."""
    return int(time.time() / _CACHE_TTL_SECONDS)

@lru_cache(maxsize=1)
def _load_all_cached(_ttl_key):
    """Cached file reads. _ttl_key is ignored by the function body but causes
    lru_cache to invalidate when the TTL window changes."""
    soil = pd.read_csv(os.path.join(base,'..','data','soil_data_validated.csv'))
    grid = pd.read_csv(os.path.join(base,'..','data','real_prediction_grid.csv'))
    ndvi = pd.read_csv(os.path.join(base,'..','data','ndvi_processed.csv'))
    val  = pd.read_csv(os.path.join(base,'..','data','model_validation_real.csv'))

    st_path = os.path.join(base,'..','data','last_update.json')
    st = json.load(open(st_path)) if os.path.exists(st_path) else {
        'last_run':'Never','sentinel2_date':'Unknown',
        'ndvi_mean':0,'images_used':0,'status':'No data','alerts':[]
    }

    wx_path = os.path.join(base,'..','data','current_weather.json')
    wx = json.load(open(wx_path)) if os.path.exists(wx_path) else {
        'temperature':35,'humidity':45,'description':'N/A',
        'wind_speed':3,'rain_1h':0,'advisories':[]
    }

    adv_path = os.path.join(base,'..','data','block_advisories.csv')
    adv = pd.read_csv(adv_path) if os.path.exists(adv_path) else pd.DataFrame()

    risk_path = os.path.join(base,'..','data','point_risk_scores.csv')
    risk = pd.read_csv(risk_path) if os.path.exists(risk_path) else pd.DataFrame()

    shap_path = os.path.join(base,'..','data','shap_importance.csv')
    shap = pd.read_csv(shap_path, index_col='soil_property') if os.path.exists(shap_path) else pd.DataFrame()

    hist_path = os.path.join(base,'..','data','update_history.csv')
    hist = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame()

    sat_log_path = os.path.join(base,'..','data','satellite_log.csv')
    sat_log = pd.read_csv(sat_log_path) if os.path.exists(sat_log_path) else pd.DataFrame()

    lulc_ndvi_path = os.path.join(base,'..','data','ndvi_by_lulc_class.csv')
    lulc_ndvi = pd.read_csv(lulc_ndvi_path) if os.path.exists(lulc_ndvi_path) else pd.DataFrame()

    corr_r_path = os.path.join(base,'..','data','soil_spectral_corr_r.csv')
    corr_r = pd.read_csv(corr_r_path, index_col=0) if os.path.exists(corr_r_path) else pd.DataFrame()

    residuals_path = os.path.join(base,'..','data','soil_spectral_residuals.csv')
    residuals = pd.read_csv(residuals_path) if os.path.exists(residuals_path) else pd.DataFrame()

    return (soil, grid, ndvi, val, st, wx, adv, risk, shap, hist,
            sat_log, lulc_ndvi, corr_r, residuals)


def load_all():
    """Public API — unchanged signature, returns 14 items. Internally cached."""
    return _load_all_cached(_ttl_hash())

# ============================================
# HELPER: METRIC CARD
# ============================================

def metric_card(title, value, unit, subtitle='', color=None):
    c = color or COLORS['accent']
    is_alert = (c == COLORS['danger'])
    val_color = COLORS['danger'] if is_alert else '#ffffff'
    return html.Div([
        # Label — Inter 500, 10px, all-caps, muted
        html.Div(title, style={
            'fontFamily'   : 'Inter, sans-serif',
            'fontSize'     : '10px',
            'fontWeight'   : '500',
            'textTransform': 'uppercase',
            'letterSpacing': '0.8px',
            'color'        : COLORS['text_muted'],
            'marginBottom' : '12px',
        }),
        # Value — Inter 700, 32px, high contrast, no glow
        html.Div([
            html.Span(str(value), style={
                'fontFamily' : 'Inter, sans-serif',
                'fontSize'   : '32px',
                'fontWeight' : '700',
                'color'      : val_color,
                'letterSpacing': '-0.5px',
                'lineHeight' : '1',
            }),
            html.Span((' ' + unit) if unit else '', style={
                'fontFamily' : 'Inter, sans-serif',
                'fontSize'   : '13px',
                'fontWeight' : '400',
                'color'      : COLORS['text_muted'],
                'marginLeft' : '5px',
            }),
        ], style={'lineHeight': '1'}),
        # Subtitle — Inter 400, 12px
        html.Div(subtitle, style={
            'fontFamily' : 'Inter, sans-serif',
            'fontSize'   : '12px',
            'fontWeight' : '400',
            'color'      : COLORS['text_muted'],
            'marginTop'  : '8px',
        }) if subtitle else None,
    ], className='kpi-card', style={**CARD_STYLE})


# ============================================
# HELPER: ALERT BADGE
# ============================================

def alert_badge(text, severity):
    colors = {
        'CRITICAL': ('#ef4444','#2d1b1b'),
        'WARNING' : ('#f59e0b','#2d2518'),
        'INFO'    : ('#3b82f6','#1b2340'),
        'OK'      : ('#10b981','#1b2d25')
    }
    icons = {'CRITICAL':'✕ ','WARNING':'⚠ ','INFO':'ℹ ','OK':'✓ '}
    fc, bg = colors.get(severity, ('#94a3b8','#1e2535'))
    icon   = icons.get(severity, '· ')
    return html.Div([
        html.Span(icon, style={'fontFamily':'JetBrains Mono, monospace','fontWeight':'700','marginRight':'4px'}),
        html.Span(text, style={'fontFamily':'Inter, sans-serif'})
    ], style={
        'backgroundColor':bg,'color':fc,'border':f'1px solid {fc}22',
        'borderLeft':f'3px solid {fc}',
        'padding':'8px 12px','fontSize':'12px',
        'marginBottom':'6px','fontWeight':'500',
    })


# ============================================
# HELPER: SATELLITE STATUS BADGE
# ============================================

def _sat_status_badge(status):
    cfg = {
        'ACCEPTED': (COLORS['success'], '#1b2d25', '✓ ACCEPTED'),
        'REJECTED': (COLORS['danger'],  '#2d1b1b', '✗ REJECTED'),
    }
    fc, bg, label = cfg.get(status, (COLORS['text_muted'], COLORS['card'], status))
    return html.Span(label, style={
        'backgroundColor':bg,'color':fc,'border':f'1px solid {fc}',
        'borderRadius':'4px','padding':'2px 8px','fontSize':'11px','fontWeight':'600',
    })


# ============================================
# HELPER: SOIL TIER ROW
# ============================================

def _tier_row(grid, col, label, tiers):
    if col not in grid.columns:
        return html.Div()
    n    = len(grid)
    vals = grid[col]
    segments, pills = [], []
    for name, cond_fn, color in tiers:
        pct = round(cond_fn(vals).sum() / n * 100, 1)
        if pct > 0:
            segments.append(html.Div(style={
                'width':f'{pct}%','height':'9px',
                'backgroundColor':color,'display':'inline-block',
            }, title=f'{name}: {pct}%'))
        pills.append(html.Span([
            html.Span(style={
                'display':'inline-block','width':'7px','height':'7px',
                'borderRadius':'2px','backgroundColor':color,
                'marginRight':'3px','verticalAlign':'middle',
            }),
            html.Span(f'{name}  {pct}%', style={
                'fontSize':'10px','color':COLORS['text_muted'],'marginRight':'10px',
            })
        ]))
    return html.Div([
        html.Div([
            html.Span(label, style={
                'color':COLORS['text_muted'],'fontSize':'11px','fontWeight':'600',
                'width':'115px','flexShrink':'0','display':'inline-block',
            }),
            html.Div(segments, style={
                'flex':'1','height':'9px','borderRadius':'5px',
                'overflow':'hidden','backgroundColor':COLORS['border'],'display':'flex',
            }),
        ], style={'display':'flex','alignItems':'center','marginBottom':'4px'}),
        html.Div(pills, style={'paddingLeft':'115px','marginBottom':'12px'}),
    ])


# ============================================
# TAB STYLE HELPERS
# ============================================

def tab_style():
    return {
        'backgroundColor' : 'transparent',
        'color'           : '#64748b',
        'border'          : 'none',
        'padding'         : '13px 24px',
        'fontFamily'      : 'Rajdhani, sans-serif',
        'fontSize'        : '12px',
        'fontWeight'      : '700',
        'textTransform'   : 'uppercase',
        'letterSpacing'   : '1.4px',
        'borderBottom'    : '2px solid transparent',
        'transition'      : 'color 0.18s ease, border-color 0.18s ease',
    }

def tab_selected_style():
    return {
        'backgroundColor' : 'rgba(13,16,45,0.70)',
        'color'           : '#00d4aa',          # teal only per spec
        'border'          : 'none',
        'borderBottom'    : '2px solid #00d4aa',
        'padding'         : '13px 24px',
        'fontFamily'      : 'Rajdhani, sans-serif',
        'fontSize'        : '12px',
        'fontWeight'      : '700',
        'textTransform'   : 'uppercase',
        'letterSpacing'   : '1.4px',
        'textShadow'      : '0 0 10px rgba(0,212,170,0.35)',
    }


# ============================================
# HELPER: BUILD LULC MAP
# ============================================

def _build_lulc_map(base_path):
    rabi_csv   = os.path.join(base_path,'..','data','lulc_rabi_map.csv')
    kharif_csv = os.path.join(base_path,'..','data','lulc_kharif_map.csv')
    fig = go.Figure()
    rabi_loaded = kharif_loaded = False

    if os.path.exists(rabi_csv):
        rabi_df = pd.read_csv(rabi_csv)
        for cls_name, grp in rabi_df.groupby('class_name'):
            color = grp['color'].iloc[0]
            fig.add_trace(go.Scattermap(
                lat=grp['lat'], lon=grp['lon'], mode='markers',
                name=f'{cls_name} (Rabi)', legendgroup='rabi',
                legendgrouptitle_text='Rabi 2025-26',
                marker=dict(size=7, color=color, opacity=0.85),
                hovertemplate=f'<b>{cls_name}</b><br>Rabi 2025-26<br>Lat: %{{lat:.4f}}<br>Lon: %{{lon:.4f}}<extra></extra>',
                visible=True
            ))
        rabi_loaded = True

    if os.path.exists(kharif_csv):
        kharif_df = pd.read_csv(kharif_csv)
        for cls_name, grp in kharif_df.groupby('class_name'):
            color = grp['color'].iloc[0]
            fig.add_trace(go.Scattermap(
                lat=grp['lat'], lon=grp['lon'], mode='markers',
                name=f'{cls_name} (Kharif)', legendgroup='kharif',
                legendgrouptitle_text='Kharif 2025',
                marker=dict(size=7, color=color, opacity=0.85),
                hovertemplate=f'<b>{cls_name}</b><br>Kharif 2025<br>Lat: %{{lat:.4f}}<br>Lon: %{{lon:.4f}}<extra></extra>',
                visible='legendonly'
            ))
        kharif_loaded = True

    if not rabi_loaded and not kharif_loaded:
        fig.add_trace(go.Scattermap(
            lat=[30.795], lon=[76.352], mode='markers+text',
            text=['Run lulc_map_processor.py to show LULC map'],
            textposition='top center',
            marker=dict(size=12, color='#00d4aa'), showlegend=False
        ))

    fig.update_layout(
        map=dict(style='dark', center=dict(lat=30.795, lon=76.352), zoom=10),
        paper_bgcolor=COLORS['card'], height=520, margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(
            bgcolor='rgba(30,37,53,0.9)', bordercolor=COLORS['border'], borderwidth=1,
            font=dict(color=COLORS['text'], size=11), itemsizing='constant',
            groupclick='toggleitem', x=0.01, y=0.99, xanchor='left', yanchor='top'
        ),
        uirevision='constant'
    )
    return fig


# ============================================
# APP INITIALIZATION
# ============================================

_lulc_map_cache = _build_lulc_map(base)
app = Dash(__name__, suppress_callback_exceptions=True)


# ============================================
# BACKGROUND SCHEDULER — 24-HOUR AUTO-UPDATE
# ============================================

def _run_updater_job():
    updater = os.path.join(base, 'realtime_updater.py')
    if not os.path.exists(updater):
        print("[Scheduler] realtime_updater.py not found — skipping")
        return
    try:
        print(f"[Scheduler] Starting pipeline run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = subprocess.run(
            [sys.executable, updater], capture_output=True, text=True, timeout=480
        )
        if result.returncode == 0:
            print("[Scheduler] Pipeline completed successfully")
        else:
            print(f"[Scheduler] Exit code: {result.returncode}")
        if result.stdout:
            print("[Scheduler] stdout:", result.stdout[-600:])
        if result.stderr:
            print("[Scheduler] stderr:", result.stderr[-300:])
    except subprocess.TimeoutExpired:
        print("[Scheduler] Timed out after 8 minutes")
    except Exception as e:
        print(f"[Scheduler] Error: {e}")

_scheduler = BackgroundScheduler(job_defaults={'max_instances':1,'misfire_grace_time':3600})
_scheduler.add_job(
    _run_updater_job, trigger='interval', hours=24, id='daily_pipeline',
    next_run_time=datetime.now() + timedelta(minutes=3)
)
_scheduler.start()
atexit.register(lambda: _scheduler.shutdown(wait=False))
print("[Scheduler] Started — first run in 3min, then every 24h")


# ============================================
# APP LAYOUT
# ============================================

# ============================================
# APP LAYOUT
# ============================================

app.layout = html.Div([
    # ── Earth background layers ───────────────────────────────────────────────
    html.Div(id='earth-bg'),
    html.Div(id='earth-overlay'),

    dcc.Interval(id='refresh', interval=300*1000, n_intervals=0),
    dcc.Store(id='sat-log-open', data=False),

    # ── Sticky Console Header ─────────────────────────────────────────────────
    html.Div([
        # Main header bar
        html.Div([
            html.Div([
                html.Div([
                    html.Span('Khamanon ', style={
                        'color'      : '#ffffff',
                        'fontSize'   : '16px',
                        'fontWeight' : '700',
                        'fontFamily' : 'Inter, sans-serif',
                        'letterSpacing': '-0.3px',
                    }),
                    html.Span('Digital Twin', style={
                        'color'      : COLORS['accent'],
                        'fontSize'   : '16px',
                        'fontWeight' : '400',
                        'fontFamily' : 'Inter, sans-serif',
                        'letterSpacing': '-0.3px',
                    }),
                ], style={'display':'flex','alignItems':'center'}),
                html.Div(
                    'Fatehgarh Sahib, Punjab',
                    style={
                        'color'      : COLORS['text_muted'],
                        'fontSize'   : '12px',
                        'marginTop'  : '3px',
                        'fontFamily' : 'Inter, sans-serif',
                        'fontWeight' : '400',
                        'letterSpacing': '0.2px',
                    }
                )
            ], style={'flex':'1'}),
            html.Div(id='nav-status', style={'textAlign':'right'})
        ], style={
            'backgroundColor': 'rgba(8,10,25,0.92)',
            'padding'        : '14px 30px',
            'display'        : 'flex',
            'alignItems'     : 'center',
            'borderBottom'   : '1px solid rgba(0,212,170,0.07)',
        }),

    ], style={'position':'sticky','top':'0','zIndex':'1000','isolation':'isolate'}, className='sticky-header'),

    html.Div([
        dcc.Tabs(id='tabs', value='overview', children=[
            dcc.Tab(label='Overview', value='overview',
                    style=tab_style(), selected_style=tab_selected_style()),

            dcc.Tab(label='🌾 Land & Crops', value='landcrops',
                    style={**tab_style(),'borderTop':f"2px solid {COLORS['success']}"},
                    selected_style={**tab_selected_style(),'color':COLORS['success'],
                                    'borderBottom':f"2px solid {COLORS['success']}"}),

            dcc.Tab(label='🛰 Field Ops', value='fieldops',
                    style={**tab_style(),'borderTop':f"2px solid {COLORS['accent2']}"},
                    selected_style={**tab_selected_style(),'color':COLORS['accent2'],
                                    'borderBottom':f"2px solid {COLORS['accent2']}"}),

            dcc.Tab(label='🧪 Soil Analysis', value='soilanalysis',
                    style={**tab_style(),'borderTop':f"2px solid #f59e0b"},
                    selected_style={**tab_selected_style(),'color':'#f59e0b',
                                    'borderBottom':f"2px solid #f59e0b"}),

            dcc.Tab(label='🔬 Soil-Satellite', value='correlation',
                    style={**tab_style(),'borderTop':f"2px solid {COLORS['accent2']}"},
                    selected_style={**tab_selected_style(),'color':COLORS['accent2'],
                                    'borderBottom':f"2px solid {COLORS['accent2']}"}),

            dcc.Tab(label='⚠ Risk & Advisory', value='riskadvisory',
                    style={**tab_style(),'borderTop':f"2px solid {COLORS['danger']}"},
                    selected_style={**tab_selected_style(),'color':COLORS['danger'],
                                    'borderBottom':f"2px solid {COLORS['danger']}"}),

            dcc.Tab(label='📊 Model Validation', value='analytics',
                    style=tab_style(), selected_style=tab_selected_style()),
        ], style={
            'backgroundColor': 'rgba(8,10,25,0.95)',
            'backdropFilter' : 'blur(20px)',
            'WebkitBackdropFilter': 'blur(20px)',
            'borderBottom'   : '1px solid rgba(255,255,255,0.06)',
            'display'        : 'flex',
            'flexDirection'  : 'row',
            'position'       : 'relative',  # must be positioned for z-index to work
            'zIndex'         : '200',        # above the fixed Earth layers (z-index 0 & 1)
        })
    ], style={'position':'relative','zIndex':'200'}),

    html.Div(
        id='page-content',
        style={'backgroundColor':'transparent','minHeight':'90vh','padding':'28px 32px',
               'position':'relative','zIndex':'2',**TEXT_STYLE}
    )
], style={'background':'linear-gradient(160deg, #0a0d1a 0%, #0d1033 50%, #0f0a2e 100%)',
          'minHeight':'100vh',**TEXT_STYLE})
# ============================================
# NAV STATUS CALLBACK
# ============================================

@callback(Output('nav-status','children'), Input('refresh','n_intervals'))
def update_nav(n):
    _, _, _, _, st, _, _, _, _, _, _, _, _, _ = load_all()
    is_success   = st.get('status') == 'SUCCESS'
    pulse_cls    = 'pulse-node pulse-node-success' if is_success else 'pulse-node pulse-node-warning'
    status_color = COLORS['success'] if is_success else COLORS['warning']
    return html.Div([
        html.Div([
            html.Span(className=pulse_cls),
            html.Span(st.get('status', 'Unknown'), style={
                'color'     : status_color,
                'fontSize'  : '12px',
                'fontWeight': '600',
                'fontFamily': 'Inter, sans-serif',
                'letterSpacing': '0.5px',
                'textTransform': 'uppercase',
            }),
            html.Span(
                f"  ·  Orbital {st.get('sentinel2_date','—')}",
                style={
                    'color'     : COLORS['text_muted'],
                    'fontSize'  : '11px',
                    'fontFamily': 'Inter, sans-serif',
                    'marginLeft': '8px',
                }
            ),
        ], style={'display':'flex','alignItems':'center'}),
    ], style={'textAlign':'right'})


# ============================================
# MAIN PAGE CALLBACK
# ============================================

# Update your existing decorator and function signature like this:
@app.callback(
    Output('page-content', 'children'),
    Input('tabs', 'value'),
    Input('refresh', 'n_intervals'),
    State('sat-log-open', 'data')
)
def render_page(tab, n, sat_open=False):
    # Your internal load_all() unpacking remains untouched here (exactly 14 items)
    # Mandatory Unpack Target Verification: Exactly 14 mapping structures defined
    (soil, grid, ndvi_df, val, st, wx, adv, risk, shap, hist, sat_log, lulc_ndvi, corr_r, residuals) = load_all()
# Seasonal advisory filter — suppress R14 (NDVI crop stress) during
# Rabi post-harvest fallow: April(4) May(5) June(6)
# NDVI of 0.246 in these months is normal bare soil, not active crop stress
    _month = datetime.now().month
    if not adv.empty and _month in [4, 5, 6]:
        adv = adv[~adv['rule_id'].astype(str).str.contains('R14')]
    ndvi_mean = st.get('ndvi_mean', 0)
    s2_date   = st.get('sentinel2_date', '—')
    zones     = [c for c in ndvi_df.columns if c != 'month']

    # ==========================================
    # TAB 1 — OVERVIEW
    # ==========================================
    if tab == 'overview':

        ndvi_col = COLORS['danger'] if ndvi_mean < 0.35 else COLORS['success']
        temp_col = COLORS['danger'] if wx.get('temperature', 0) > 35 else COLORS['success']

        _T   = wx.get('temperature', 35)
        _RH  = wx.get('humidity', 45)
        _es  = 0.61078 * math.exp((17.27 * _T) / (_T + 237.3))
        _ea  = _es * (_RH / 100)
        vpd  = round(_es - _ea, 2)
        vpd_label = (
            'Low - good for crops'     if vpd < 0.5
            else 'Moderate - monitor'      if vpd < 1.0
            else 'High - stomatal stress'  if vpd < 1.5
            else 'Critical - growth halted'
        )
        vpd_col = (
            COLORS['success'] if vpd < 0.5
            else COLORS['accent']  if vpd < 1.0
            else COLORS['warning'] if vpd < 1.5
            else COLORS['danger']
        )

        def kpi_sm(label, value, unit, subtitle='', alert=False):
            """Compact KPI card for the Overview hero zone."""
            val_col = '#ef4444' if alert else '#ffffff'
            return html.Div([
                html.Div(label, style={
                    'fontFamily':'Inter, sans-serif','fontSize':'10px','fontWeight':'500',
                    'textTransform':'uppercase','letterSpacing':'1px',
                    'color':'rgba(255,255,255,0.45)','marginBottom':'10px',
                }),
                html.Div([
                    html.Span(str(value), style={
                        'fontFamily':'Inter, sans-serif','fontSize':'26px',
                        'fontWeight':'700','color':val_col,'letterSpacing':'-0.5px',
                    }),
                    html.Span(f' {unit}' if unit else '', style={
                        'fontFamily':'Inter, sans-serif','fontSize':'12px',
                        'fontWeight':'400','color':'rgba(255,255,255,0.4)','marginLeft':'3px',
                    }),
                ], style={'lineHeight':'1'}),
                html.Div(subtitle, style={
                    'fontFamily':'Inter, sans-serif','fontSize':'11px','fontWeight':'400',
                    'color':'rgba(255,255,255,0.35)','marginTop':'6px',
                }) if subtitle else None,
            ], className='kpi-card-sm')

        cards = html.Div([
            # Row 1 — Satellite data
            html.Div([
                kpi_sm('Block NDVI',        round(ndvi_mean,3), '', '', ndvi_col == COLORS['danger']),
                kpi_sm('Sentinel-2 Images', st.get('images_used',0), 'used', 'Last 30 days'),
            ], className='overview-kpi-row'),

            # Climate section label
            html.Div('Climate Conditions — Study Area', className='climate-label'),

            # Row 2 — Climate
            html.Div([
                kpi_sm('Temperature', wx.get('temperature','—'), '°C',
                       str(wx.get('description','Loading...')).title(), temp_col == COLORS['danger']),
                kpi_sm('Humidity',    wx.get('humidity','—'),     '%',  'Relative humidity'),
                kpi_sm('Atm. VPD',   vpd,                        'kPa', vpd_label,
                       vpd_col == COLORS['danger']),
            ], className='overview-kpi-row'),
        ], className='overview-kpi-container', style={'marginBottom':'32px'})

        
        fig_ndvi = go.Figure()
        # Use CHART_COLORS palette (Image 4 ref: pink→teal gradient palette)
        zone_list = list(zones)
        # Seasonal background bands
        for x0, x1, lbl, sc in [
            ('Jan-2025','May-2025','Rabi 2024-25','rgba(0,212,170,0.05)'),
            ('Jun-2025','Oct-2025','Kharif 2025','rgba(233,30,140,0.05)'),
            ('Nov-2025','May-2026','Rabi 2025-26','rgba(0,212,170,0.05)'),
        ]:
            fig_ndvi.add_vrect(x0=x0, x1=x1, fillcolor=sc, line_width=0,
                               annotation_text=lbl,
                               annotation_font_color=COLORS['text_muted'],
                               annotation_font_size=9)
        for i, zone in enumerate(zone_list):
            zc = CHART_COLORS[i % len(CHART_COLORS)]
            r  = int(zc[1:3],16); g = int(zc[3:5],16); b = int(zc[5:7],16)
            fig_ndvi.add_trace(go.Scatter(
                x=ndvi_df['month'], y=ndvi_df[zone], name=zone,
                mode='lines+markers',
                fill='tozeroy',
                fillcolor=f'rgba({r},{g},{b},0.10)',
                line=dict(color=zc, width=2.5),
                marker=dict(size=7, symbol='circle',
                            line=dict(color=zc, width=1.5),
                            color=f'rgba({r},{g},{b},0.25)'),
                hovertemplate=f'<b>{zone}</b><br>%{{x}}<br>NDVI: %{{y:.3f}}<extra></extra>',
            ))
        fig_ndvi.add_hline(y=0.40, line_dash='dash', line_color='rgba(239,68,68,0.5)',
                           annotation_text='Fallow Base', annotation_font_size=9,
                           annotation_font_color='#ef4444')
        fig_ndvi.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=40,r=20,t=30,b=70),
            legend=dict(orientation='h', y=-0.40, x=0,
                        font=dict(size=10, color=COLORS['text_muted']),
                        bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(tickangle=45, tickfont=dict(size=9, color=COLORS['text_muted']),
                       gridcolor='rgba(0,212,170,0.05)', zeroline=False),
            yaxis=dict(gridcolor='rgba(0,212,170,0.05)', zeroline=False,
                       tickfont=dict(size=9, color=COLORS['text_muted'])),
            title=dict(text='NDVI Time Series — Khamanon Block',
                       font=dict(size=13, color=COLORS['text'],
                                 family='Rajdhani, sans-serif')),
            hovermode='x unified',
            uirevision='ndvi',
        )
        # Build collapsible satellite log section
        sat_log_section = html.Div([
            html.Div([
                html.Div([
                    html.Span('🛰 Sentinel-2 Acquisition Log',
                              style={'color':COLORS['text'],'fontSize':'14px',
                                     'fontWeight':'600'}),
                    html.Span(
                        f'  ·  Rabi <25% cloud  |  Kharif <50% cloud'
                        f'  ·  {len(sat_log)} orbital passes',
                        style={'color':COLORS['text_muted'],'fontSize':'11px'}
                    ),
                ], style={'flex':'1'}),
                html.Button(
                    '▲ Collapse' if sat_open else f'▼ Show {len(sat_log)} passes',
                    id='sat-log-toggle-btn',
                    n_clicks=0,
                    style={
                        'backgroundColor':COLORS['secondary'],
                        'color':COLORS['accent'],
                        'border':f"1px solid {COLORS['border']}",
                        'borderRadius':'6px','padding':'4px 14px',
                        'fontSize':'11px','cursor':'pointer','fontWeight':'600'
                    }
                )
            ], style={'display':'flex','alignItems':'center',
                      'marginBottom':'14px' if sat_open else '0'}),

            html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th(h, style={
                            'color':COLORS['text_muted'],'fontSize':'11px',
                            'textTransform':'uppercase','letterSpacing':'1px',
                            'padding':'8px 14px','textAlign':'left',
                            'borderBottom':f"1px solid {COLORS['border']}"
                        }) for h in ['Date','Satellite','Cloud','Season',
                                     'Status','Action']
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(row['date'],
                                    style={'color':COLORS['text_muted'],
                                           'padding':'8px 14px','fontSize':'12px',
                                           'whiteSpace':'nowrap'}),
                            html.Td(row['satellite'],
                                    style={'color':COLORS['text'],
                                           'padding':'8px 14px','fontSize':'12px'}),
                            html.Td(html.Span(f"{row['cloud_pct']}%", style={
                                'color':(COLORS['danger']
                                         if row['cloud_pct'] > row['threshold']
                                         else COLORS['success']),
                                'fontWeight':'600','fontSize':'12px'
                            }), style={'padding':'8px 14px'}),
                            html.Td(row['season'],
                                    style={'color':COLORS['text_muted'],
                                           'padding':'8px 14px','fontSize':'12px'}),
                            html.Td(_sat_status_badge(row['status']),
                                    style={'padding':'8px 14px'}),
                            html.Td(row['action'],
                                    style={'color':COLORS['text_muted'],
                                           'padding':'8px 14px','fontSize':'11px'}),
                        ], style={
                            'borderBottom':f"1px solid {COLORS['border']}",
                            'backgroundColor':(
                                'rgba(239,68,68,0.04)'
                                if row['status']=='REJECTED' else 'transparent')
                        })
                        for _, row in
                        sat_log.sort_values('date',ascending=False).head(25).iterrows()
                    ] if not sat_log.empty else [
                        html.Tr([html.Td(
                            'Run realtime_updater.py to populate.',
                            colSpan=6,
                            style={'color':COLORS['text_muted'],
                                   'padding':'16px 14px','fontSize':'12px'}
                        )])
                    ])
                ], style={'width':'100%','borderCollapse':'collapse'})
            ], id='sat-log-body',
               style={'display':'block' if sat_open else 'none',
                      'marginTop':'14px'})

        ], style={**CARD_STYLE,'marginTop':'20px'})

        # Return layout configuration pointing to the new sat_log_section
        return html.Div([
            # Full-viewport Earth background (Hercules ref: space top, Earth bottom)
            html.Div(className='overview-earth-img'),
            html.Div(className='overview-earth-grad'),

            # Hero content — centered, floats in the dark upper zone
            html.Div([
                html.Div('Khamanon Digital Twin', style={
                    'fontFamily':'Inter, sans-serif','fontSize':'13px','fontWeight':'400',
                    'color':'rgba(255,255,255,0.35)','textTransform':'uppercase',
                    'letterSpacing':'3px','marginBottom':'6px','textAlign':'center',
                }),
                html.Div('Overview', style={
                    'fontFamily':'Inter, sans-serif','fontSize':'36px','fontWeight':'800',
                    'color':'#ffffff','letterSpacing':'-1px','textAlign':'center',
                    'marginBottom':'4px','lineHeight':'1.1',
                }),
                html.Div(f"Updated {st.get('last_run', 'Never')}", style={
                    'fontFamily':'Inter, sans-serif','fontSize':'12px','fontWeight':'400',
                    'color':'rgba(255,255,255,0.30)','textAlign':'center','marginBottom':'40px',
                }),
                cards,
            ], style={'paddingTop':'60px'}),

            # Satellite log sits below the hero, back on normal flow
            html.Div([sat_log_section], style={'marginTop':'24px'}),

        ], className='overview-wrapper')

    # ==========================================
    # TAB 2 — SOIL MAPS
    # ==========================================
    

    # ==========================================
    # TAB 3 — CROP MONITOR
    # ==========================================
    
    # ==========================================
    # TAB 4 — PAU ADVISORY
    # ==========================================

        sev_colors = {
            'CRITICAL':COLORS['danger'],'WARNING':COLORS['warning'],
            'INFO':'#3b82f6','OK':COLORS['success']
        }

        adv_cards = []
        if not adv.empty:
            for _, row in adv.iterrows():
                c = sev_colors.get(str(row.get('severity','INFO')),'#3b82f6')
                adv_cards.append(html.Div([
                    html.Div([
                        html.Span(f"[{row.get('rule_id','')}]",
                                  style={'backgroundColor':c+'22','color':c,'padding':'2px 8px',
                                         'borderRadius':'4px','fontSize':'11px','fontWeight':'700','marginRight':'10px'}),
                        html.Span(row.get('message',''),
                                  style={'color':c,'fontWeight':'600','fontSize':'13px'})
                    ], style={'marginBottom':'8px','display':'flex','alignItems':'center'}),
                    html.Div(row.get('action',''),
                             style={'color':COLORS['text_muted'],'fontSize':'12px','lineHeight':'1.6'})
                ], style={**CARD_STYLE,'marginBottom':'12px','borderLeft':f'3px solid {c}'}))

        source_card = html.Div([
            html.Div('Data Source', style={'color':COLORS['text'],'fontWeight':'600','fontSize':'13px','marginBottom':'10px'}),
            html.Div([
                html.Div('PAU Package of Practices Kharif 2025 (Vol. 42)',
                         style={'color':COLORS['text_muted'],'fontSize':'12px','marginBottom':'4px'}),
                html.Div('PAU Package of Practices Rabi 2025',
                         style={'color':COLORS['text_muted'],'fontSize':'12px','marginBottom':'4px'}),
                html.Div('Punjab Soil Health Card — NPK + Micronutrients',
                         style={'color':COLORS['text_muted'],'fontSize':'12px'})
            ])
        ], style={**CARD_STYLE,'marginBottom':'20px'})

        return html.Div([
            html.Div('PAU Advisory Engine', style={'fontSize':'20px','fontWeight':'700','color':COLORS['text'],'marginBottom':'6px'}),
            html.Div(
                f"Source: PAU Package of Practices 2025  ·  Punjab Soil Health Card  ·  {len(adv)} block-level advisories",
                style={'color':COLORS['text_muted'],'fontSize':'12px','marginBottom':'20px'}
            ),
            source_card, html.Div(adv_cards)
        ])

    # ==========================================
    # TAB 5 — FIELD POINTS
    # ==========================================
    
    # ==========================================
    # TAB 6 — ANALYTICS
    # ==========================================
    elif tab == 'riskadvisory':
        # SECTION A — Risk Assessment
        _RISK_CONTEXT = {
            'Soil Degradation' : ['Low OC (<0.4%) in 60%+ of fields — SOC replenishment priority',
                                  'Alkaline pH (>8.0) locks out micronutrients',
                                  'Stubble burning accelerating organic matter loss'],
            'Crop Failure'     : ['Nitrogen deficiency is the primary yield limiter',
                                  'High spatial variability in available N supply',
                                  'Recommend split N application per PAU Rabi schedule'],
            'Salinity Stress'  : ['EC safely <0.25 dS/m in most zones — low immediate risk',
                                  'Localised stress pockets near irrigation drains',
                                  'Annual EC monitoring advised for borderline parcels'],
            'Overall Risk'     : ['Soil health is the dominant constraint for Khamanon block',
                                  'OC improvement programme is the single highest-impact action',
                                  'PAU Package of Practices 2025 provides recommended inputs'],
        }
        _risk_rows = [
            ('Soil Degradation',  risk['degradation_risk'].mean()  if not risk.empty else 0),
            ('Crop Failure',      risk['crop_failure_risk'].mean() if not risk.empty else 0),
            ('Salinity Stress',   risk['salinity_risk'].mean()     if not risk.empty else 0),
            ('Overall Risk',      risk['overall_risk'].mean()      if not risk.empty else 0),
        ]
        risk_section = html.Div([
            html.Div('Risk Assessment', style={'fontSize':'16px','fontWeight':'700','color':'#ffffff','marginBottom':'4px'}),
            html.Div('Random Forest predicted scores · 19,290 grid points · 4 risk dimensions',
                     style={'color':'rgba(255,255,255,0.55)','fontSize':'12px','marginBottom':'16px'}),
            html.Div([
                # Left: risk bars (≈ half width)
                html.Div([
                    html.Div([
                        html.Div(r_name, style={
                            'color':'rgba(255,255,255,0.80)','fontSize':'13px',
                            'fontWeight':'600','marginBottom':'5px','fontFamily':'Inter',
                        }),
                        html.Div([
                            html.Div(style={
                                'height':'10px','borderRadius':'5px',
                                'backgroundColor':'rgba(255,255,255,0.10)',
                                'flex':'1','marginRight':'12px','overflow':'hidden',
                            }, children=[
                                html.Div(style={
                                    'height':'10px','width':f"{r_val}%",'borderRadius':'5px',
                                    'backgroundColor':(COLORS['danger'] if r_val > 70
                                                       else COLORS['warning'] if r_val > 40
                                                       else COLORS['success']),
                                })
                            ]),
                            html.Span(f"{r_val:.0f}/100", style={
                                'color':'#ffffff','fontSize':'13px','fontWeight':'700','minWidth':'52px',
                            }),
                        ], style={'display':'flex','alignItems':'center','marginBottom':'14px'}),
                    ]) for r_name, r_val in _risk_rows
                ], style={'flex':'1','minWidth':'0','maxWidth':'50%'}),
                # Right: interpretation bullets
                html.Div([
                    html.Div('Key Findings', style={
                        'color':'#ffffff','fontSize':'13px','fontWeight':'700',
                        'marginBottom':'12px','fontFamily':'Inter','letterSpacing':'0.3px',
                    }),
                    html.Div([
                        html.Div([
                            html.Div(r_name, style={
                                'color': (COLORS['danger'] if r_val > 70
                                          else COLORS['warning'] if r_val > 40
                                          else COLORS['success']),
                                'fontSize':'11px','fontWeight':'700','marginBottom':'4px','fontFamily':'Inter',
                            }),
                            html.Div([
                                html.Div(f'· {pt}', style={
                                    'color':'#e2e8f0','fontSize':'12px',
                                    'lineHeight':'1.55','fontFamily':'Inter','marginBottom':'2px',
                                }) for pt in _RISK_CONTEXT.get(r_name, [])
                            ], style={'marginBottom':'10px'}),
                        ]) for r_name, r_val in _risk_rows
                    ]),
                ], style={
                    'flex':'1','minWidth':'0','paddingLeft':'20px',
                    'borderLeft':'1px solid rgba(255,255,255,0.08)',
                }),
            ], style={'display':'flex','alignItems':'flex-start','gap':'0'}),
        ], style={**CARD_STYLE, 'marginBottom': '20px'})

        # SECTION B — PAU Advisory
        sev_colors = {
            'CRITICAL': COLORS['danger'], 'WARNING': COLORS['warning'],
            'INFO': '#3b82f6', 'OK': COLORS['success']
        }
        adv_cards = []
        if not adv.empty:
            for _, row in adv.iterrows():
                c = sev_colors.get(str(row.get('severity', 'INFO')), '#3b82f6')
                adv_cards.append(html.Div([
                    html.Div([
                        html.Span(f"[{row.get('rule_id', '')}]", style={'backgroundColor': c + '22', 'color': c, 'padding': '2px 8px', 'borderRadius': '4px', 'fontSize': '11px', 'fontWeight': '700', 'marginRight': '10px'}),
                        html.Span(row.get('message', ''), style={'color': c, 'fontWeight': '600', 'fontSize': '13px'})
                    ], style={'marginBottom': '8px', 'display': 'flex', 'alignItems': 'center'}),
                    html.Div(row.get('action', ''), style={'color': COLORS['text_muted'], 'fontSize': '12px', 'lineHeight': '1.6'})
                ], style={**CARD_STYLE, 'marginBottom': '12px', 'borderLeft': f'3px solid {c}'}))
        
        source_card = html.Div([
            # PAU logo — centered, full opacity
            html.Img(src='/assets/pau_logo.png', style={
                'height':'100px','display':'block','margin':'0 auto 14px',
            }),
            # Title
            html.Div('Data Source', style={
                'color':'#ffffff','fontWeight':'700','fontSize':'15px',
                'textAlign':'center','marginBottom':'14px','letterSpacing':'0.5px',
            }),
            # All three sources in one flex row, separated by dividers
            html.Div([
                html.Span('PAU Package of Practices Kharif 2025 (Vol. 42)', style={
                    'color':'#e2e8f0','fontSize':'12px','fontFamily':'Inter',
                }),
                html.Span(' · ', style={
                    'color':'rgba(255,255,255,0.35)','margin':'0 10px','fontSize':'16px','fontWeight':'700',
                }),
                html.Span('PAU Package of Practices Rabi 2025', style={
                    'color':'#e2e8f0','fontSize':'12px','fontFamily':'Inter',
                }),
                html.Span(' · ', style={
                    'color':'rgba(255,255,255,0.35)','margin':'0 10px','fontSize':'16px','fontWeight':'700',
                }),
                html.Span('Punjab Soil Health Card — NPK + Micronutrients', style={
                    'color':'#e2e8f0','fontSize':'12px','fontFamily':'Inter',
                }),
            ], style={
                'display':'flex','alignItems':'center','justifyContent':'center',
                'flexWrap':'wrap','gap':'4px','textAlign':'center',
            }),
        ], style={
            **CARD_STYLE,
            'marginBottom':'20px','textAlign':'center',
            'padding':'22px 32px 20px',
        })

        return html.Div([
            html.Div('Risk & Advisory', className='section-title'),
            html.Div(f"PAU Package of Practices 2025  ·  {len(adv)} active advisories  ·  RF risk model",
                     className='section-subtitle'),
            source_card,
            risk_section,
            html.Div(adv_cards)
        ], style=_tab_bg('recommendations.png'))
    elif tab == 'analytics':

        shap_chart = html.Div()
        if not shap.empty and 'pH' in shap.index:
            shap_pH = shap.loc['pH'].sort_values(ascending=True)
            fig_shap = go.Figure(go.Bar(
                x=shap_pH.values, y=shap_pH.index, orientation='h',
                marker=dict(color=shap_pH.values,colorscale='Teal',showscale=False)
            ))
            fig_shap.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=300, margin=dict(l=110,r=20,t=40,b=40),
                title=dict(text='SHAP Feature Importance — pH',
                           font=dict(size=14, color='#ffffff',
                                     family='Rajdhani, sans-serif')),
                xaxis=dict(title='Mean |SHAP|', gridcolor='rgba(0,212,170,0.06)',
                           zeroline=False,
                           tickfont=dict(color='rgba(255,255,255,0.80)', size=11),
                           title_font=dict(color='rgba(255,255,255,0.70)', size=11)),
                yaxis=dict(gridcolor='rgba(0,212,170,0.06)', zeroline=False,
                           tickfont=dict(color='rgba(255,255,255,0.85)', size=11)),
            )
            shap_chart = dcc.Graph(figure=fig_shap)

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(x=val['soil_property'],y=val['R2'],name='Test R2',
                                  marker_color=CHART_COLORS[0],opacity=0.85))
        fig_acc.add_trace(go.Bar(x=val['soil_property'],y=val['CV_R2'],name='CV R2 (5-fold)',
                                  marker_color=CHART_COLORS[1],opacity=0.85))
        fig_acc.add_hline(y=0.5,line_dash='dash',line_color=COLORS['warning'],annotation_text='Moderate (0.5)')
        fig_acc.update_layout(
            barmode='group', template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(l=55,r=20,t=50,b=65),
            title=dict(text='RF Model Accuracy — All Soil Properties',
                       font=dict(size=14, color='#ffffff',
                                 family='Rajdhani, sans-serif')),
            xaxis=dict(tickangle=30, gridcolor='rgba(0,212,170,0.06)',
                       tickfont=dict(color='rgba(255,255,255,0.80)', size=11)),
            yaxis=dict(range=[-0.5,1.0], gridcolor='rgba(0,212,170,0.06)',
                       zeroline=False, tickfont=dict(color='rgba(255,255,255,0.80)', size=11)),
            legend=dict(font=dict(color='rgba(255,255,255,0.85)', size=12), bgcolor='rgba(0,0,0,0)'),
        )

        soil_num = soil[['pH','OC','EC','K2O','available_P','available_N','CEC']]
        corr = soil_num.corr().round(2)
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale='RdBu', zmid=0,
            text=corr.values, texttemplate='%{text}',
            textfont=dict(size=10), showscale=True
        ))
        fig_corr.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(l=85,r=20,t=50,b=60),
            title=dict(text='Soil Property Correlations',
                       font=dict(size=14, color='#ffffff',
                                 family='Rajdhani, sans-serif')),
            xaxis=dict(tickfont=dict(color='rgba(255,255,255,0.80)', size=11)),
            yaxis=dict(tickfont=dict(color='rgba(255,255,255,0.80)', size=11)),
        )

        _narrow = {**CARD_STYLE, 'padding':'14px 18px', 'maxWidth':'88%'}
        return html.Div([
            html.Div('Analytics', className='section-title'),
            html.Div('SHAP explainability  ·  Model validation  ·  Soil correlations',
                     className='section-subtitle'),
            html.Div([
                html.Div([html.Div(shap_chart, style=_narrow)], style={'flex':'1','marginRight':'16px'}),
                html.Div([html.Div([dcc.Graph(figure=fig_acc)], style=_narrow)], style={'flex':'1'}),
            ], style={'display':'flex','marginBottom':'16px'}),
            html.Div([dcc.Graph(figure=fig_corr)], style={**_narrow,'maxWidth':'88%'}),
        ], style=_tab_bg('analytic.png'))

    # ==========================================
    # TAB 7 — FIELD OPERATIONS MONITOR
    # ==========================================
    elif tab == 'landcrops':

        lulc_path = os.path.join(base,'..','data','lulc_summary.json')
        ev_t_path = os.path.join(base,'..','data','field_events_tagged.csv')

        if not os.path.exists(lulc_path):
            return html.Div([
                html.Div('Land Cover data not found.',
                         style={'color':COLORS['danger'],'fontSize':'16px','marginBottom':'8px'}),
                html.Div('Run lulc_event_tagger.py first, then refresh.',
                         style={'color':COLORS['text_muted']})
            ], style={**CARD_STYLE,'marginTop':'40px'})

        with open(lulc_path, encoding='utf-8') as f:
            lulc = json.load(f)
        ev_t   = pd.read_csv(ev_t_path) if os.path.exists(ev_t_path) else pd.DataFrame()
        rabi   = lulc['rabi_2025_26']
        kharif = lulc['kharif_2025']

        # ── Block Area stat only (KPI row removed per spec) ──
        block_area_stat = html.Div([
            html.Div('Block Area', style={
                'fontFamily':'Inter, sans-serif','fontSize':'10px','fontWeight':'500',
                'textTransform':'uppercase','letterSpacing':'1px','color':COLORS['text_muted'],
                'marginBottom':'8px',
            }),
            html.Div([
                html.Span(f"{lulc['block_area_ha']:,.0f}", style={
                    'fontFamily':'Inter, sans-serif','fontSize':'28px','fontWeight':'700',
                    'color':'#ffffff','letterSpacing':'-0.5px',
                }),
                html.Span(' ha', style={
                    'fontFamily':'Inter, sans-serif','fontSize':'14px',
                    'color':COLORS['text_muted'],'marginLeft':'4px',
                }),
            ]),
            html.Div('Total Khamanon Block', style={
                'fontFamily':'Inter, sans-serif','fontSize':'12px',
                'color':COLORS['text_muted'],'marginTop':'4px',
            }),
        ], style={**CARD_STYLE, 'display':'inline-block', 'marginBottom':'20px',
                  'borderLeft':f"3px solid {COLORS['accent']}",
                  'padding':'14px 22px', 'width':'auto'})

        # ── Chart data prep ──────────────────────────────────────────────────────
        friendly = {
            'Wheat':'Wheat','Spring_Maize':'Spring Maize',
            'Rice':'Rice','Kharif_Maize':'Kharif Maize','Agroforestry':'Agroforestry'
        }
        rabi_classes   = list(rabi['classes'].keys())
        kharif_classes = list(kharif['classes'].keys())

        # ── ONE combined stacked chart: 2 rows (Rabi + Kharif), segments = crop classes
        # Matches reference exactly: stacked horizontal bars, tight, 2 rows ──────────
        all_cls   = list(dict.fromkeys(rabi_classes + kharif_classes))  # preserve order
        fig_compare = go.Figure()
        for i, cls in enumerate(all_cls):
            r_pct = float(rabi['classes'].get(cls, {}).get('pct', 0))
            k_pct = float(kharif['classes'].get(cls, {}).get('pct', 0))
            r_ha  = rabi['classes'].get(cls, {}).get('area_ha', 0)
            k_ha  = kharif['classes'].get(cls, {}).get('area_ha', 0)
            col   = CHART_COLORS[i % len(CHART_COLORS)]
            fig_compare.add_trace(go.Bar(
                name=friendly.get(cls, cls),
                y=['Rabi 2025-26', 'Kharif 2025'],
                x=[r_pct, k_pct],
                orientation='h',
                marker=dict(color=col, line=dict(color='rgba(0,0,0,0.3)', width=1)),
                text=[f'{r_pct:.0f}%' if r_pct > 4 else '',
                      f'{k_pct:.0f}%' if k_pct > 4 else ''],
                textposition='inside',
                textfont=dict(color='rgba(255,255,255,0.95)', size=11, family='Inter', weight=600),
                hovertemplate=(
                    f'<b>{friendly.get(cls,cls)}</b><br>'
                    'Rabi: %{customdata[0]:.1f}% · %{customdata[1]:,} ha<br>'
                    'Kharif: %{customdata[2]:.1f}% · %{customdata[3]:,} ha<extra></extra>'
                ),
                customdata=[[r_pct, r_ha, k_pct, k_ha], [r_pct, r_ha, k_pct, k_ha]],
            ))

        fig_compare.update_layout(
            barmode='stack',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=110,
            margin=dict(l=8, r=8, t=4, b=28),
            xaxis=dict(
                range=[0, 102], showgrid=True,
                gridcolor='rgba(255,255,255,0.08)',
                zeroline=False, showticklabels=False,
                tickfont=dict(color='rgba(255,255,255,0.35)', size=9),
            ),
            yaxis=dict(
                showgrid=False, zeroline=False,
                tickfont=dict(color='rgba(255,255,255,0.95)', size=12, family='Inter'),
                ticklabelstandoff=4,
            ),
            legend=dict(
                orientation='h', x=0, y=-0.50, xanchor='left',
                font=dict(size=11, color='rgba(255,255,255,0.88)', family='Inter'),
                bgcolor='rgba(0,0,0,0)', itemsizing='constant',
                traceorder='normal',
            ),
            font=dict(family='Inter, sans-serif', color='#ffffff', size=11),
            bargap=0.28,
            uirevision='compare',
        )

        # ── Compact donut pie (rotation) ──────────────────────────────────────
        rot      = lulc.get('cropping_rotation', [])
        r_labels = [r['rotation'] for r in rot]
        r_areas  = [r['area_est'] for r in rot]
        fig_rot  = go.Figure(go.Pie(
            labels=r_labels, values=r_areas,
            hole=0.48,
            marker=dict(colors=CHART_COLORS[:len(r_labels)],
                        line=dict(color='rgba(0,0,0,0.35)', width=1)),
            texttemplate='%{percent:.0%}',
            textposition='inside',
            textfont=dict(family='Inter', size=11, color='#ffffff'),
            hovertemplate='<b>%{label}</b><br>%{value:,} ha  (%{percent:.1%})<extra></extra>',
            direction='clockwise', sort=True,
        ))
        fig_rot.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=110,
            margin=dict(l=0, r=30, t=4, b=10),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{sum(r_areas):,}</b><br><span>ha</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=11, color='#ffffff', family='Inter'),
                align='center',
            )],
            uirevision='rotation',
        )

        ev_table = html.Div('Land cover event log not loaded.',
                            style={'color':COLORS['text_muted'],'fontSize':'12px',
                                   'padding':'12px'}) if ev_t.empty else html.Div(
            f"{len(ev_t)} tagged operations logged.",
            style={'color':COLORS['text_muted'],'fontSize':'12px','padding':'12px'})

        # Spectral profiles section
        CLASS_COLORS = {'Wheat'       : CHART_COLORS[0],
                        'Spring_Maize': CHART_COLORS[1],
                        'Rice'        : CHART_COLORS[2],
                        'Kharif_Maize': CHART_COLORS[3],
                        'Agroforestry': CHART_COLORS[4]}
        CLASS_LABELS = {'Wheat':'Wheat (Rabi)','Spring_Maize':'Spring Maize (Rabi)',
                        'Rice':'Rice (Kharif)','Kharif_Maize':'Kharif Maize',
                        'Agroforestry':'Agroforestry'}

        index_selector = html.Div([
            html.Label('Spectral Index', style={
                'fontFamily':'Inter, sans-serif','fontSize':'13px','fontWeight':'600',
                'textTransform':'uppercase','letterSpacing':'0.8px',
                'color':'#ffffff','display':'block','marginBottom':'8px',
            }),
            dcc.Dropdown(
                id='crop-index-drop',
                options=[
                    {'label':'NDVI — Canopy Greenness',    'value':'ndvi_mean'},
                    {'label':'BSI — Bare Soil Exposure',   'value':'bsi_mean'},
                    {'label':'NDWI — Field Water Content', 'value':'ndwi_mean'},
                ],
                value='ndvi_mean', clearable=False,
                style={
                    'width'          : '300px',
                    'backgroundColor': COLORS['card'],
                    'color'          : COLORS['text'],
                    'border'         : '1px solid rgba(255,255,255,0.10)',
                    'borderRadius'   : '8px',
                },
            )
        # z-index 500 + overflow visible = dropdown menu never clips under the chart
        ], style={
            **CARD_STYLE,
            'marginBottom'   : '0',
            'display'        : 'inline-block',
            'position'       : 'relative',
            'zIndex'         : '500',
            'overflow'       : 'visible',
        })

        if not lulc_ndvi.empty:
            fig_crop = go.Figure()
            months_order = lulc_ndvi['month'].unique().tolist()
            for x0,x1,label,c in [
                ('Jan-2025','May-2025','Rabi 2024-25','#1D9E75'),
                ('Jun-2025','Oct-2025','Kharif 2025','#378ADD'),
                ('Nov-2025','May-2026','Rabi 2025-26','#1D9E75')
            ]:
                fig_crop.add_vrect(x0=x0,x1=x1,fillcolor=c,opacity=0.06,
                                    line_width=0,annotation_text=label,
                                    annotation_font_color='rgba(255,255,255,0.72)',
                                    annotation_font_size=12)
            for cls in CLASS_COLORS:
                sub = lulc_ndvi[lulc_ndvi['class_name']==cls]
                if sub.empty: continue
                fig_crop.add_trace(go.Scatter(
                    x=sub['month'], y=sub['ndvi_mean'],
                    name=CLASS_LABELS.get(cls,cls), mode='lines+markers',
                    line=dict(color=CLASS_COLORS[cls],width=2.5),
                    marker=dict(size=7)))
            fig_crop.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=450, margin=dict(l=55,r=20,t=44,b=95),
                legend=dict(orientation='h', y=-0.28, x=0,
                            font=dict(size=13, color='rgba(255,255,255,0.88)'),
                            bgcolor='rgba(0,0,0,0)'),
                xaxis=dict(tickangle=45, tickfont=dict(size=12, color='rgba(255,255,255,0.80)'),
                           categoryorder='array', categoryarray=months_order,
                           gridcolor='rgba(0,212,170,0.06)', zeroline=False),
                yaxis=dict(title='Index value',
                           title_font=dict(size=12, color='rgba(255,255,255,0.75)'),
                           gridcolor='rgba(0,212,170,0.06)',
                           zeroline=False,
                           tickfont=dict(size=12, color='rgba(255,255,255,0.80)')),
                title=dict(text='Class-Stratified Spectral Profiles',
                           font=dict(size=15, color='#ffffff',
                                     family='Rajdhani, sans-serif')),
                uirevision='crop',
            )
            chart = dcc.Graph(figure=fig_crop, id='crop-index-chart')
        else:
            chart = html.Div(
                'Class-stratified data not found. Run GEE export first.',
                style={**CARD_STYLE,'color':COLORS['text_muted'],'fontSize':'12px'})

        return html.Div([
            # Header
            html.Div('Land Use & Crop Profiles', className='section-title'),
            html.Div('2 seasonal LULC maps  ·  5 crop classes  ·  Sentinel-2 + SAR  ·  19,616 ha',
                     className='section-subtitle',
                     style={'color':'#ffffff','fontWeight':'500'}),

            # Block Area stat
            block_area_stat,

            # ── Single farmland-bg card: accuracy badges + stacked chart + pie ──
            html.Div([
                # Dark overlay
                html.Div(style={
                    'position':'absolute','top':0,'left':0,'right':0,'bottom':0,
                    'background':'linear-gradient(135deg,rgba(4,6,16,0.86) 0%,rgba(10,12,28,0.78) 100%)',
                    'borderRadius':'12px','zIndex':0,'pointerEvents':'none',
                }),
                # Content
                html.Div([
                    # ── Row 1: accuracy badges (tiny, 1 line each) ────────────
                    html.Div([
                        # Rabi badge
                        html.Div([
                            html.Span('Rabi 2025-26 · RF · S2  ', style={
                                'fontFamily':'Inter','fontSize':'10px',
                                'color':'rgba(255,255,255,0.50)','fontWeight':'500',
                            }),
                            html.Span(f"{rabi['model_accuracy']}%", style={
                                'fontFamily':'Inter','fontSize':'12px','fontWeight':'700',
                                'color':COLORS['accent'],
                            }),
                            html.Span(' OOB', style={
                                'fontFamily':'Inter','fontSize':'9px',
                                'color':'rgba(255,255,255,0.30)',
                            }),
                        ], style={'marginRight':'24px'}),
                        # Kharif badge
                        html.Div([
                            html.Span('Kharif 2025 · SAR+Optical  ', style={
                                'fontFamily':'Inter','fontSize':'10px',
                                'color':'rgba(255,255,255,0.50)','fontWeight':'500',
                            }),
                            html.Span(f"{kharif['model_accuracy']}%", style={
                                'fontFamily':'Inter','fontSize':'12px','fontWeight':'700',
                                'color':COLORS['warning'],
                            }),
                            html.Span(' OOB', style={
                                'fontFamily':'Inter','fontSize':'9px',
                                'color':'rgba(255,255,255,0.30)',
                            }),
                        ]),
                        # Rotation label (right)
                        html.Div('Crop Rotation', style={
                            'marginLeft':'auto','fontFamily':'Inter','fontSize':'10px',
                            'color':'rgba(255,255,255,0.40)','fontWeight':'500',
                        }),
                    ], style={
                        'display':'flex','alignItems':'center',
                        'marginBottom':'8px','flexWrap':'nowrap',
                    }),

                    # ── Row 2: stacked bar chart (left 68%) + pie (right 32%) ─
                    html.Div([
                        # Stacked bar — both seasons, all crops
                        html.Div([
                            dcc.Graph(figure=fig_compare, config={'displayModeBar':False},
                                      style={'height':'155px'}),
                        ], style={
                            'flex':'55','minWidth':'0',
                            'borderRight':'1px solid rgba(255,255,255,0.08)',
                            'paddingRight':'10px', 'marginRight':'10px',
                        }),
                        # Donut pie
                        html.Div([
                            dcc.Graph(figure=fig_rot, config={'displayModeBar':False},
                                      style={'height':'130px'}),
                            # Inline legend
                            html.Div([
                                html.Div([
                                    html.Span(style={
                                        'display':'inline-block','width':'9px','height':'9px',
                                        'borderRadius':'2px',
                                        'backgroundColor':CHART_COLORS[i % len(CHART_COLORS)],
                                        'marginRight':'5px','verticalAlign':'middle',
                                    }),
                                    html.Span(r_labels[i], style={
                                        'fontFamily':'Inter','fontSize':'11px',
                                        'color':'rgba(255,255,255,0.82)',
                                    }),
                                ], style={'marginBottom':'4px','whiteSpace':'nowrap'})
                                for i in range(len(r_labels))
                            ], style={
                                'display':'flex','flexDirection':'column',
                                'paddingLeft':'6px','marginTop':'4px',
                            }),
                        ], style={'flex':'45','minWidth':'0'}),
                    ], style={'display':'flex','alignItems':'flex-start'}),

                ], style={'position':'relative','zIndex':1}),
            ], style={
                'backgroundImage'   : "url('/assets/crop_land.jpeg')",
                'backgroundSize'    : 'cover',
                'backgroundPosition': 'center',
                'borderRadius'      : '12px',
                'padding'           : '14px 16px',
                'position'          : 'relative',
                'overflow'          : 'hidden',
                'marginBottom'      : '24px',
                'border'            : '1px solid rgba(255,255,255,0.08)',
            }),

            # Seasonal Spectral Profiles
            html.Div([
                html.Div('Seasonal Spectral Profiles', style={
                    'fontFamily':'Inter, sans-serif','fontSize':'16px','fontWeight':'700',
                    'color':'#ffffff','marginBottom':'2px',
                }),
                html.Div('Class-stratified NDVI / BSI / NDWI  ·  17 months  ·  Sentinel-2  ·  5 LULC classes',
                         style={'fontFamily':'Inter, sans-serif','fontSize':'13px','fontWeight':'500',
                                'color':'#ffffff','marginBottom':'16px'}),
            ]),
            # Dropdown ABOVE chart, with z-index so menu never clips
            html.Div([
                index_selector,
                html.Div([chart], style={**CARD_STYLE, 'marginTop':'12px', 'maxWidth':'85%'}),
            ], style={'position':'relative', 'zIndex':'10'}),

            html.Div([
                html.Div([
                    html.Span('🗺 Land Cover Map',
                              style={'color':COLORS['text'],'fontSize':'14px',
                                     'fontWeight':'600'}),
                    html.Span('  ·  Toggle seasons in legend',
                              style={'color':COLORS['text_muted'],'fontSize':'12px'})
                ], style={'marginBottom':'14px'}),
                dcc.Graph(id='lulc-map', figure=_lulc_map_cache,
                          config={'displayModeBar':True,
                                  'modeBarButtonsToRemove':['select2d','lasso2d']})
            ], style={**CARD_STYLE,'marginTop':'20px'}),
        ], style=_tab_bg('satellie.png'))

    # ==========================================
    # TAB 8 — LAND COVER MONITOR
    # ==========================================
         

    # =====================================================================
    # NEW TAB SYSTEM BLOCK — SOIL SATELLITE INTERACTION LAYER
    # =====================================================================
    elif tab == 'fieldops':

        ts_path  = os.path.join(base,'..','data','multiindex_timeseries_clean.csv')
        ev_path  = os.path.join(base,'..','data','field_events.csv')
        fst_path = os.path.join(base,'..','data','field_ops_status.json')

        if not os.path.exists(ts_path):
            return html.Div([
                html.Div('Field Operations data not found.',
                         style={'color':COLORS['danger'],'fontSize':'16px','marginBottom':'8px'}),
                html.Div('Run field_ops_detector.py first, then refresh.',
                         style={'color':COLORS['text_muted']})
            ], style={**CARD_STYLE,'marginTop':'40px'})

        ts_df = pd.read_csv(ts_path)
        ts_df['date'] = pd.to_datetime(ts_df['date'])
        ev_df = pd.read_csv(ev_path) if os.path.exists(ev_path) else pd.DataFrame()
        if not ev_df.empty:
            ev_df['date'] = pd.to_datetime(ev_df['date'])
        fops_st = json.load(open(fst_path)) if os.path.exists(fst_path) else {}

        ndvi_now = fops_st.get('latest_NDVI', 0)
        nbr_now  = fops_st.get('latest_NBR', 0)
        if nbr_now < 0.05 and ndvi_now < 0.30:
            current_phase = 'Post-Harvest · Stubble Burning'
            phase_col     = COLORS['danger']
        elif ndvi_now > 0.60:
            current_phase = 'Active Crop Growth'
            phase_col     = COLORS['success']
        elif ndvi_now > 0.30:
            current_phase = 'Crop Established'
            phase_col     = COLORS['accent']
        else:
            current_phase = 'Post-Harvest Fallow'
            phase_col     = COLORS['warning']

        if not ev_df.empty:
            last_ev       = ev_df.iloc[-1]
            last_label    = EV_LABEL.get(last_ev['event'], last_ev['event'])
            last_date_str = last_ev['date'].strftime('%b %d, %Y')
            last_col      = EV_COLOR.get(last_ev['event'], COLORS['accent'])
        else:
            last_label    = 'No events yet'
            last_date_str = '—'
            last_col      = COLORS['text_muted']

        def fops_card(title, val, sub, col):
            """Compact KPI card — clean Inter typography, no cyberpunk."""
            return html.Div([
                html.Div(title, style={
                    'fontFamily':'Inter, sans-serif','fontSize':'9px','fontWeight':'500',
                    'textTransform':'uppercase','letterSpacing':'1px',
                    'color':'rgba(255,255,255,0.40)','marginBottom':'6px',
                }),
                html.Div(val, style={
                    'fontFamily':'Inter, sans-serif','fontSize':'16px','fontWeight':'700',
                    'color':'#ffffff','lineHeight':'1.2','marginBottom':'3px',
                }),
                html.Div(sub, style={
                    'fontFamily':'Inter, sans-serif','fontSize':'10px',
                    'color':'rgba(255,255,255,0.35)',
                }),
            ], style={
                'flex':'1','minWidth':'0','padding':'10px 12px',
                'borderTop':f'2px solid {col}',
                'background':'rgba(0,0,0,0.30)',
                'borderRadius':'8px',
                'backdropFilter':'blur(8px)',
            })

        fig_main = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.70,0.30], vertical_spacing=0.06,
            subplot_titles=('Spectral indices — NDVI · NBR · BSI','Detected field operations timeline')
        )
        for x0,x1,sc in [('2025-01-01','2025-05-15','#1D9E75'),('2025-06-01','2025-11-30','#378ADD'),('2025-12-01','2026-05-26','#1D9E75')]:
            fig_main.add_shape(type='rect',xref='x',yref='y',x0=x0,x1=x1,y0=-0.25,y1=0.92,
                               fillcolor=sc,opacity=0.07,line_width=0,layer='below',row=1,col=1)
        fig_main.add_trace(go.Scatter(x=ts_df['date'],y=ts_df['NDVI'],name='NDVI',mode='lines',
                                       line=dict(color=CHART_COLORS[0],width=2.5),hovertemplate='<b>NDVI</b>: %{y:.3f}<extra></extra>'),row=1,col=1)
        fig_main.add_trace(go.Scatter(x=ts_df['date'],y=ts_df['NBR'],name='NBR',mode='lines',
                                       line=dict(color=CHART_COLORS[1],width=1.8,dash='dash'),hovertemplate='<b>NBR</b>: %{y:.3f}<extra></extra>'),row=1,col=1)
        fig_main.add_trace(go.Scatter(x=ts_df['date'],y=ts_df['BSI'],name='BSI',mode='lines',
                                       line=dict(color=CHART_COLORS[2],width=1.8,dash='dot'),hovertemplate='<b>BSI</b>: %{y:.3f}<extra></extra>'),row=1,col=1)

        if not ev_df.empty:
            for _,erow in ev_df.iterrows():
                ec = EV_COLOR.get(erow['event'],'#94a3b8')
                fig_main.add_shape(type='line',xref='x',yref='y',x0=erow['date'],x1=erow['date'],
                                   y0=-0.25,y1=0.92,line=dict(color=ec,width=1.5,dash='dot'),opacity=0.85,row=1,col=1)
            for etype in EV_COLOR:
                if etype not in ev_df['event'].unique():
                    continue
                sub   = ev_df[ev_df['event']==etype]
                sizes = [18 if c=='HIGH' else 12 for c in sub['confidence']]
                fig_main.add_trace(go.Scatter(
                    x=sub['date'],y=[EV_YPOS.get(etype,0)]*len(sub),mode='markers',
                    name=EV_LABEL.get(etype,etype),showlegend=True,
                    marker=dict(color=EV_COLOR[etype],size=sizes,symbol='diamond',line=dict(color='white',width=1.5)),
                    hovertemplate=f"<b>{EV_LABEL.get(etype,etype)}</b><br>%{{x|%d %b %Y}}<br><extra></extra>"
                ),row=2,col=1)

        fig_main.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=580, margin=dict(l=55,r=20,t=55,b=70),
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', x=0, y=-0.13,
                        font=dict(size=11, color=COLORS['text_muted'])),
            hovermode='x unified',
            font=dict(family='Inter, sans-serif', color=COLORS['text'], size=11),
            uirevision='fieldops',
        )
        fig_main.update_xaxes(gridcolor='rgba(0,212,170,0.05)', tickformat='%b %Y',
                               showgrid=True, tickfont=dict(size=10, color=COLORS['text_muted']),
                               zeroline=False)
        fig_main.update_yaxes(gridcolor='rgba(0,212,170,0.05)', range=[-0.25,0.92],
                               title_text='Index value', title_font=dict(size=11),
                               zeroline=False, row=1, col=1)
        fig_main.update_yaxes(showgrid=False, tickvals=[1,2,3],
                               ticktext=['Rice Transplant','Burning','Harvest'],
                               range=[-0.6,4.0], tickfont=dict(size=10), row=2, col=1)

        table_rows = []
        if not ev_df.empty:
            for _,erow in ev_df.iterrows():
                ec = EV_COLOR.get(erow['event'],COLORS['text_muted'])
                el = EV_LABEL.get(erow['event'],erow['event'])
                ei = EV_ICON.get(erow['event'],'')
                cc = COLORS['success'] if erow['confidence']=='HIGH' else COLORS['warning']
                table_rows.append(html.Tr([
                    html.Td(erow['date'].strftime('%d %b %Y'), style={
                        'fontFamily':'Inter','color':'rgba(255,255,255,0.55)',
                        'padding':'7px 12px','whiteSpace':'nowrap','fontSize':'12px',
                    }),
                    html.Td(html.Span(f"{ei}  {el}", style={
                        'backgroundColor':ec+'22','color':ec,
                        'padding':'2px 10px','borderRadius':'20px',
                        'fontSize':'11px','fontWeight':'500','fontFamily':'Inter',
                    }), style={'padding':'7px 12px'}),
                    html.Td(html.Span(erow['confidence'], style={
                        'color':cc,'fontWeight':'600','fontSize':'11px','fontFamily':'Inter',
                    }), style={'padding':'7px 12px'}),
                    html.Td(str(erow.get('note','')), style={
                        'fontFamily':'Inter','color':'rgba(255,255,255,0.35)',
                        'padding':'7px 12px','fontSize':'11px',
                    }),
                ], style={'borderBottom':'1px solid rgba(255,255,255,0.06)'}))

        # ── Event log table rows (re-usable) ────────────────────────────────────
        table_header_style = {
            'fontFamily':'Inter, sans-serif','fontSize':'10px','fontWeight':'600',
            'textTransform':'uppercase','letterSpacing':'1px',
            'color':'rgba(255,255,255,0.35)','padding':'8px 12px','textAlign':'left',
            'borderBottom':'1px solid rgba(255,255,255,0.08)',
        }
        table_cell_style_date = {
            'fontFamily':'Inter, sans-serif','color':'rgba(255,255,255,0.55)',
            'padding':'8px 12px','whiteSpace':'nowrap','fontSize':'12px',
        }
        table_cell_style_note = {
            'fontFamily':'Inter, sans-serif','color':'rgba(255,255,255,0.40)',
            'padding':'8px 12px','fontSize':'11px',
        }

        event_table_inner = html.Table([
            html.Thead(html.Tr([
                html.Th(h, style=table_header_style)
                for h in ['Date', 'Field Operation', 'Confidence', 'Notes']
            ])),
            html.Tbody(table_rows if table_rows else [
                html.Tr([html.Td(
                    'No events detected yet.',
                    colSpan=4,
                    style={**table_cell_style_note, 'padding':'16px 12px'},
                )])
            ])
        ], style={'width':'100%','borderCollapse':'collapse'})

        # ── Combined rice_wheat bg box: KPI cards + event log ─────────────────
        kpi_event_box = html.Div([
            # Full dark overlay
            html.Div(style={
                'position':'absolute','top':0,'left':0,'right':0,'bottom':0,
                'background':'linear-gradient(135deg,rgba(4,6,14,0.84) 0%,rgba(8,10,22,0.80) 100%)',
                'borderRadius':'12px','zIndex':0,'pointerEvents':'none',
            }),
            # Content
            html.Div([
                # ── 5 KPI cards in one row ────────────────────────────────────
                html.Div([
                    fops_card('Current Phase',
                              current_phase,
                              f"S2: {fops_st.get('latest_date','—')}",
                              phase_col),
                    fops_card('Last Detected Event',
                              last_label,
                              last_date_str,
                              last_col),
                    fops_card('Next Expected',
                              'Rice Transplanting',
                              'June – July 2026',
                              COLORS['success']),
                    fops_card('S2 Images Used',
                              str(fops_st.get('total_images', len(ts_df))),
                              'Jan 2025 – May 2026',
                              COLORS['accent']),
                    fops_card('Events Detected',
                              str(fops_st.get('total_events', len(ev_df))),
                              '2 complete crop cycles',
                              COLORS['accent']),
                ], style={
                    'display':'flex','gap':'10px',
                    'marginBottom':'16px','flexWrap':'nowrap',
                }),

                # ── Divider ───────────────────────────────────────────────────
                html.Div(style={
                    'height':'1px',
                    'background':'rgba(255,255,255,0.08)',
                    'marginBottom':'14px',
                }),

                # ── Event log heading ─────────────────────────────────────────
                html.Div([
                    html.Span('Detected Field Operations', style={
                        'fontFamily':'Inter, sans-serif','fontSize':'12px',
                        'fontWeight':'700','color':'#ffffff',
                    }),
                    html.Span('  ·  Complete log', style={
                        'fontFamily':'Inter, sans-serif','fontSize':'11px',
                        'color':'rgba(255,255,255,0.35)',
                    }),
                ], style={'marginBottom':'10px'}),

                # ── Table ─────────────────────────────────────────────────────
                event_table_inner,

            ], style={'position':'relative','zIndex':1}),
        ], style={
            'backgroundImage'   : "url('/assets/rice_wheat.png')",
            'backgroundSize'    : 'cover',
            'backgroundPosition': 'center center',
            'borderRadius'      : '12px',
            'padding'           : '16px 18px',
            'position'          : 'relative',
            'overflow'          : 'hidden',
            'marginBottom'      : '20px',
            'border'            : '1px solid rgba(255,255,255,0.08)',
        })

        return html.Div([
            html.Div('Field Operations Monitor', className='section-title'),
            html.Div('16 months  ·  Sentinel-2  ·  5 spectral indices  ·  Automated change detection',
                     className='section-subtitle', style={'color':'#ffffff','fontWeight':'500'}),

            # ── Combined box (KPI + log) ── ABOVE the chart ──────────────────
            kpi_event_box,

            # ── Spectral index chart ──────────────────────────────────────────
            html.Div([dcc.Graph(figure=fig_main, config={'displayModeBar':False})],
                     style={**CARD_STYLE, 'marginBottom':'20px', 'padding':'14px 18px'}),
        ], style=_tab_bg('satellie.png'))
    elif tab == 'soilanalysis':
        # ── SECTION A — Soil Fertility Tier Distribution as AREA CHART ─────────
        n = len(grid)

        # Compute % per tier for each property
        def pct(mask): return round(mask.sum() / n * 100, 1)

        prop_labels = ['Soil pH', 'Organic\nCarbon', 'Available N', 'Available P', 'K₂O', 'EC Salinity']
        optimal_pcts  = [
            pct((grid['pH']>=6.5)&(grid['pH']<=7.8))        if 'pH'          in grid.columns else 0,
            pct((grid['OC']>=0.40)&(grid['OC']<=0.75))      if 'OC'          in grid.columns else 0,
            pct((grid['available_N']>=280)&(grid['available_N']<=560)) if 'available_N' in grid.columns else 0,
            pct((grid['available_P']>=11)&(grid['available_P']<=35))   if 'available_P' in grid.columns else 0,
            pct((grid['K2O']>=55)&(grid['K2O']<=110))        if 'K2O'         in grid.columns else 0,
            pct(grid['EC']<0.25)                              if 'EC'          in grid.columns else 0,
        ]
        concern_pcts  = [
            pct(grid['pH']>7.8)                  if 'pH'          in grid.columns else 0,
            pct(grid['OC']<0.40)                 if 'OC'          in grid.columns else 0,
            pct(grid['available_N']<280)         if 'available_N' in grid.columns else 0,
            pct(grid['available_P']<11)          if 'available_P' in grid.columns else 0,
            pct(grid['K2O']<55)                  if 'K2O'         in grid.columns else 0,
            pct(grid['EC']>0.75)                 if 'EC'          in grid.columns else 0,
        ]
        excess_pcts   = [
            pct(grid['pH']<6.5)                              if 'pH'          in grid.columns else 0,
            pct(grid['OC']>0.75)                             if 'OC'          in grid.columns else 0,
            pct(grid['available_N']>560)                     if 'available_N' in grid.columns else 0,
            pct(grid['available_P']>35)                      if 'available_P' in grid.columns else 0,
            pct(grid['K2O']>110)                             if 'K2O'         in grid.columns else 0,
            pct((grid['EC']>=0.25)&(grid['EC']<=0.75))       if 'EC'          in grid.columns else 0,
        ]

        # ── Area chart — layered fills + data labels on every point ────────────
        fig_tiers = go.Figure()

        for vals, name, col, fill_col in [
            (optimal_pcts,  'Optimal / Safe',       '#00d4aa', 'rgba(0,212,170,0.22)'),
            (concern_pcts,  'Deficient / Concern',  '#ef4444', 'rgba(239,68,68,0.18)'),
            (excess_pcts,   'Excess / High',        '#f59e0b', 'rgba(245,158,11,0.16)'),
        ]:
            fig_tiers.add_trace(go.Scatter(
                x=prop_labels,
                y=vals,
                name=name,
                mode='lines+markers+text',
                line=dict(color=col, width=2.5),
                fill='tozeroy',
                fillcolor=fill_col,
                marker=dict(size=8, color=col,
                            line=dict(color='rgba(255,255,255,0.3)', width=1)),
                text=[f'{v:.0f}%' for v in vals],
                textposition='top center',
                textfont=dict(size=11, color=col, family='Inter', weight=600),
                hovertemplate=f'<b>%{{x}}</b><br>{name}: %{{y:.1f}}%<extra></extra>',
            ))

        fig_tiers.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
            margin=dict(l=58, r=16, t=30, b=75),
            xaxis=dict(
                showgrid=True, gridcolor='rgba(255,255,255,0.08)',
                zeroline=False,
                tickfont=dict(color='rgba(255,255,255,0.85)', size=12, family='Inter'),
            ),
            yaxis=dict(
                title=dict(
                    text='% of grid points',
                    font=dict(size=11, color='rgba(255,255,255,0.65)', family='Inter'),
                ),
                showgrid=True, gridcolor='rgba(255,255,255,0.08)',
                zeroline=False, range=[0, 130],
                tickfont=dict(color='rgba(255,255,255,0.65)', size=11, family='Inter'),
                ticksuffix='%',
                dtick=25,
            ),
            legend=dict(
                orientation='h', x=0, y=-0.32, xanchor='left',
                font=dict(size=12, color='rgba(255,255,255,0.85)', family='Inter'),
                bgcolor='rgba(0,0,0,0)',
            ),
            font=dict(family='Inter', color='#ffffff', size=12),
            hovermode='x unified',
            uirevision='tiers',
        )

        # Build interpretation bullets from the computed data
        _tier_prop_map = list(zip(
            ['Soil pH', 'Organic Carbon', 'Available N', 'Available P', 'K₂O', 'EC Salinity'],
            optimal_pcts, concern_pcts, excess_pcts
        ))
        interp_bullets = []
        for lbl, opt, conc, exc in _tier_prop_map:
            if conc > 55:
                icon, clr = '⚠', '#ef4444'
                txt = f'{lbl}: {conc:.0f}% deficient — urgent amendment needed'
            elif conc > 35:
                icon, clr = '⚡', '#f59e0b'
                txt = f'{lbl}: {conc:.0f}% deficient — monitor closely'
            elif exc > 35:
                icon, clr = '↑', '#f59e0b'
                txt = f'{lbl}: {exc:.0f}% in excess — risk of lock-up'
            else:
                icon, clr = '✓', '#00d4aa'
                txt = f'{lbl}: {opt:.0f}% within optimal range'
            interp_bullets.append(html.Div([
                html.Span(icon, style={'color': clr, 'fontWeight': '700', 'marginRight': '8px',
                                       'fontSize': '13px', 'fontFamily': 'Inter'}),
                html.Span(txt, style={'color': '#e2e8f0', 'fontSize': '12px', 'fontFamily': 'Inter',
                                      'lineHeight': '1.5'}),
            ], style={'marginBottom': '10px', 'display': 'flex', 'alignItems': 'flex-start'}))

        # Wider flex layout: chart on left, interpretation bullets on right
        section_a_tiers = html.Div([
            html.Div(style={
                'position':'absolute','top':0,'left':0,'right':0,'bottom':0,
                'background':'linear-gradient(to bottom, rgba(4,6,16,0.88) 0%, rgba(4,6,16,0.92) 100%)',
                'borderRadius':'12px','zIndex':0,'pointerEvents':'none',
            }),
            html.Div([
                html.Div([
                    # Chart (left)
                    html.Div([
                        dcc.Graph(figure=fig_tiers, config={'displayModeBar':False}),
                    ], style={'flex':'3','minWidth':'0'}),
                    # Interpretation bullets (right)
                    html.Div([
                        html.Div('Fertility Snapshot', style={
                            'fontFamily':'Inter, sans-serif','fontSize':'13px','fontWeight':'700',
                            'color':'#ffffff','marginBottom':'14px','letterSpacing':'0.3px',
                        }),
                        html.Div(interp_bullets),
                        html.Div(f'Based on {len(grid):,} RF-IDW prediction points · PAU thresholds', style={
                            'fontFamily':'Inter','fontSize':'10px',
                            'color':'rgba(255,255,255,0.35)','marginTop':'12px',
                        }),
                    ], style={
                        'flex':'2','minWidth':'0','paddingLeft':'20px',
                        'borderLeft':'1px solid rgba(255,255,255,0.08)',
                        'display':'flex','flexDirection':'column','justifyContent':'center',
                    }),
                ], style={'display':'flex','alignItems':'flex-start'}),
            ], style={'position':'relative','zIndex':1}),
        ], style={
            'backgroundImage'   : "url('/assets/area_chart.jpeg')",
            'backgroundSize'    : 'cover',
            'backgroundPosition': 'center',
            'borderRadius'      : '12px',
            'padding'           : '16px 18px',
            'position'          : 'relative',
            'overflow'          : 'hidden',
            'marginBottom'      : '24px',
            'border'            : '1px solid rgba(255,255,255,0.08)',
        })

        # SECTION B — Soil Property Maps (Retained from old soilmaps block)
        section_b_maps_header = html.Div([
            html.Div('Digital Soil Mapping Spatial Infrastructure', 
                     style={'fontSize':'16px','fontWeight':'700','color':COLORS['text'],'marginTop':'12px'}),
            html.Div('Random Forest Inverse Distance Weighted (RF-IDW) Predictions · 30m Resolution', 
                     style={'color':COLORS['text_muted'],'fontSize':'12px','marginBottom':'12px'})
        ])
        
        # Keep your exact old dropdown card layout containing id='soil-prop'
        soil_prop_dropdown_card = html.Div([
            html.Label('Select Target Soil Property Parameter:', style={
                'color':'#e2e8f0','fontSize':'11px','fontWeight':'600',
                'textTransform':'uppercase','letterSpacing':'0.8px','display':'block','marginBottom':'8px',
            }),
            dcc.Dropdown(
                id='soil-prop',
                options=[{'label':p.replace('_',' ').title()+' ('+v['unit']+')','value':p}
         for p,v in {
             'pH'          :{'unit':'pH'},
             'OC'          :{'unit':'%'},
             'EC'          :{'unit':'dS/m'},
             'available_N' :{'unit':'kg/ha'},
             'available_P' :{'unit':'kg/ha'},
             'K2O'         :{'unit':'kg/ha'},
             'CEC'         :{'unit':'meq/100g'},
             'bulk_density':{'unit':'g/L'},
             'CaCO3'       :{'unit':'%'}
         }.items()],
                value='pH',
                clearable=False,
                style={'backgroundColor':COLORS['secondary'],'color':COLORS['text'],'width':'320px'}
            )
        ], style={
            **CARD_STYLE,
            'marginBottom':'14px','display':'inline-block',
            'position':'relative','zIndex':600,'overflow':'visible',
        })

        # Keep your exact old flex block containing id='soil-map-v2' and id='soil-stats-v2'
        soil_maps_flex = html.Div([
            html.Div([dcc.Graph(id='soil-map-v2')], style={'flex':'2'}),
            html.Div(id='soil-stats-v2', style={'flex':'1', 'marginLeft':'20px'})
        ], style={'display':'flex'})

        # SECTION C — Field Sample Points (Retained from old points block)
        section_c_header = html.Div('208 cLHS Field Sample Points',
                                    style={'fontSize':'16px','fontWeight':'700',
                                           'color':COLORS['text'],'marginTop':'32px',
                                           'paddingTop':'20px','marginBottom':'4px',
                                           'borderTop':f"1px solid {COLORS['border']}"})

        # Keep your exact code that references your 208 points scattermap figure (fig_points)
        fig_points = go.Figure()
        fig_points.add_trace(go.Scattermap(
            lat=soil['latitude'], lon=soil['longitude'], mode='markers',
            marker=dict(
                size=10, color=soil['pH'], colorscale='RdYlGn_r',
                colorbar=dict(
                    title=dict(text='pH', font=dict(family='Inter, sans-serif', color=COLORS['text'], size=11)),
                    tickfont=dict(family='Inter, sans-serif', color=COLORS['text'], size=11), thickness=12),
                showscale=True, opacity=0.85
            ),
            text=soil.apply(
                lambda r: f"<b>{r['sample_id']}</b><br>pH: {r['pH']:.2f}<br>"
                          f"OC: {r['OC']:.3f}%<br>EC: {r['EC']:.3f} dS/m<br>"
                          f"N: {r['available_N']:.1f} kg/ha<br>"
                          f"K: {r['K2O']:.1f} kg/ha", axis=1
            ),
            hoverinfo='text'
        ))
        fig_points.update_layout(
            map=dict(style='dark', center=dict(lat=30.795,lon=76.352), zoom=11),
            paper_bgcolor=COLORS['card'], height=520,
            margin=dict(l=0,r=0,t=0,b=0)
        )

        return html.Div([
            html.Div('Soil Analysis', className='section-title'),
            html.Div('RF prediction  ·  PAU thresholds  ·  208 cLHS points  ·  19,290 grid points',
                     className='section-subtitle'),

            # ── 1. Digital Soil Mapping — FIRST ──────────────────────────────
            section_b_maps_header,
            html.Div([
                soil_prop_dropdown_card,
                html.Div(style={'clear':'both'}),
            ], style={'position':'relative','zIndex':600,'overflow':'visible'}),
            soil_maps_flex,

            # ── 2. Soil Fertility Tier Distribution (area chart) — BELOW ─────
            html.Div('Soil Fertility Tier Distribution', style={
                'fontFamily':'Inter, sans-serif','fontSize':'16px','fontWeight':'700',
                'color':'#ffffff','marginTop':'28px','marginBottom':'4px',
                'paddingTop':'20px','borderTop':'1px solid rgba(255,255,255,0.07)',
            }),
            html.Div('PAU thresholds · % of 19,290 grid points per fertility tier',
                     style={'fontFamily':'Inter, sans-serif','fontSize':'12px',
                            'color':COLORS['text_muted'],'marginBottom':'14px'}),
            section_a_tiers,

            # ── 3. Field Sample Points ────────────────────────────────────────
            section_c_header,
            html.Div('208 cLHS sample points · Coloured by pH · Hover for soil profile',
                     style={'fontFamily':'Inter, sans-serif','color':COLORS['text_muted'],
                            'fontSize':'12px','marginBottom':'12px'}),
            html.Div([dcc.Graph(figure=fig_points)], style=CARD_STYLE),
        ], style=_tab_bg('satellie.png'))
    elif tab == 'correlation':
      try:

        # Section 1: Clean Executive Title Banner
        header = html.Div([
            html.Div("Soil–Satellite Correlation Analysis", className='section-title'),
            html.Div("208 cLHS validation checkpoints  ·  Wheat-stratified grids  ·  Spearman rank engine  ·  Sentinel-2 Rabi 2025-26",
                     className='section-subtitle'),
        ])

        # Section 2: Compulsory Evaluation Integrity Disclaimer
        disclaimer = html.Div([
            html.Span("⚠️ Technical Prototype Footnote: ", style={'fontWeight': '700'}),
            "Current statistical representations evaluate synthetic calibration models. "
            "All computed tracking matrices, interactive charts, and point distributions serve as structural placeholders. "
            "Calculated trends resolve to true physical agronomic profiles automatically upon loading official PAU laboratory records "
            "and executing matching coordinate re-extractions."
        ], style={**CARD_STYLE, 'borderLeft': f"4px solid {COLORS['warning']}", 'marginBottom': '20px',
                  'color': COLORS['warning'], 'fontSize': '12px', 'lineHeight': '1.7'})

        # Section 3: Primary Matrix Cross-Correlation Matrix Render Engine
        if not corr_r.empty:
            SOIL_LABELS = {
                'pH':'pH', 'OC':'Organic Carbon', 'EC':'EC Salinity', 'K2O':'K2O',
                'available_P':'Available P', 'available_N':'Available N', 'CEC':'CEC',
                'bulk_density':'Bulk Density', 'CaCO3':'CaCO3'
            }
            y_axis_text = [SOIL_LABELS.get(idx, idx) for idx in corr_r.index]

            fig_hm = go.Figure(go.Heatmap(
                z=corr_r.values,
                x=['NDVI', 'NDBI', 'SAVI', 'BSI'],
                y=y_axis_text,
                colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
                text=[[f"{val:.2f}" for val in row] for row in corr_r.values],
                texttemplate="%{text}",
                textfont=dict(size=13, color='#1a202c'),
                showscale=True,
                colorbar=dict(
                    title=dict(text='Spearman R', font=dict(color=COLORS['text'], size=12)),
                    tickfont=dict(color=COLORS['text'], size=11), thickness=14
                )
            ))
            fig_hm.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=380, margin=dict(l=190, r=40, t=60, b=40),
                title=dict(text='Spearman R Coefficient Matrix (Wheat Footprints Isolated)',
                           font=dict(size=14, color='#ffffff',
                                     family='Rajdhani, sans-serif')),
                xaxis=dict(tickfont=dict(color='rgba(255,255,255,0.88)', size=13), side='bottom'),
                yaxis=dict(tickfont=dict(color='rgba(255,255,255,0.88)', size=12), autorange='reversed'),
            )
            heatmap_block = html.Div([dcc.Graph(figure=fig_hm, config={'displayModeBar': False})],
                                     style={**CARD_STYLE, 'marginBottom': '20px',
                                            'padding':'14px 18px', 'maxWidth':'78%',
                                            'marginLeft':'auto', 'marginRight':'auto'})
        else:
            heatmap_block = html.Div("Execute scripts/soil_spectral_correlation.py to assemble correlation matrices.",
                                     style={**CARD_STYLE, 'color': COLORS['text_muted'], 'fontSize': '13px', 'marginBottom': '20px'})

        # Section 4: Dynamic Interactive Validation Scatters
        SOIL_COLS = ['pH', 'OC', 'EC', 'K2O', 'available_P', 'available_N', 'CEC', 'bulk_density', 'CaCO3']
        SPECTRAL_COLS = ['NDVI', 'NDBI', 'SAVI', 'BSI']

        selectors_card = html.Div([
            html.Div([
                html.Div([
                    html.Label("Target Soil Attribute",
                               style={'color': COLORS['text_muted'], 'fontSize': '11px', 'fontWeight': '600',
                                      'textTransform': 'uppercase', 'letterSpacing': '0.8px', 'display': 'block', 'marginBottom': '8px'}),
                    dcc.Dropdown(
                        id='corr-soil-drop',
                        options=[{'label': col.replace('_', ' ').title(), 'value': col} for col in SOIL_COLS],
                        value='OC', clearable=False,
                        style={'width': '260px', 'backgroundColor': COLORS['card'], 'color': COLORS['text'],
                               'border': f"1px solid {COLORS['border']}", 'borderRadius': '8px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': '24px'}),

                html.Div([
                    html.Label("Satellite Spectral Index",
                               style={'color': COLORS['text_muted'], 'fontSize': '11px', 'fontWeight': '600',
                                      'textTransform': 'uppercase', 'letterSpacing': '0.8px', 'display': 'block', 'marginBottom': '8px'}),
                    dcc.Dropdown(
                        id='corr-spec-drop',
                        options=[{'label': col, 'value': col} for col in SPECTRAL_COLS],
                        value='NDVI', clearable=False,
                        style={'width': '200px', 'backgroundColor': COLORS['card'], 'color': COLORS['text'],
                               'border': f"1px solid {COLORS['border']}", 'borderRadius': '8px'}
                    )
                ], style={'display': 'inline-block'})
            ], style={'display': 'flex', 'alignItems': 'flex-end'})
        ], style={
            **CARD_STYLE,
            'marginBottom': '16px', 'display': 'inline-block',
            'position': 'relative', 'zIndex': 600, 'overflow': 'visible',
        })

        chart_container = html.Div([
            dcc.Graph(id='corr-scatter-chart', style={'height': '360px'})
        ], style={**CARD_STYLE, 'marginBottom': '20px', 'position': 'relative', 'zIndex': 1,
                  'padding':'14px 18px'})

        # Section 5: Geographical Spatial Interpretation Maps
        # Left Panel Sub-Module: Multi-Parameter Extraction Bubble Map
        if not residuals.empty and all(k in residuals.columns for k in ['latitude', 'longitude', 'OC', 'NDVI']):
            n_val = residuals['NDVI']
            # Mandatory Check 3: Map scale ranges to clear bounds, safeguarding resolution clipping
            bubble_sizes = 8 + 14 * (n_val - n_val.min()) / (n_val.max() - n_val.min() + 1e-9)

            fig_bub = go.Figure(go.Scattermap(
                lat=residuals['latitude'], lon=residuals['longitude'], mode='markers',
                marker=dict(
                    size=bubble_sizes, color=residuals['OC'], colorscale='YlGn', showscale=True, opacity=0.88,
                    colorbar=dict(title=dict(text='OC %', font=dict(color=COLORS['text'], size=10)),
                                  tickfont=dict(color=COLORS['text'], size=9), thickness=12)
                ),
                hovertemplate="<b>OC Footprint: %{marker.color:.3f}%</b><br>NDVI: Scaled to Size<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
            ))
            fig_bub.update_layout(
                map=dict(style='dark', center=dict(lat=30.795, lon=76.352), zoom=10.5),
                paper_bgcolor=COLORS['card'], margin=dict(l=0, r=0, t=0, b=0), height=500
            )
            left_map_plot = dcc.Graph(figure=fig_bub, config={'displayModeBar': True})
        else:
            left_map_plot = html.Div("Awaiting execution pipeline compilation to populate spatial bubble vectors.",
                                     style={'color': COLORS['text_muted'], 'fontSize': '13px', 'padding': '40px 20px', 'textAlign': 'center'})

        left_map_layout = html.Div([
            html.Div("Bubble Map — Attribute Cross-Examination Pattern",
                     style={'color': COLORS['text'], 'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '4px'}),
            html.Div("Bubble Size maps canopy vigor (NDVI)  |  Color space tracks Organic Carbon percentage",
                     style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '12px'}),
            left_map_plot
        ], style={**CARD_STYLE, 'flex': '1', 'marginRight': '16px'})

        # Right Panel Sub-Module: Clean Variance Residual Map
        if not residuals.empty and 'residual' in residuals.columns:
            # Mandatory Check 2: Dynamic self-scaling mapping preserving variance geometry symmetric balance
            absolute_ceiling = residuals['residual'].abs().max()
            if absolute_ceiling == 0:
                absolute_ceiling = 0.1

            fig_res = go.Figure(go.Scattermap(
                lat=residuals['latitude'], lon=residuals['longitude'], mode='markers',
                marker=dict(
                    size=10, color=residuals['residual'], colorscale='RdBu',
                    cmin=-absolute_ceiling, cmax=absolute_ceiling, showscale=True, opacity=0.90,
                    colorbar=dict(title=dict(text='Deviation', font=dict(color=COLORS['text'], size=10)),
                                  tickfont=dict(color=COLORS['text'], size=9), thickness=12)
                ),
                hovertemplate="<b>Model Error: %{marker.color:.3f}</b><br>Observed: %{customdata[0]:.3f}<br>Estimated: %{customdata[1]:.3f}<extra></extra>",
                customdata=(
                    residuals[['actual_NDVI', 'fitted_NDVI']].values
                    if 'actual_NDVI' in residuals.columns and 'fitted_NDVI' in residuals.columns
                    else residuals[['NDVI', 'fitted_NDVI']].values
                    if 'NDVI' in residuals.columns and 'fitted_NDVI' in residuals.columns
                    else np.zeros((len(residuals), 2))
                )
            ))
            fig_res.update_layout(
                map=dict(style='dark', center=dict(lat=30.795, lon=76.352), zoom=10.5),
                paper_bgcolor=COLORS['card'], margin=dict(l=0, r=0, t=0, b=0), height=500
            )
            right_map_plot = dcc.Graph(figure=fig_res, config={'displayModeBar': True})
        else:
            right_map_plot = html.Div("Awaiting execution pipeline compilation to populate deviation vectors.",
                                      style={'color': COLORS['text_muted'], 'fontSize': '13px', 'padding': '40px 20px', 'textAlign': 'center'})

        right_map_layout = html.Div([
            html.Div("Residual Map — Satellite-Derived Model Deviation",
                     style={'color': COLORS['text'], 'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '4px'}),
            html.Div("Blue = Performance advantage (Irrigation anomaly)  |  Red = Field boundary/Stress deficit",
                     style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginBottom': '12px'}),
            right_map_plot
        ], style={**CARD_STYLE, 'flex': '1'})

        maps_row = html.Div([left_map_layout, right_map_layout], style={'display': 'flex', 'marginBottom': '20px'})

        # ── Build fig_ndvi here so it is available in this tab (it is only built
        #    inside the overview block, causing a NameError when correlation tab opens) ──
        fig_ndvi = go.Figure()
        zone_list = list(zones)
        for x0, x1, lbl, sc in [
            ('Jan-2025', 'May-2025', 'Rabi 2024-25',  'rgba(0,212,170,0.05)'),
            ('Jun-2025', 'Oct-2025', 'Kharif 2025',   'rgba(233,30,140,0.05)'),
            ('Nov-2025', 'May-2026', 'Rabi 2025-26',  'rgba(0,212,170,0.05)'),
        ]:
            fig_ndvi.add_vrect(x0=x0, x1=x1, fillcolor=sc, line_width=0,
                               annotation_text=lbl,
                               annotation_font_color=COLORS['text_muted'],
                               annotation_font_size=9)
        for i, zone in enumerate(zone_list):
            zc = CHART_COLORS[i % len(CHART_COLORS)]
            r = int(zc[1:3], 16); g = int(zc[3:5], 16); b = int(zc[5:7], 16)
            fig_ndvi.add_trace(go.Scatter(
                x=ndvi_df['month'], y=ndvi_df[zone], name=zone,
                mode='lines+markers',
                fill='tozeroy',
                fillcolor=f'rgba({r},{g},{b},0.10)',
                line=dict(color=zc, width=2.5),
                marker=dict(size=7, symbol='circle',
                            line=dict(color=zc, width=1.5),
                            color=f'rgba({r},{g},{b},0.25)'),
                hovertemplate=f'<b>{zone}</b><br>%{{x}}<br>NDVI: %{{y:.3f}}<extra></extra>',
            ))
        fig_ndvi.add_hline(y=0.40, line_dash='dash', line_color='rgba(239,68,68,0.5)',
                           annotation_text='Fallow Base', annotation_font_size=9,
                           annotation_font_color='#ef4444')
        fig_ndvi.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=40, r=20, t=30, b=70),
            legend=dict(orientation='h', y=-0.40, x=0,
                        font=dict(size=10, color=COLORS['text_muted']),
                        bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(tickangle=45, tickfont=dict(size=9, color=COLORS['text_muted']),
                       gridcolor='rgba(0,212,170,0.05)', zeroline=False),
            yaxis=dict(gridcolor='rgba(0,212,170,0.05)', zeroline=False,
                       tickfont=dict(size=9, color=COLORS['text_muted'])),
            title=dict(text='NDVI Time Series — Khamanon Block',
                       font=dict(size=13, color=COLORS['text'], family='Rajdhani, sans-serif')),
            hovermode='x unified',
            uirevision='ndvi',
        )

        ndvi_card = html.Div(
            [dcc.Graph(figure=fig_ndvi, config={'displayModeBar': False})],
            style={**CARD_STYLE, 'marginBottom': '20px', 'padding':'14px 18px', 'maxWidth':'88%'}
        )
        return html.Div([
            ndvi_card, header, disclaimer, heatmap_block,
            html.Div([
                html.Div([selectors_card], style={'position':'relative','zIndex':600,'overflow':'visible'}),
                chart_container,
            ], style={'position':'relative'}),
            maps_row,
        ], style=_tab_bg('satellie.png'))

      except Exception as _corr_err:
        import traceback
        _tb = traceback.format_exc()
        return html.Div([
            html.Div('🔬 Soil-Satellite — Runtime Error', style={
                'color': COLORS['danger'], 'fontSize': '16px', 'fontWeight': '700', 'marginBottom': '12px'
            }),
            html.Pre(_tb, style={
                'backgroundColor': '#1a1a2e', 'color': '#ff6b6b',
                'padding': '16px', 'borderRadius': '8px', 'fontSize': '12px',
                'fontFamily': 'JetBrains Mono, monospace', 'whiteSpace': 'pre-wrap',
                'border': '1px solid rgba(239,68,68,0.3)', 'overflowX': 'auto'
            }),
            html.Div('Fix the error above, save the file, and refresh the browser.',
                     style={'color': COLORS['text_muted'], 'fontSize': '12px', 'marginTop': '10px'})
        ], style={**CARD_STYLE, 'marginTop': '20px'})

    return html.Div('Tab not found')


# ============================================
# SOIL MAP CALLBACK
# ============================================

@callback(
    Output('soil-map-v2','figure'),
    Output('soil-stats-v2','children'),
    Input('soil-prop','value'),
    Input('refresh','n_intervals')
)
def update_map(prop, n):
    # Mandatory Unpack Target Verification: Exactly 14 mapping structures defined
    _, grid, _, _, st, _, _, _, _, _, _, _, _, _ = load_all()
    s2_date = st.get('sentinel2_date','—')

    cmaps = {'pH':'RdYlGn_r','OC':'YlGn','EC':'OrRd','available_N':'Blues','available_P':'Purples',
             'K2O':'BuGn','CEC':'PuBu','bulk_density':'YlOrBr','CaCO3':'RdPu'}
    units = {'pH':'pH','OC':'%','EC':'dS/m','available_N':'kg/ha','available_P':'kg/ha',
             'K2O':'kg/ha','CEC':'meq/100g','bulk_density':'g/L','CaCO3':'%'}

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32643","EPSG:4326",always_xy=True)
    grid_lon, grid_lat = transformer.transform(grid['easting'].values, grid['northing'].values)

    vals = grid[prop]
    fig  = go.Figure(data=go.Densitymap(
        lat=grid_lat, lon=grid_lon, z=vals, radius=3,
        colorscale=cmaps.get(prop,'Viridis'), showscale=True,
        colorbar=dict(title=dict(text=units.get(prop,''),font=dict(family='Inter, sans-serif', color=COLORS['text'], size=11)),
                      tickfont=dict(family='Inter, sans-serif', color=COLORS['text'], size=11),thickness=12),
        hovertemplate=prop.replace('_',' ').title()+': %{z:.3f} '+units.get(prop,'')+'<extra></extra>'
    ))
    fig.update_layout(
        map=dict(style='dark', center=dict(lat=30.795,lon=76.352), zoom=11),
        paper_bgcolor='rgba(0,0,0,0)', height=500, margin=dict(l=0,r=0,t=40,b=0),
        title=dict(text=prop.replace('_',' ').title()+' ('+units.get(prop,'')+')  ·  S2: '+s2_date,
                   font=dict(size=12, color=COLORS['text'], family='Rajdhani, sans-serif')),
        uirevision=prop,
    )

    context = {
        'pH'         :'Mean 8.39 — strongly alkaline. Values >8.5 risk micronutrient lock-up.',
        'OC'         :'Mean 0.43% — critically low. Healthy soil needs >0.75%.',
        'EC'         :'Mean 0.10 dS/m — no salinity stress. Safe for all crops.',
        'available_N':'Mean 225 kg/ha — moderate. High spatial variability detected.',
        'available_P':'Mean 30.5 kg/ha — moderate. PAU optimal: 25-35 kg/ha.',
        'K2O'        :'Mean 71 kg/ha — some deficiency zones detected (<55 kg/acre).',
        'CEC'        :'Mean 14 meq/100g — typical for loamy alluvial Punjab soils.',
        'bulk_density':'Mean 234 g/L — some compaction zones above 260 g/L.',
        'CaCO3'      :'Mean 0.76% — locks up P and micronutrients in high zones.'
    }

    stats = [html.Div(prop.replace('_',' ').title(),
                      style={'color':COLORS['text'],'fontWeight':'700','fontSize':'15px','marginBottom':'16px'})]
    for label, value in [
        ('Mean',    f"{vals.mean():.3f} {units.get(prop,'')}"),
        ('Min',     f"{vals.min():.3f}"),
        ('Max',     f"{vals.max():.3f}"),
        ('Std Dev', f"{vals.std():.3f}"),
        ('Points',  f"{len(vals):,}"),
        ('S2 Date', s2_date)
    ]:
        stats.append(html.Div([
            html.Div(label,style={'color':COLORS['text_muted'],'fontSize':'10px','fontWeight':'600',
                                   'textTransform':'uppercase','letterSpacing':'0.6px','marginBottom':'2px'}),
            html.Div(value,style={'color':COLORS['text'],'fontSize':'13px','fontWeight':'500',
                                   'marginBottom':'12px','paddingBottom':'12px',
                                   'borderBottom':f"1px solid {COLORS['border']}"})
        ]))
    stats.append(html.Div(context.get(prop,''),style={
        'color':COLORS['accent'],'fontSize':'11px','lineHeight':'1.6',
        'backgroundColor':COLORS['accent']+'11','padding':'10px','borderRadius':'6px','marginTop':'4px'
    }))
    return fig, stats


# =========================================================================
# INTERACTIVE SOIL-SATELLITE SCATTER DIAGNOSTIC CALLBACK
# =========================================================================
@callback(
    Output('corr-scatter-chart', 'figure'),
    Input('corr-soil-drop', 'value'),
    Input('corr-spec-drop', 'value'),
    Input('refresh', 'n_intervals'),
    prevent_initial_call=False
)
def update_corr_scatter(soil_col, spec_col, n):
    try:
     return _update_corr_scatter_inner(soil_col, spec_col, n)
    except Exception as _e:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[dict(text=f"Chart error: {_e}", showarrow=False,
                              font=dict(color='#ef4444', size=12))]
        )
        return fig

def _update_corr_scatter_inner(soil_col, spec_col, n):
    # Mandatory Unpack Target Verification: Exactly 14 mapping structures defined
    (_, _, _, _, _, _, _, _, _, _, _, _, _, residuals) = load_all()

    # Establish fallback alerts if source arrays are missing
    if residuals.empty:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[dict(text="Please execute scripts/soil_spectral_correlation.py to parse validation points.",
                              showarrow=False, font=dict(color=COLORS['text_muted'], size=13))]
        )
        return fig

    if soil_col not in residuals.columns or spec_col not in residuals.columns:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[dict(text="Selected variable matrix signature not found in target registry structure.",
                              showarrow=False, font=dict(color=COLORS['text_muted'], size=13))]
        )
        return fig

    clean_subset = residuals[[soil_col, spec_col]].dropna()
    if len(clean_subset) < 5:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[dict(text="Insufficient data footprints remaining after removing null rows.",
                              showarrow=False, font=dict(color=COLORS['text_muted'], size=13))]
        )
        return fig

    x_data = clean_subset[soil_col].values
    y_data = clean_subset[spec_col].values

    # Evaluate Spearman Rank significance
    from scipy.stats import spearmanr
    r_coefficient, p_significance = spearmanr(x_data, y_data)

    # Derive Ordinary Least Squares trend line vector
    regression_line_slope, intercept_constant = np.polyfit(x_data, y_data, 1)
    x_regression_steps = np.linspace(x_data.min(), x_data.max(), 100)
    y_regression_steps = np.polyval([regression_line_slope, intercept_constant], x_regression_steps)

    fig = go.Figure()

    # Inject point distribution array
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data, mode='markers', name='Sample points',
        marker=dict(color=CHART_COLORS[0], size=8, opacity=0.75,
                    line=dict(color='rgba(0,212,170,0.15)', width=0.5)),
        hovertemplate=f"{soil_col.replace('_',' ').title()}: %{{x:.3f}}<br>{spec_col}: %{{y:.3f}}<extra></extra>"
    ))

    # Inject trendline overlay vector
    fig.add_trace(go.Scatter(
        x=x_regression_steps, y=y_regression_steps, mode='lines', name='Linear Trend',
        line=dict(color=COLORS['warning'], width=2, dash='dash'),
        showlegend=False
    ))

    # Display statistical matrix scorecard card within presentation block
    fig.add_annotation(
        xref='paper', yref='paper', x=0.02, y=0.96,
        text=f"<b>Spearman R</b> = {r_coefficient:.3f}   |   <b>p-value</b> = {p_significance:.4e}   |   <b>n</b> = {len(x_data)}",
        showarrow=False, font=dict(color=COLORS['accent'], size=11), align='left',
        bgcolor='rgba(30,37,53,0.9)', bordercolor=COLORS['border'], borderwidth=1, borderpad=7
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=360, margin=dict(l=60, r=30, t=50, b=60),
        xaxis=dict(title=soil_col.replace('_', ' ').title(),
                   gridcolor='rgba(0,212,170,0.05)', zeroline=False,
                   tickfont=dict(color=COLORS['text_muted'], size=10)),
        yaxis=dict(title=spec_col,
                   gridcolor='rgba(0,212,170,0.05)', zeroline=False,
                   tickfont=dict(color=COLORS['text_muted'], size=10)),
        title=dict(text=f"Co-located Regression Profile: {soil_col.replace('_',' ').title()} vs {spec_col}",
                   font=dict(size=13, color=COLORS['text'], family='Rajdhani, sans-serif')),
        showlegend=False,
        uirevision=f'{soil_col}-{spec_col}',
    )

    return fig


# ============================================
# CROP INDEX DROPDOWN CALLBACK
# ============================================
# Add this right above your update_crop_index callback block near the bottom:
@callback(
    Output('sat-log-open', 'data'),
    Input('sat-log-toggle-btn', 'n_clicks'),
    State('sat-log-open', 'data'),
    prevent_initial_call=True
)
def toggle_sat_log(n_clicks, current):
    return not (current or False)

@callback(
    Output('sat-log-body', 'style'),
    Input('sat-log-open', 'data'),
)
def show_sat_log(is_open):
    return {'display': 'block' if is_open else 'none', 'marginTop': '14px'}

@callback(
    Output('crop-index-chart','figure'),
    Input('crop-index-drop','value'),
    Input('refresh','n_intervals'),
    prevent_initial_call=True
)
def update_crop_index(index_col, n):
    # Mandatory Unpack Target Verification: Exactly 14 mapping structures defined
    _, _, _, _, _, _, _, _, _, _, _, lulc_ndvi, _, _ = load_all()
    if lulc_ndvi.empty:
        return go.Figure()

    CLASS_COLORS = {'Wheat':CHART_COLORS[0],'Spring_Maize':CHART_COLORS[1],'Rice':CHART_COLORS[2],'Kharif_Maize':CHART_COLORS[3],'Agroforestry':CHART_COLORS[4]}
    CLASS_LABELS = {'Wheat':'Wheat (Rabi)','Spring_Maize':'Spring Maize (Rabi)','Rice':'Rice (Kharif)','Kharif_Maize':'Kharif Maize','Agroforestry':'Agroforestry'}
    INDEX_TITLES = {'ndvi_mean':'NDVI — Canopy Greenness','bsi_mean':'BSI — Bare Soil Exposure','ndwi_mean':'NDWI — Field Water Content'}
    months_order = lulc_ndvi['month'].unique().tolist()

    fig = go.Figure()
    for x0,x1,label,c in [('Jan-2025','May-2025','Rabi 2024-25','#1D9E75'),('Jun-2025','Oct-2025','Kharif 2025','#378ADD'),('Nov-2025','May-2026','Rabi 2025-26','#1D9E75')]:
        fig.add_vrect(x0=x0,x1=x1,fillcolor=c,opacity=0.06,line_width=0,annotation_text=label,annotation_font_color='#94a3b8',annotation_font_size=10)
    for cls, color in CLASS_COLORS.items():
        sub = lulc_ndvi[lulc_ndvi['class_name']==cls]
        if sub.empty or index_col not in sub.columns:
            continue
        fig.add_trace(go.Scatter(x=sub['month'],y=sub[index_col],name=CLASS_LABELS.get(cls,cls),
                                  mode='lines+markers',line=dict(color=color,width=2.5),marker=dict(size=7)))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=450, margin=dict(l=50,r=20,t=40,b=80),
        legend=dict(orientation='h', y=-0.25, x=0,
                    font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(tickangle=45, tickfont=dict(size=10, color=COLORS['text_muted']),
                   categoryorder='array', categoryarray=months_order,
                   gridcolor='rgba(0,212,170,0.05)', zeroline=False),
        yaxis=dict(title=INDEX_TITLES.get(index_col,index_col),
                   gridcolor='rgba(0,212,170,0.05)', zeroline=False,
                   tickfont=dict(color=COLORS['text_muted'])),
        title=dict(text=f"{INDEX_TITLES.get(index_col,index_col)} — Class-Stratified · Khamanon Block",
                   font=dict(size=13, color=COLORS['text'], family='Rajdhani, sans-serif')),
        uirevision=index_col,
    )
    return fig


# ============================================
# RUN
# ============================================

server = app.server  # required for Render/gunicorn

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  KHAMANON DIGITAL TWIN v2.0")
    print("=" * 55)
    print("\nOpen browser: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop.")
    print("=" * 55 + "\n")
    app.run(debug=False, port=8050)