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
from dash import Dash, dcc, html, Input, Output, callback
import json
import os
import math
from datetime import datetime

print("=" * 55)
print("  DASHBOARD v2.0 — PROFESSIONAL REDESIGN")
print("  Khamanon Block Digital Twin")
print("=" * 55)

base = os.path.dirname(os.path.abspath(__file__))

# ============================================
# DESIGN TOKENS
# ============================================

COLORS = {
    'primary'    : '#1a1f2e',
    'secondary'  : '#252b3b',
    'accent'     : '#00d4aa',
    'accent2'    : '#7c5cbf',
    'warning'    : '#f59e0b',
    'danger'     : '#ef4444',
    'success'    : '#10b981',
    'text'       : '#f1f5f9',
    'text_muted' : '#94a3b8',
    'card'       : '#1e2535',
    'border'     : '#2d3748',
    'white'      : '#ffffff',
    'chart_bg'   : '#1a1f2e'
}

CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'borderRadius'   : '12px',
    'padding'        : '20px',
    'border'         : f"1px solid {COLORS['border']}",
    'boxShadow'      : '0 4px 20px rgba(0,0,0,0.3)'
}

TEXT_STYLE = {
    'color'     : COLORS['text'],
    'fontFamily': 'Segoe UI, Inter, Arial, sans-serif'
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
# DATA LOADER
# ============================================

def load_all():
    soil = pd.read_csv(
        os.path.join(base,'..','data',
                     'soil_data_validated.csv')
    )
    grid = pd.read_csv(
        os.path.join(base,'..','data',
                     'real_prediction_grid.csv')
    )
    ndvi = pd.read_csv(
        os.path.join(base,'..','data',
                     'ndvi_processed.csv')
    )
    val = pd.read_csv(
        os.path.join(base,'..','data',
                     'model_validation_real.csv')
    )

    st_path = os.path.join(base,'..','data',
                           'last_update.json')
    st = json.load(open(st_path)) \
         if os.path.exists(st_path) \
         else {'last_run':'Never',
               'sentinel2_date':'Unknown',
               'ndvi_mean':0,'images_used':0,
               'status':'No data','alerts':[]}

    wx_path = os.path.join(base,'..','data',
                           'current_weather.json')
    wx = json.load(open(wx_path)) \
         if os.path.exists(wx_path) \
         else {'temperature':35,'humidity':45,
               'description':'N/A',
               'wind_speed':3,'rain_1h':0,
               'advisories':[]}

    adv_path = os.path.join(base,'..','data',
                            'block_advisories.csv')
    adv = pd.read_csv(adv_path) \
          if os.path.exists(adv_path) \
          else pd.DataFrame()

    risk_path = os.path.join(base,'..','data',
                             'point_risk_scores.csv')
    risk = pd.read_csv(risk_path) \
           if os.path.exists(risk_path) \
           else pd.DataFrame()

    shap_path = os.path.join(base,'..','data',
                             'shap_importance.csv')
    shap = pd.read_csv(shap_path,
                       index_col='soil_property') \
           if os.path.exists(shap_path) \
           else pd.DataFrame()

    hist_path = os.path.join(base,'..','data',
                             'update_history.csv')
    hist = pd.read_csv(hist_path) \
           if os.path.exists(hist_path) \
           else pd.DataFrame()

    sat_log_path = os.path.join(base,'..','data','satellite_log.csv')
    sat_log = pd.read_csv(sat_log_path) if os.path.exists(sat_log_path) else pd.DataFrame()

    return (soil, grid, ndvi, val,
            st, wx, adv, risk, shap, hist, sat_log)

# ============================================
# HELPER: METRIC CARD
# ============================================

def metric_card(title, value, unit,
                subtitle='', color=None):
    c = color or COLORS['accent']
    return html.Div([
        html.Div(title, style={
            'color'        : COLORS['text_muted'],
            'fontSize'     : '11px',
            'fontWeight'   : '600',
            'textTransform': 'uppercase',
            'letterSpacing': '0.8px',
            'marginBottom' : '8px'
        }),
        html.Div([
            html.Span(str(value), style={
                'fontSize'  : '28px',
                'fontWeight': '700',
                'color'     : c
            }),
            html.Span(' ' + unit, style={
                'fontSize'  : '13px',
                'color'     : COLORS['text_muted'],
                'marginLeft': '4px'
            })
        ]),
        html.Div(subtitle, style={
            'color'    : COLORS['text_muted'],
            'fontSize' : '11px',
            'marginTop': '4px'
        })
    ], style={**CARD_STYLE,
              'borderTop': f'3px solid {c}'})


# ============================================
# HELPER: ALERT BADGE
# ============================================

def alert_badge(text, severity):
    colors = {
        'CRITICAL': ('#ef4444', '#2d1b1b'),
        'WARNING' : ('#f59e0b', '#2d2518'),
        'INFO'    : ('#3b82f6', '#1b2340'),
        'OK'      : ('#10b981', '#1b2d25')
    }
    fc, bg = colors.get(severity, ('#94a3b8', '#1e2535'))
    return html.Div(text, style={
        'backgroundColor': bg,
        'color'          : fc,
        'border'         : f'1px solid {fc}',
        'borderRadius'   : '6px',
        'padding'        : '8px 12px',
        'fontSize'       : '12px',
        'marginBottom'   : '6px',
        'fontWeight'     : '500'
    })


def _sat_status_badge(status):
    cfg = {
        'ACCEPTED': (COLORS['success'], '#1b2d25', '✓ ACCEPTED'),
        'REJECTED': (COLORS['danger'],  '#2d1b1b', '✗ REJECTED'),
    }
    fc, bg, label = cfg.get(status, (COLORS['text_muted'], COLORS['card'], status))
    return html.Span(label, style={
        'backgroundColor': bg,
        'color'          : fc,
        'border'         : f'1px solid {fc}',
        'borderRadius'   : '4px',
        'padding'        : '2px 8px',
        'fontSize'       : '11px',
        'fontWeight'     : '600',
    })

# ============================================
# HELPER: SOIL RANGE DISTRIBUTION TIER
# ============================================

def _tier_row(grid, col, label, tiers):
    """Stacked bar row showing % of grid points per PAU agronomic tier."""
    if col not in grid.columns:
        return html.Div()
    n   = len(grid)
    vals = grid[col]
    segments, pills = [], []
    for name, cond_fn, color in tiers:
        pct = round(cond_fn(vals).sum() / n * 100, 1)
        if pct > 0:
            segments.append(html.Div(style={
                'width'          : f'{pct}%',
                'height'         : '9px',
                'backgroundColor': color,
                'display'        : 'inline-block',
            }, title=f'{name}: {pct}%'))
        pills.append(html.Span([
            html.Span(style={
                'display'        : 'inline-block',
                'width'          : '7px',
                'height'         : '7px',
                'borderRadius'   : '2px',
                'backgroundColor': color,
                'marginRight'    : '3px',
                'verticalAlign'  : 'middle',
            }),
            html.Span(f'{name}  {pct}%', style={
                'fontSize'   : '10px',
                'color'      : COLORS['text_muted'],
                'marginRight': '10px',
            })
        ]))
    return html.Div([
        html.Div([
            html.Span(label, style={
                'color'      : COLORS['text_muted'],
                'fontSize'   : '11px',
                'fontWeight' : '600',
                'width'      : '115px',
                'flexShrink' : '0',
                'display'    : 'inline-block',
            }),
            html.Div(segments, style={
                'flex'           : '1',
                'height'         : '9px',
                'borderRadius'   : '5px',
                'overflow'       : 'hidden',
                'backgroundColor': COLORS['border'],
                'display'        : 'flex',
            }),
        ], style={'display': 'flex', 'alignItems': 'center',
                  'marginBottom': '4px'}),
        html.Div(pills, style={
            'paddingLeft': '115px',
            'marginBottom': '12px',
        }),
    ])


# ============================================
# TAB STYLE HELPERS
# ============================================

def tab_style():
    return {
        'backgroundColor': COLORS['primary'],
        'color'          : COLORS['text_muted'],
        'border'         : 'none',
        'padding'        : '12px 20px',
        'fontFamily'     : 'Segoe UI, Arial, sans-serif',
        'fontSize'       : '13px'
    }

def tab_selected_style():
    return {
        'backgroundColor': COLORS['secondary'],
        'color'          : COLORS['accent'],
        'border'         : 'none',
        'borderBottom'   : f"2px solid {COLORS['accent']}",
        'padding'        : '12px 20px',
        'fontFamily'     : 'Segoe UI, Arial, sans-serif',
        'fontSize'       : '13px',
        'fontWeight'     : '600'
    }


# ============================================
# HELPER: BUILD LULC MAP FROM SAMPLED CSVs
# ============================================

def _build_lulc_map(base_path):
    rabi_csv   = os.path.join(base_path, '..', 'data', 'lulc_rabi_map.csv')
    kharif_csv = os.path.join(base_path, '..', 'data', 'lulc_kharif_map.csv')

    fig = go.Figure()
    rabi_loaded   = False
    kharif_loaded = False

    if os.path.exists(rabi_csv):
        rabi_df = pd.read_csv(rabi_csv)
        for cls_name, grp in rabi_df.groupby('class_name'):
            color = grp['color'].iloc[0]
            fig.add_trace(go.Scattermap(
                lat=grp['lat'], lon=grp['lon'],
                mode='markers',
                name=f'{cls_name} (Rabi)',
                legendgroup='rabi',
                legendgrouptitle_text='Rabi 2025-26',
                marker=dict(size=7, color=color, opacity=0.85),
                hovertemplate=(
                    f'<b>{cls_name}</b><br>Rabi 2025-26<br>'
                    'Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
                ),
                visible=True
            ))
        rabi_loaded = True

    if os.path.exists(kharif_csv):
        kharif_df = pd.read_csv(kharif_csv)
        for cls_name, grp in kharif_df.groupby('class_name'):
            color = grp['color'].iloc[0]
            fig.add_trace(go.Scattermap(
                lat=grp['lat'], lon=grp['lon'],
                mode='markers',
                name=f'{cls_name} (Kharif)',
                legendgroup='kharif',
                legendgrouptitle_text='Kharif 2025',
                marker=dict(size=7, color=color, opacity=0.85),
                hovertemplate=(
                    f'<b>{cls_name}</b><br>Kharif 2025<br>'
                    'Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
                ),
                visible='legendonly'
            ))
        kharif_loaded = True

    if not rabi_loaded and not kharif_loaded:
        fig.add_trace(go.Scattermap(
            lat=[30.795], lon=[76.352],
            mode='markers+text',
            text=['Run lulc_map_processor.py to show LULC map'],
            textposition='top center',
            marker=dict(size=12, color='#00d4aa'),
            showlegend=False
        ))

    fig.update_layout(
        map=dict(style='dark', center=dict(lat=30.795, lon=76.352), zoom=10),
        paper_bgcolor=COLORS['card'],
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor='rgba(30,37,53,0.9)',
            bordercolor=COLORS['border'],
            borderwidth=1,
            font=dict(color=COLORS['text'], size=11),
            itemsizing='constant',
            groupclick='toggleitem',
            x=0.01, y=0.99,
            xanchor='left', yanchor='top'
        ),
        uirevision='constant'
    )
    return fig


# ============================================
# APP INITIALIZATION
# ============================================
_lulc_map_cache = _build_lulc_map(base)
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([

    dcc.Interval(id='refresh', interval=30 * 1000, n_intervals=0),

    html.Div([
        html.Div([
            html.Div([
                html.Span('⬡ ', style={'color': COLORS['accent'], 'fontSize': '22px'}),
                html.Span('Khamanon Digital Twin', style={
                    'color': COLORS['text'], 'fontSize': '18px',
                    'fontWeight': '700', 'letterSpacing': '0.5px'
                })
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div(
                'Fatehgarh Sahib, Punjab  ·  Real-Time Sentinel-2  ·  '
                '208 cLHS Points  ·  PAU Advisory Engine',
                style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginTop': '4px'}
            )
        ], style={'flex': '1'}),
        html.Div(id='nav-status', style={'textAlign': 'right'})
    ], style={
        'backgroundColor': COLORS['primary'],
        'padding'        : '16px 30px',
        'display'        : 'flex',
        'alignItems'     : 'center',
        'borderBottom'   : f"1px solid {COLORS['border']}",
        'position'       : 'sticky',
        'top'            : '0',
        'zIndex'         : '1000'
    }),

    html.Div([
        dcc.Tabs(
            id='tabs',
            value='overview',
            children=[
                dcc.Tab(label='Overview',     value='overview',
                        style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label='Soil Maps',    value='soilmaps',
                        style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label='Crop Monitor', value='crop',
                        style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label='PAU Advisory', value='advisory',
                        style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label='Field Points', value='points',
                        style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label='Analytics',    value='analytics',
                        style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label='🛰 Field Ops', value='fieldops',
                        style={**tab_style(), 'borderTop': f"2px solid {COLORS['accent2']}"},
                        selected_style={**tab_selected_style(),
                                        'color': COLORS['accent2'],
                                        'borderBottom': f"2px solid {COLORS['accent2']}"}),
                dcc.Tab(label='🗺 Land Cover', value='landcover',
                        style={**tab_style(), 'borderTop': f"2px solid {COLORS['success']}"},
                        selected_style={**tab_selected_style(),
                                        'color': COLORS['success'],
                                        'borderBottom': f"2px solid {COLORS['success']}"}),
            ],
            style={
                'backgroundColor': COLORS['primary'],
                'borderBottom'   : f"1px solid {COLORS['border']}",
                'display'        : 'flex',
                'flexDirection'  : 'row'
            }
        )
    ]),

    html.Div(
        id='page-content',
        style={
            'backgroundColor': COLORS['primary'],
            'minHeight'      : '90vh',
            'padding'        : '24px 30px',
            **TEXT_STYLE
        }
    )

], style={'backgroundColor': COLORS['primary'], 'minHeight': '100vh', **TEXT_STYLE})


# ============================================
# NAV STATUS CALLBACK
# ============================================

@callback(
    Output('nav-status', 'children'),
    Input('refresh', 'n_intervals')
)
def update_nav(n):
    _, _, _, _, st, wx, _, _, _, _, sat_log = load_all()
    ndvi      = st.get('ndvi_mean', 0)
    status_col = COLORS['success'] if st.get('status') == 'SUCCESS' else COLORS['warning']
    return html.Div([
        html.Div([
            html.Span('● ', style={'color': status_col, 'fontSize': '14px'}),
            html.Span(st.get('status', 'Unknown'),
                      style={'color': COLORS['text'], 'fontSize': '12px', 'fontWeight': '600'})
        ]),
        html.Div(
            f"S2: {st.get('sentinel2_date', '—')}  "
            f"NDVI: {ndvi:.3f}  "
            f"Temp: {wx.get('temperature', '—')}°C",
            style={'color': COLORS['text_muted'], 'fontSize': '11px', 'marginTop': '3px'}
        )
    ])


# ============================================
# MAIN PAGE CALLBACK
# ============================================

@callback(
    Output('page-content', 'children'),
    Input('tabs', 'value'),
    Input('refresh', 'n_intervals')
)
def render_page(tab, n):
    (soil, grid, ndvi_df, val,
     st, wx, adv, risk, shap, hist, sat_log) = load_all()

    ndvi_mean = st.get('ndvi_mean', 0)
    s2_date   = st.get('sentinel2_date', '—')
    zones     = [c for c in ndvi_df.columns if c != 'month']

    # ==========================================
    # TAB 1 — OVERVIEW
    # ==========================================
    if tab == 'overview':

        ndvi_col = (COLORS['danger']
                    if ndvi_mean < 0.35 else COLORS['success'])
        temp_col = (COLORS['danger']
                    if wx.get('temperature', 0) > 35
                    else COLORS['success'])

        _T   = wx.get('temperature', 35)
        _RH  = wx.get('humidity', 45)
        _es  = 0.61078 * math.exp((17.27 * _T) / (_T + 237.3))
        _ea  = _es * (_RH / 100)
        vpd  = round(_es - _ea, 2)
        vpd_label = (
            'Low — good for crops'     if vpd < 0.5
            else 'Moderate — monitor'      if vpd < 1.0
            else 'High — stomatal stress'  if vpd < 1.5
            else 'Critical — growth halted'
        )
        vpd_col = (
            COLORS['success'] if vpd < 0.5
            else COLORS['accent']  if vpd < 1.0
            else COLORS['warning'] if vpd < 1.5
            else COLORS['danger']
        )

        cards = html.Div([
            html.Div([metric_card(
                'Block NDVI', round(ndvi_mean, 3), '',
                f"S2: {s2_date}", ndvi_col
            )], style={'flex': '1', 'marginRight': '12px'}),
            html.Div([metric_card(
                'Sentinel-2 Images',
                st.get('images_used', 0), 'used',
                'Last 30 days', COLORS['accent']
            )], style={'flex': '1', 'marginRight': '12px'}),
            html.Div([metric_card(
                'Temperature',
                wx.get('temperature', '—'), '°C',
                str(wx.get('description', 'Loading...')).title(),
                temp_col
            )], style={'flex': '1', 'marginRight': '12px'}),
            html.Div([metric_card(
                'Humidity',
                wx.get('humidity', '—'), '%',
                'Current', COLORS['accent2']
            )], style={'flex': '1', 'marginRight': '12px'}),
            html.Div([metric_card(
                'Field Samples', len(soil), 'points',
                'cLHS validated', COLORS['accent']
            )], style={'flex': '1', 'marginRight': '12px'}),
            html.Div([metric_card(
                'Atm. VPD',
                vpd, 'kPa',
                vpd_label,
                vpd_col
            )], style={'flex': '1'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px', 'marginBottom': '20px'})

        soil_stats = html.Div([
            html.Div([
                html.Div('Soil Fertility Tier Distribution', style={
                    'color': COLORS['text'], 'fontSize': '14px',
                    'fontWeight': '600', 'marginBottom': '4px'
                }),
                html.Div(
                    f'PAU thresholds  ·  {len(grid):,} prediction points  ·  10m grid',
                    style={'color': COLORS['text_muted'], 'fontSize': '11px',
                           'marginBottom': '16px'}
                ),
                html.Div([
                    _tier_row(grid, 'pH', 'Soil pH',
                        [('Optimal 6.5–7.8', lambda x: (x>=6.5)&(x<=7.8), COLORS['success']),
                         ('Alkaline >7.8',   lambda x: x > 7.8,            COLORS['danger']),
                         ('Acidic <6.5',     lambda x: x < 6.5,            '#3b82f6')]),
                    _tier_row(grid, 'OC', 'Organic Carbon',
                        [('Critical <0.40%',  lambda x: x < 0.40,               COLORS['danger']),
                         ('Optimum 0.40–0.75%', lambda x: (x>=0.40)&(x<=0.75), COLORS['success']),
                         ('High >0.75%',      lambda x: x > 0.75,               '#3b82f6')]),
                    _tier_row(grid, 'available_N', 'Available N',
                        [('Deficient <280',  lambda x: x < 280,              COLORS['danger']),
                         ('Optimum 280–560', lambda x: (x>=280)&(x<=560),   COLORS['success']),
                         ('Excess >560',     lambda x: x > 560,              COLORS['warning'])]),
                    _tier_row(grid, 'available_P', 'Available P',
                        [('Deficient <11',     lambda x: x < 11,             COLORS['danger']),
                         ('Optimum 11–35',     lambda x: (x>=11)&(x<=35),   COLORS['success']),
                         ('Excess/Locked >35', lambda x: x > 35,             COLORS['warning'])]),
                    _tier_row(grid, 'K2O', 'K₂O',
                        [('Deficient <55',  lambda x: x < 55,               COLORS['danger']),
                         ('Optimum 55–110', lambda x: (x>=55)&(x<=110),    COLORS['success']),
                         ('High >110',      lambda x: x > 110,              '#3b82f6')]),
                    _tier_row(grid, 'EC', 'EC Salinity',
                        [('Safe <0.25',       lambda x: x < 0.25,            COLORS['success']),
                         ('Marginal 0.25–0.75', lambda x: (x>=0.25)&(x<=0.75), COLORS['warning']),
                         ('Saline >0.75',    lambda x: x > 0.75,             COLORS['danger'])]),
                ])
            ], style={**CARD_STYLE, 'flex': '1.3', 'marginRight': '16px'}),

            html.Div([
                html.Div('Risk Assessment', style={
                    'color': COLORS['text'], 'fontSize': '14px',
                    'fontWeight': '600', 'marginBottom': '15px'
                }),
                html.Div([
                    html.Div([
                        html.Div(r_name, style={
                            'color': COLORS['text_muted'],
                            'fontSize': '11px', 'marginBottom': '4px'
                        }),
                        html.Div([
                            html.Div(style={
                                'height': '8px', 'borderRadius': '4px',
                                'backgroundColor': COLORS['border'],
                                'flex': '1', 'marginRight': '10px',
                                'overflow': 'hidden'
                            }, children=[
                                html.Div(style={
                                    'height': '8px', 'width': f"{r_val}%",
                                    'borderRadius': '4px',
                                    'backgroundColor': (
                                        COLORS['danger'] if r_val > 70
                                        else COLORS['warning'] if r_val > 40
                                        else COLORS['success']
                                    )
                                })
                            ]),
                            html.Span(f"{r_val:.0f}/100",
                                      style={'color': COLORS['text'], 'fontSize': '12px',
                                             'fontWeight': '600', 'minWidth': '50px'})
                        ], style={'display': 'flex', 'alignItems': 'center',
                                  'marginBottom': '15px'})
                    ])
                    for r_name, r_val in [
                        ('Soil Degradation',
                         risk['degradation_risk'].mean() if not risk.empty else 0),
                        ('Crop Failure',
                         risk['crop_failure_risk'].mean() if not risk.empty else 0),
                        ('Salinity Stress',
                         risk['salinity_risk'].mean() if not risk.empty else 0),
                        ('Overall Risk',
                         risk['overall_risk'].mean() if not risk.empty else 0)
                    ]
                ])
            ], style={**CARD_STYLE, 'flex': '1', 'marginRight': '16px'}),

            html.Div([
                html.Div('PAU Active Advisories', style={
                    'color': COLORS['text'], 'fontSize': '14px',
                    'fontWeight': '600', 'marginBottom': '15px'
                }),
                html.Div([
                    alert_badge(
                        f"[{row.get('rule_id', '')}] {row.get('message', '')}",
                        row.get('severity', 'INFO')
                    )
                    for _, row in adv.iterrows()
                ] if not adv.empty
                  else [alert_badge('✓ All parameters normal', 'OK')])
            ], style={**CARD_STYLE, 'flex': '1'})

        ], style={'display': 'flex', 'marginBottom': '20px'})

        fig_ndvi = go.Figure()
        zone_colors_map = {
            'Healthy Cropland (North)'   : '#10b981',
            'Stressed Cropland (Central)': '#f59e0b',
            'Peri-urban SE'              : '#ef4444',
            'Vegetation West'            : '#3b82f6'
        }
        for zone in zones:
            if zone in zone_colors_map:
                fig_ndvi.add_trace(go.Scatter(
                    x=ndvi_df['month'], y=ndvi_df[zone],
                    name=zone, mode='lines+markers',
                    line=dict(color=zone_colors_map[zone], width=2),
                    marker=dict(size=6)
                ))
        fig_ndvi.add_hline(
            y=0.40, line_dash='dash', line_color='#ef4444',
            annotation_text='Seasonal Fallow Base (Post-Harvest)'
        )
        fig_ndvi.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['card'],
            plot_bgcolor=COLORS['card'],
            height=280,
            margin=dict(l=40, r=20, t=30, b=60),
            legend=dict(orientation='h', y=-0.35, x=0,
                        font=dict(size=10, color=COLORS['text_muted'])),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            title=dict(text='NDVI Time Series — Khamanon Block',
                       font=dict(size=12, color=COLORS['text']))
        )

        return html.Div([
            html.Div('Overview', style={
                'fontSize': '20px', 'fontWeight': '700',
                'color': COLORS['text'], 'marginBottom': '6px'
            }),
            html.Div(
                f"Last updated: {st.get('last_run', 'Never')}  ·  "
                f"Sentinel-2: {s2_date}  ·  Auto-refresh: 30s",
                style={'color': COLORS['text_muted'], 'fontSize': '12px',
                       'marginBottom': '20px'}
            ),
            cards, soil_stats,
            html.Div([
                dcc.Graph(figure=fig_ndvi, style={'height': '280px'})
            ], style=CARD_STYLE),

            # ── SATELLITE ACQUISITION LOG TABLE ─────────────────────────
            html.Div([
                html.Div([
                    html.Span('🛰 Sentinel-2 Acquisition Log', style={
                        'color': COLORS['text'], 'fontSize': '14px',
                        'fontWeight': '600'
                    }),
                    html.Span(
                        f'  ·  Last 30 days  ·  '
                        f'QA: Rabi <25% cloud  |  Kharif <50% cloud  ·  '
                        f'{len(sat_log)} orbital passes',
                        style={'color': COLORS['text_muted'], 'fontSize': '11px'}
                    ),
                ], style={'marginBottom': '14px'}),

                html.Table([
                    html.Thead(html.Tr([
                        html.Th(h, style={
                            'color'        : COLORS['text_muted'],
                            'fontSize'     : '11px',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'padding'      : '8px 14px',
                            'textAlign'    : 'left',
                            'borderBottom' : f"1px solid {COLORS['border']}"
                        }) for h in ['Date', 'Satellite', 'Cloud Cover', 'Season', 'Status', 'Action']
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(row['date'],
                                    style={'color': COLORS['text_muted'],
                                           'padding': '8px 14px', 'fontSize': '12px',
                                           'whiteSpace': 'nowrap'}),
                            html.Td(row['satellite'],
                                    style={'color': COLORS['text'],
                                           'padding': '8px 14px', 'fontSize': '12px'}),
                            html.Td(
                                html.Span(f"{row['cloud_pct']}%", style={
                                    'color': (COLORS['danger'] if row['cloud_pct'] > row['threshold'] else COLORS['success']),
                                    'fontWeight': '600', 'fontSize': '12px'
                                }),
                                style={'padding': '8px 14px'}
                            ),
                            html.Td(row['season'],
                                    style={'color': COLORS['text_muted'],
                                           'padding': '8px 14px', 'fontSize': '12px'}),
                            html.Td(_sat_status_badge(row['status']),
                                    style={'padding': '8px 14px'}),
                            html.Td(row['action'],
                                    style={'color': COLORS['text_muted'],
                                           'padding': '8px 14px', 'fontSize': '11px'}),
                        ], style={
                            'borderBottom': f"1px solid {COLORS['border']}",
                            'backgroundColor': ('rgba(239,68,68,0.04)' if row['status'] == 'REJECTED' else 'transparent')
                        })
                        for _, row in sat_log.sort_values('date', ascending=False).head(25).iterrows()
                    ] if not sat_log.empty else [
                        html.Tr([html.Td(
                            'Run realtime_updater.py to populate log.',
                            colSpan=6,
                            style={'color': COLORS['text_muted'],
                                   'padding': '16px 14px', 'fontSize': '12px'}
                        )])
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse'})

            ], style={**CARD_STYLE, 'marginTop': '20px'}),
        ])

    # ==========================================
    # TAB 2 — SOIL MAPS
    # ==========================================
    elif tab == 'soilmaps':

        soil_props = {
            'pH'          : {'unit': 'pH',      'cmap': 'RdYlGn_r'},
            'OC'          : {'unit': '%',        'cmap': 'YlGn'},
            'EC'          : {'unit': 'dS/m',     'cmap': 'OrRd'},
            'available_N' : {'unit': 'kg/ha',    'cmap': 'Blues'},
            'available_P' : {'unit': 'kg/ha',    'cmap': 'Purples'},
            'K2O'         : {'unit': 'kg/ha',    'cmap': 'BuGn'},
            'CEC'         : {'unit': 'meq/100g', 'cmap': 'PuBu'},
            'bulk_density': {'unit': 'g/L',      'cmap': 'YlOrBr'},
            'CaCO3'       : {'unit': '%',        'cmap': 'RdPu'}
        }

        return html.Div([
            html.Div('Soil Property Maps', style={
                'fontSize': '20px', 'fontWeight': '700',
                'color': COLORS['text'], 'marginBottom': '6px'
            }),
            html.Div(
                f"{len(grid):,} prediction points  ·  10m resolution  ·  "
                f"Random Forest prediction  ·  S2: {s2_date}",
                style={'color': COLORS['text_muted'], 'fontSize': '12px',
                       'marginBottom': '20px'}
            ),
            html.Div([
                html.Label('Select Soil Property', style={
                    'color': COLORS['text_muted'], 'fontSize': '11px',
                    'fontWeight': '600', 'textTransform': 'uppercase',
                    'letterSpacing': '0.8px', 'display': 'block',
                    'marginBottom': '8px'
                }),
                dcc.Dropdown(
                    id='soil-prop',
                    options=[{'label': p.replace('_', ' ').title() + ' (' + v['unit'] + ')',
                              'value': p}
                             for p, v in soil_props.items()],
                    value='pH', clearable=False,
                    style={'width': '320px', 'backgroundColor': COLORS['card'],
                           'color': COLORS['text'],
                           'border': f"1px solid {COLORS['border']}",
                           'borderRadius': '8px'}
                )
            ], style={**CARD_STYLE, 'marginBottom': '16px', 'display': 'inline-block'}),

            html.Div([
                html.Div([dcc.Graph(id='soil-map-v2', style={'height': '520px'})],
                         style={**CARD_STYLE, 'flex': '2', 'marginRight': '16px'}),
                html.Div(id='soil-stats-v2',
                         style={**CARD_STYLE, 'flex': '0.6', 'minWidth': '200px'})
            ], style={'display': 'flex'})
        ])

    # ==========================================
    # TAB 3 — CROP MONITOR
    # ==========================================
    elif tab == 'crop':

        ndvi_status = (
            '⚠️ SEVERE STRESS'  if ndvi_mean < 0.20
            else '🌱 MODERATE'    if ndvi_mean < 0.35
            else '🌿 HEALTHY'     if ndvi_mean < 0.60
            else '🌿 VERY HEALTHY'
        )

        fig = go.Figure()
        zone_colors_map = {
            'Healthy Cropland (North)'   : '#10b981',
            'Stressed Cropland (Central)': '#f59e0b',
            'Peri-urban SE'              : '#ef4444',
            'Vegetation West'            : '#3b82f6'
        }
        zone_markers = {
            'Healthy Cropland (North)'   : 'circle',
            'Stressed Cropland (Central)': 'square',
            'Peri-urban SE'              : 'triangle-up',
            'Vegetation West'            : 'diamond'
        }

        for zone in zones:
            if zone in zone_colors_map:
                fig.add_trace(go.Scatter(
                    x=ndvi_df['month'], y=ndvi_df[zone],
                    name=zone, mode='lines+markers',
                    line=dict(color=zone_colors_map[zone], width=2.5),
                    marker=dict(symbol=zone_markers.get(zone, 'circle'), size=9)
                ))

        fig.add_vrect(x0='Jan-2025', x1='Mar-2025',
                      fillcolor='rgba(245,158,11,0.1)',
                      annotation_text='Rabi 2024-25',
                      annotation_font_color=COLORS['text_muted'])
        fig.add_vrect(x0='Jun-2025', x1='Oct-2025',
                      fillcolor='rgba(16,185,129,0.1)',
                      annotation_text='Kharif 2025',
                      annotation_font_color=COLORS['text_muted'])
        fig.add_vrect(x0='Jan-2026', x1='Mar-2026',
                      fillcolor='rgba(245,158,11,0.1)',
                      annotation_text='Rabi 2025-26',
                      annotation_font_color=COLORS['text_muted'])
        fig.add_hline(y=0.40, line_dash='dash', line_color=COLORS['danger'],
                      annotation_text='Seasonal Fallow Base (Post-Harvest) 0.40',
                      annotation_font_color=COLORS['danger'])
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['card'],
            plot_bgcolor=COLORS['card'],
            height=460,
            margin=dict(l=50, r=30, t=50, b=100),
            legend=dict(orientation='h', y=-0.30, x=0, font=dict(size=11)),
            xaxis=dict(tickangle=45, tickfont=dict(size=10)),
            yaxis=dict(range=[0, 1.0], title='NDVI', gridcolor=COLORS['border']),
            title=dict(
                text='Real Sentinel-2 NDVI Time Series (2025-2026) — Khamanon Block',
                font=dict(size=14, color=COLORS['text'])
            )
        )

        return html.Div([
            html.Div('Crop Growth Monitor', style={
                'fontSize': '20px', 'fontWeight': '700',
                'color': COLORS['text'], 'marginBottom': '6px'
            }),
            html.Div('15 months · Real Sentinel-2 · 4 zones · 5-day revisit',
                     style={'color': COLORS['text_muted'], 'fontSize': '12px',
                            'marginBottom': '20px'}),
            html.Div([
                html.Div([
                    html.Span('Current Status: ',
                              style={'color': COLORS['text_muted'], 'fontSize': '13px'}),
                    html.Span(ndvi_status,
                              style={'fontWeight': '700', 'fontSize': '14px',
                                     'color': COLORS['text']}),
                    html.Span(f"  ·  Block NDVI: {ndvi_mean:.3f}  ·  S2 Date: {s2_date}",
                              style={'color': COLORS['text_muted'], 'fontSize': '12px'})
                ])
            ], style={**CARD_STYLE, 'marginBottom': '16px',
                      'borderLeft': f"4px solid {COLORS['accent']}"}),
            html.Div([dcc.Graph(figure=fig)], style=CARD_STYLE)
        ])

    # ==========================================
    # TAB 4 — PAU ADVISORY
    # ==========================================
    elif tab == 'advisory':

        sev_colors = {
            'CRITICAL': COLORS['danger'],
            'WARNING' : COLORS['warning'],
            'INFO'    : '#3b82f6',
            'OK'      : COLORS['success']
        }

        adv_cards = []
        if not adv.empty:
            for _, row in adv.iterrows():
                c = sev_colors.get(str(row.get('severity', 'INFO')), '#3b82f6')
                adv_cards.append(html.Div([
                    html.Div([
                        html.Span(f"[{row.get('rule_id', '')}]",
                                  style={'backgroundColor': c + '22', 'color': c,
                                         'padding': '2px 8px', 'borderRadius': '4px',
                                         'fontSize': '11px', 'fontWeight': '700',
                                         'marginRight': '10px'}),
                        html.Span(row.get('message', ''),
                                  style={'color': c, 'fontWeight': '600', 'fontSize': '13px'})
                    ], style={'marginBottom': '8px', 'display': 'flex', 'alignItems': 'center'}),
                    html.Div(row.get('action', ''),
                             style={'color': COLORS['text_muted'], 'fontSize': '12px',
                                    'lineHeight': '1.6'})
                ], style={**CARD_STYLE, 'marginBottom': '12px',
                          'borderLeft': f'3px solid {c}'}))

        source_card = html.Div([
            html.Div('Data Source', style={
                'color': COLORS['text'], 'fontWeight': '600',
                'fontSize': '13px', 'marginBottom': '10px'
            }),
            html.Div([
                html.Div('PAU Package of Practices Kharif 2025 (Vol. 42)',
                         style={'color': COLORS['text_muted'], 'fontSize': '12px',
                                'marginBottom': '4px'}),
                html.Div('PAU Package of Practices Rabi 2025',
                         style={'color': COLORS['text_muted'], 'fontSize': '12px',
                                'marginBottom': '4px'}),
                html.Div('Punjab Soil Health Card — NPK + Micronutrients',
                         style={'color': COLORS['text_muted'], 'fontSize': '12px'})
            ])
        ], style={**CARD_STYLE, 'marginBottom': '20px'})

        return html.Div([
            html.Div('PAU Advisory Engine', style={
                'fontSize': '20px', 'fontWeight': '700',
                'color': COLORS['text'], 'marginBottom': '6px'
            }),
            html.Div(
                'Source: PAU Package of Practices 2025  ·  Punjab Soil Health Card  ·  '
                f"{len(adv)} block-level advisories",
                style={'color': COLORS['text_muted'], 'fontSize': '12px',
                       'marginBottom': '20px'}
            ),
            source_card,
            html.Div(adv_cards)
        ])

    # ==========================================
    # TAB 5 — FIELD POINTS
    # ==========================================
    elif tab == 'points':

        fig = go.Figure()
        fig.add_trace(go.Scattermap(
            lat=soil['latitude'], lon=soil['longitude'],
            mode='markers',
            marker=dict(
                size=10, color=soil['pH'],
                colorscale='RdYlGn_r',
                colorbar=dict(
                    title=dict(text='pH', font=dict(color=COLORS['text'])),
                    tickfont=dict(color=COLORS['text']),
                    thickness=12
                ),
                showscale=True, opacity=0.85
            ),
            text=soil.apply(
                lambda r: f"<b>{r['sample_id']}</b><br>"
                          f"pH: {r['pH']:.2f}<br>"
                          f"OC: {r['OC']:.3f}%<br>"
                          f"EC: {r['EC']:.3f} dS/m<br>"
                          f"N: {r['available_N']:.1f} kg/ha<br>"
                          f"K: {r['K2O']:.1f} kg/ha",
                axis=1
            ),
            hoverinfo='text'
        ))
        fig.update_layout(
            map=dict(style='dark', center=dict(lat=30.795, lon=76.352), zoom=11),
            paper_bgcolor=COLORS['card'],
            height=580,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        return html.Div([
            html.Div('Field Sample Points', style={
                'fontSize': '20px', 'fontWeight': '700',
                'color': COLORS['text'], 'marginBottom': '6px'
            }),
            html.Div('208 cLHS sample points  ·  Coloured by pH value  ·  Hover for soil profile',
                     style={'color': COLORS['text_muted'], 'fontSize': '12px',
                            'marginBottom': '16px'}),
            html.Div([dcc.Graph(figure=fig)], style=CARD_STYLE)
        ])

    # ==========================================
    # TAB 6 — ANALYTICS
    # ==========================================
    elif tab == 'analytics':

        shap_chart = html.Div()
        if not shap.empty and 'pH' in shap.index:
            shap_pH = shap.loc['pH'].sort_values(ascending=True)
            fig_shap = go.Figure(go.Bar(
                x=shap_pH.values, y=shap_pH.index,
                orientation='h',
                marker=dict(color=shap_pH.values, colorscale='Teal', showscale=False)
            ))
            fig_shap.update_layout(
                template='plotly_dark',
                paper_bgcolor=COLORS['card'],
                plot_bgcolor=COLORS['card'],
                height=300,
                margin=dict(l=100, r=20, t=40, b=40),
                title=dict(text='SHAP Feature Importance — pH',
                           font=dict(size=13, color=COLORS['text'])),
                xaxis=dict(title='Mean |SHAP|', gridcolor=COLORS['border'])
            )
            shap_chart = dcc.Graph(figure=fig_shap)

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=val['soil_property'], y=val['R2'],
            name='Test R2', marker_color=COLORS['accent'], opacity=0.85
        ))
        fig_acc.add_trace(go.Bar(
            x=val['soil_property'], y=val['CV_R2'],
            name='CV R2 (5-fold)', marker_color=COLORS['accent2'], opacity=0.85
        ))
        fig_acc.add_hline(y=0.5, line_dash='dash', line_color=COLORS['warning'],
                          annotation_text='Moderate (0.5)')
        fig_acc.update_layout(
            barmode='group', template='plotly_dark',
            paper_bgcolor=COLORS['card'], plot_bgcolor=COLORS['card'],
            height=320, margin=dict(l=50, r=20, t=50, b=60),
            title=dict(text='RF Model Accuracy — All Soil Properties',
                       font=dict(size=13, color=COLORS['text'])),
            xaxis=dict(tickangle=30),
            yaxis=dict(range=[-0.5, 1.0], gridcolor=COLORS['border']),
            legend=dict(font=dict(color=COLORS['text_muted']))
        )

        soil_num = soil[['pH', 'OC', 'EC', 'K2O', 'available_P', 'available_N', 'CEC']]
        corr = soil_num.corr().round(2)
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale='RdBu', zmid=0,
            text=corr.values, texttemplate='%{text}',
            textfont=dict(size=10), showscale=True
        ))
        fig_corr.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['card'], plot_bgcolor=COLORS['card'],
            height=320, margin=dict(l=80, r=20, t=50, b=60),
            title=dict(text='Soil Property Correlations',
                       font=dict(size=13, color=COLORS['text']))
        )

        return html.Div([
            html.Div('Analytics', style={
                'fontSize': '20px', 'fontWeight': '700',
                'color': COLORS['text'], 'marginBottom': '6px'
            }),
            html.Div('SHAP explainability  ·  Model validation  ·  Soil correlations',
                     style={'color': COLORS['text_muted'], 'fontSize': '12px',
                            'marginBottom': '20px'}),
            html.Div([
                html.Div([html.Div(shap_chart, style=CARD_STYLE)],
                         style={'flex': '1', 'marginRight': '16px'}),
                html.Div([html.Div([dcc.Graph(figure=fig_acc)], style=CARD_STYLE)],
                         style={'flex': '1'})
            ], style={'display': 'flex', 'marginBottom': '16px'}),
            html.Div([dcc.Graph(figure=fig_corr)], style=CARD_STYLE)
        ])

    # ==========================================
    # TAB 7 — FIELD OPERATIONS MONITOR
    # ==========================================
    elif tab == 'fieldops':

        ts_path  = os.path.join(base, '..', 'data', 'multiindex_timeseries_clean.csv')
        ev_path  = os.path.join(base, '..', 'data', 'field_events.csv')
        fst_path = os.path.join(base, '..', 'data', 'field_ops_status.json')

        if not os.path.exists(ts_path):
            return html.Div([
                html.Div('Field Operations data not found.',
                         style={'color': COLORS['danger'], 'fontSize': '16px',
                                'marginBottom': '8px'}),
                html.Div('Run field_ops_detector.py first, then refresh the dashboard.',
                         style={'color': COLORS['text_muted']})
            ], style={**CARD_STYLE, 'marginTop': '40px'})

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
            return html.Div([
                html.Div(title, style={
                    'color': COLORS['text_muted'], 'fontSize': '11px',
                    'fontWeight': '600', 'textTransform': 'uppercase',
                    'letterSpacing': '0.8px', 'marginBottom': '8px'
                }),
                html.Div(val, style={
                    'color': col, 'fontSize': '18px', 'fontWeight': '700',
                    'lineHeight': '1.2', 'marginBottom': '4px'
                }),
                html.Div(sub, style={'color': COLORS['text_muted'], 'fontSize': '11px'})
            ], style={**CARD_STYLE, 'flex': '1', 'minWidth': '140px',
                      'borderTop': f'3px solid {col}'})

        status_row = html.Div([
            fops_card('Current Phase', current_phase,
                      f"S2: {fops_st.get('latest_date', '—')}", phase_col),
            fops_card('Last Detected Event', last_label, last_date_str, last_col),
            fops_card('Next Expected', 'Rice Transplanting', 'June - July 2026', COLORS['success']),
            fops_card('S2 Images Used', str(fops_st.get('total_images', len(ts_df))),
                      'Jan 2025 to May 2026', COLORS['accent']),
            fops_card('Events Detected', str(fops_st.get('total_events', len(ev_df))),
                      '2 complete crop cycles', COLORS['accent2']),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '20px', 'flexWrap': 'wrap'})

        fig_main = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.70, 0.30], vertical_spacing=0.06,
            subplot_titles=(
                'Spectral indices — NDVI · NBR · BSI',
                'Detected field operations timeline'
            )
        )

        for x0, x1, sc in [
            ('2025-01-01', '2025-05-15', '#1D9E75'),
            ('2025-06-01', '2025-11-30', '#378ADD'),
            ('2025-12-01', '2026-05-26', '#1D9E75'),
        ]:
            fig_main.add_shape(
                type='rect', xref='x', yref='y',
                x0=x0, x1=x1, y0=-0.25, y1=0.92,
                fillcolor=sc, opacity=0.07, line_width=0, layer='below', row=1, col=1
            )

        fig_main.add_trace(go.Scatter(
            x=ts_df['date'], y=ts_df['NDVI'], name='NDVI', mode='lines',
            line=dict(color='#10b981', width=2.5),
            hovertemplate='<b>NDVI</b>: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        fig_main.add_trace(go.Scatter(
            x=ts_df['date'], y=ts_df['NBR'], name='NBR', mode='lines',
            line=dict(color='#ef4444', width=1.8, dash='dash'),
            hovertemplate='<b>NBR</b>: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        fig_main.add_trace(go.Scatter(
            x=ts_df['date'], y=ts_df['BSI'], name='BSI', mode='lines',
            line=dict(color='#f59e0b', width=1.8, dash='dot'),
            hovertemplate='<b>BSI</b>: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        fig_main.add_shape(
            type='line', xref='x', yref='y',
            x0=ts_df['date'].min(), x1=ts_df['date'].max(), y0=0, y1=0,
            line=dict(color='rgba(255,255,255,0.12)', width=1), row=1, col=1
        )

        if not ev_df.empty:
            for _, erow in ev_df.iterrows():
                ec = EV_COLOR.get(erow['event'], '#94a3b8')
                fig_main.add_shape(
                    type='line', xref='x', yref='y',
                    x0=erow['date'], x1=erow['date'], y0=-0.25, y1=0.92,
                    line=dict(color=ec, width=1.5, dash='dot'), opacity=0.85, row=1, col=1
                )
            active_ytypes = ev_df['event'].unique()
            for etype in EV_COLOR:
                if etype not in active_ytypes:
                    continue
                sub = ev_df[ev_df['event'] == etype]
                sizes = [18 if c == 'HIGH' else 12 for c in sub['confidence']]
                fig_main.add_trace(go.Scatter(
                    x=sub['date'],
                    y=[EV_YPOS.get(etype, 0)] * len(sub),
                    mode='markers',
                    name=EV_LABEL.get(etype, etype),
                    showlegend=True,
                    marker=dict(color=EV_COLOR[etype], size=sizes, symbol='diamond',
                                line=dict(color='white', width=1.5)),
                    hovertemplate=(
                        f"<b>{EV_LABEL.get(etype, etype)}</b><br>"
                        "%{x|%d %b %Y}<br><extra></extra>"
                    )
                ), row=2, col=1)

        for x_pos, lbl, lc in [
            ('2025-02-20', 'Rabi 2024-25', '#1D9E75'),
            ('2025-08-15', 'Kharif 2025',  '#378ADD'),
            ('2026-02-15', 'Rabi 2025-26', '#1D9E75'),
        ]:
            fig_main.add_annotation(
                x=x_pos, y=3.6, text=lbl, showarrow=False,
                font=dict(color=lc, size=9), row=2, col=1
            )

        fig_main.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['card'], plot_bgcolor=COLORS['card'],
            height=580, margin=dict(l=55, r=20, t=55, b=70),
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', x=0, y=-0.13,
                        font=dict(size=11, color=COLORS['text_muted'])),
            hovermode='x unified', font=dict(color=COLORS['text'])
        )
        fig_main.update_xaxes(
            gridcolor='rgba(255,255,255,0.05)', tickformat='%b %Y',
            showgrid=True, tickfont=dict(size=10)
        )
        fig_main.update_yaxes(
            gridcolor='rgba(255,255,255,0.05)', range=[-0.25, 0.92],
            title_text='Index value', title_font=dict(size=11), row=1, col=1
        )
        fig_main.update_yaxes(
            showgrid=False, tickvals=[1, 2, 3],
            ticktext=['Rice Transplant', 'Burning', 'Harvest'],
            range=[-0.6, 4.0], tickfont=dict(size=10), row=2, col=1
        )

        table_rows = []
        if not ev_df.empty:
            for _, erow in ev_df.iterrows():
                ec = EV_COLOR.get(erow['event'], COLORS['text_muted'])
                el = EV_LABEL.get(erow['event'], erow['event'])
                ei = EV_ICON.get(erow['event'], '')
                cc = COLORS['success'] if erow['confidence'] == 'HIGH' else COLORS['warning']
                table_rows.append(html.Tr([
                    html.Td(erow['date'].strftime('%d %b %Y'),
                            style={'color': COLORS['text_muted'], 'padding': '10px 14px',
                                   'whiteSpace': 'nowrap', 'fontSize': '13px'}),
                    html.Td(html.Span(f"{ei}  {el}", style={
                        'backgroundColor': ec + '22', 'color': ec,
                        'padding': '3px 12px', 'borderRadius': '20px',
                        'fontSize': '12px', 'fontWeight': '500'
                    }), style={'padding': '10px 14px'}),
                    html.Td(html.Span(erow['confidence'],
                                      style={'color': cc, 'fontWeight': '600',
                                             'fontSize': '13px'}),
                            style={'padding': '10px 14px'}),
                    html.Td(str(erow.get('note', '')),
                            style={'color': COLORS['text_muted'], 'padding': '10px 14px',
                                   'fontSize': '12px'}),
                ], style={'borderBottom': f"1px solid {COLORS['border']}"}))

        event_table = html.Div([
            html.Div('Detected field operations — complete log',
                     style={'color': COLORS['text'], 'fontSize': '14px',
                            'fontWeight': '600', 'marginBottom': '16px'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th(h, style={
                        'color': COLORS['text_muted'], 'fontSize': '11px',
                        'textTransform': 'uppercase', 'letterSpacing': '1px',
                        'padding': '10px 14px', 'textAlign': 'left',
                        'borderBottom': f"1px solid {COLORS['border']}"
                    }) for h in ['Date', 'Field Operation', 'Confidence', 'Scientific basis']
                ])),
                html.Tbody(table_rows if table_rows else [
                    html.Tr([html.Td(
                        'No events detected yet. Run field_ops_detector.py.',
                        colSpan=4,
                        style={'color': COLORS['text_muted'], 'padding': '20px 14px',
                               'fontSize': '13px'}
                    )])
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ], style=CARD_STYLE)

        return html.Div([
            html.Div('Field Operations Monitor', style={
                'fontSize': '20px', 'fontWeight': '700',
                'color': COLORS['text'], 'marginBottom': '6px'
            }),
            html.Div(
                '16 months  ·  Sentinel-2  ·  5 spectral indices  ·  '
                'Automated change detection  ·  Jan 2025 to May 2026',
                style={'color': COLORS['text_muted'], 'fontSize': '12px',
                       'marginBottom': '20px'}
            ),
            status_row,
            html.Div([dcc.Graph(figure=fig_main, config={'displayModeBar': False})],
                     style={**CARD_STYLE, 'marginBottom': '20px'}),
            event_table,
        ])

    # ==========================================
    # TAB 8 — LAND COVER MONITOR
    # ==========================================
    elif tab == 'landcover':

        lulc_path = os.path.join(base, '..', 'data', 'lulc_summary.json')
        ev_t_path = os.path.join(base, '..', 'data', 'field_events_tagged.csv')

        if not os.path.exists(lulc_path):
            return html.Div([
                html.Div('Land Cover data not found.',
                         style={'color': COLORS['danger'], 'fontSize': '16px',
                                'marginBottom': '8px'}),
                html.Div('Run lulc_event_tagger.py first, then refresh.',
                         style={'color': COLORS['text_muted']})
            ], style={**CARD_STYLE, 'marginTop': '40px'})

        with open(lulc_path, encoding='utf-8') as f:
            lulc = json.load(f)

        ev_t = pd.read_csv(ev_t_path) if os.path.exists(ev_t_path) else pd.DataFrame()
        rabi   = lulc['rabi_2025_26']
        kharif = lulc['kharif_2025']

        def lc_card(title, val, sub, col):
            return html.Div([
                html.Div(title, style={
                    'color': COLORS['text_muted'], 'fontSize': '11px',
                    'fontWeight': '600', 'textTransform': 'uppercase',
                    'letterSpacing': '0.8px', 'marginBottom': '8px'
                }),
                html.Div(val, style={'color': col, 'fontSize': '18px',
                                     'fontWeight': '700', 'lineHeight': '1.2',
                                     'marginBottom': '4px'}),
                html.Div(sub, style={'color': COLORS['text_muted'], 'fontSize': '11px'})
            ], style={**CARD_STYLE, 'flex': '1', 'minWidth': '140px',
                      'borderTop': f'3px solid {col}'})

        lc_cards = html.Div([
            lc_card('Block Area', f"{lulc['block_area_ha']:,.0f} ha",
                    'Total Khamanon Block', COLORS['accent']),
            lc_card('Rabi Dominant', f"Wheat {rabi['classes']['Wheat']['pct']}%",
                    f"{rabi['classes']['Wheat']['area_ha']:,.0f} ha · Nov-May", '#fbbf24'),
            lc_card('Spring Maize', f"{rabi['classes']['Spring_Maize']['area_ha']:,.0f} ha",
                    f"{rabi['classes']['Spring_Maize']['pct']}% of Rabi season", '#16a34a'),
            lc_card('Kharif Dominant', f"Rice {kharif['classes']['Rice']['pct']}%",
                    f"{kharif['classes']['Rice']['area_ha']:,.0f} ha · Jun-Nov", '#22c55e'),
            lc_card('Kharif Maize', f"{kharif['classes']['Kharif_Maize']['area_ha']:,.0f} ha",
                    f"{kharif['classes']['Kharif_Maize']['pct']}% of Kharif season", '#fb923c'),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '20px', 'flexWrap': 'wrap'})

        rabi_classes   = list(rabi['classes'].keys())
        kharif_classes = list(kharif['classes'].keys())
        rabi_areas     = [rabi['classes'][c]['area_ha']  for c in rabi_classes]
        rabi_colors    = [rabi['classes'][c]['color']    for c in rabi_classes]
        kharif_areas   = [kharif['classes'][c]['area_ha'] for c in kharif_classes]
        kharif_colors  = [kharif['classes'][c]['color']   for c in kharif_classes]

        fig_compare = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Rabi 2025-26 · {rabi['model_accuracy']}% accuracy · S2",
                f"Kharif 2025 · {kharif['model_accuracy']}% accuracy · S2+SAR"
            ),
            horizontal_spacing=0.10
        )
        fig_compare.add_trace(go.Bar(
            x=rabi_classes, y=rabi_areas, marker_color=rabi_colors,
            text=[f"{a:,.0f} ha\n{rabi['classes'][c]['pct']}%"
                  for c, a in zip(rabi_classes, rabi_areas)],
            textposition='outside', textfont=dict(color=COLORS['text'], size=10),
            showlegend=False, hovertemplate='%{x}<br>%{y:,.0f} ha<extra></extra>'
        ), row=1, col=1)
        fig_compare.add_trace(go.Bar(
            x=kharif_classes, y=kharif_areas, marker_color=kharif_colors,
            text=[f"{a:,.0f} ha\n{kharif['classes'][c]['pct']}%"
                  for c, a in zip(kharif_classes, kharif_areas)],
            textposition='outside', textfont=dict(color=COLORS['text'], size=10),
            showlegend=False, hovertemplate='%{x}<br>%{y:,.0f} ha<extra></extra>'
        ), row=1, col=2)
        fig_compare.update_layout(
            template='plotly_dark', paper_bgcolor=COLORS['card'],
            plot_bgcolor=COLORS['card'], height=340,
            margin=dict(l=50, r=30, t=60, b=60), font=dict(color=COLORS['text'])
        )
        fig_compare.update_yaxes(title_text='Area (ha)',
                                 gridcolor='rgba(255,255,255,0.05)')
        fig_compare.update_xaxes(tickfont=dict(size=11))

        rot      = lulc['cropping_rotation']
        r_labels = [r['rotation'] for r in rot]
        r_areas  = [r['area_est'] for r in rot]
        r_colors = [r['color']    for r in rot]
        r_pcts   = [r['pct']      for r in rot]

        fig_rot = go.Figure(go.Bar(
            x=r_areas, y=r_labels, orientation='h', marker_color=r_colors,
            text=[f"{a:,} ha  ({p}%)" for a, p in zip(r_areas, r_pcts)],
            textposition='outside', textfont=dict(color=COLORS['text'], size=11),
            hovertemplate='%{y}<br>~%{x:,} ha<extra></extra>'
        ))
        fig_rot.update_layout(
            template='plotly_dark', paper_bgcolor=COLORS['card'],
            plot_bgcolor=COLORS['card'], height=280,
            margin=dict(l=220, r=120, t=30, b=40), font=dict(color=COLORS['text']),
            xaxis=dict(title='Estimated area (ha)', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(autorange='reversed', tickfont=dict(size=11)),
            title=dict(
                text='Cropping rotation patterns · based on 138 ground-truth points',
                font=dict(size=13, color=COLORS['text_muted']), x=0, pad=dict(l=10)
            )
        )

        CLASS_COLOR_MAP = {
            'Wheat'               : '#fbbf24',
            'Wheat (post-harvest)': '#f59e0b',
            'Spring Maize'        : '#16a34a',
            'Rice'                : '#22c55e',
            'Rice (transplanted)' : '#00d4aa',
            'Rice (post-harvest)' : '#ef4444',
            'Rice (puddling)'     : '#3b82f6',
        }
        EV_ICON_MAP = {
            'HARVESTING'        : '🌾',
            'STUBBLE_BURNING'   : '🔥',
            'RICE_TRANSPLANTING': '🌱',
            'FIELD_FLOODING'    : '💧',
            'PLOUGHING'         : '🚜',
        }

        ev_rows = []
        if not ev_t.empty:
            for _, erow in ev_t.iterrows():
                cc   = CLASS_COLOR_MAP.get(str(erow.get('crop_class', '')), COLORS['text_muted'])
                cfr = COLORS['success'] if erow.get('confidence') == 'HIGH' else COLORS['warning']
                ei   = EV_ICON_MAP.get(str(erow.get('event', '')), '')
                ev_rows.append(html.Tr([
                    html.Td(str(erow.get('date', '')),
                            style={'color': COLORS['text_muted'], 'padding': '10px 14px',
                                   'whiteSpace': 'nowrap', 'fontSize': '13px'}),
                    html.Td(html.Span(
                        f"{ei}  {str(erow.get('event', '')).replace('_', ' ').title()}",
                        style={'backgroundColor': cfr + '22', 'color': cfr,
                               'padding': '3px 10px', 'borderRadius': '20px',
                               'fontSize': '12px', 'fontWeight': '500'}
                    ), style={'padding': '10px 14px'}),
                    html.Td(html.Span(str(erow.get('crop_class', '')),
                                      style={'backgroundColor': cc + '22', 'color': cc,
                                             'padding': '3px 10px', 'borderRadius': '20px',
                                             'fontSize': '12px', 'fontWeight': '500'}),
                            style={'padding': '10px 14px'}),
                    html.Td(f"{erow.get('area_ha', 0):,.0f} ha",
                            style={'color': COLORS['text'], 'padding': '10px 14px',
                                   'fontSize': '13px', 'fontWeight': '600'}),
                    html.Td(html.Span(str(erow.get('confidence', '')),
                                      style={'color': (COLORS['success']
                                                       if erow.get('confidence') == 'HIGH'
                                                       else COLORS['warning']),
                                             'fontWeight': '600', 'fontSize': '12px'}),
                            style={'padding': '10px 14px'}),
                    html.Td(str(erow.get('class_note', ''))[:80] + '...'
                            if len(str(erow.get('class_note', ''))) > 80
                            else str(erow.get('class_note', '')),
                            style={'color': COLORS['text_muted'], 'padding': '10px 14px',
                                   'fontSize': '11px'}),
                ], style={'borderBottom': f"1px solid {COLORS['border']}"}))

        ev_table = html.Div([
            html.Div('Per-class field operations log — events linked to seasonal LULC',
                     style={'color': COLORS['text'], 'fontSize': '14px',
                            'fontWeight': '600', 'marginBottom': '16px'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th(h, style={
                        'color': COLORS['text_muted'], 'fontSize': '11px',
                        'textTransform': 'uppercase', 'letterSpacing': '1px',
                        'padding': '10px 14px', 'textAlign': 'left',
                        'borderBottom': f"1px solid {COLORS['border']}"
                    }) for h in ['Date', 'Field Operation', 'Crop Class',
                                 'Area Affected', 'Confidence', 'Scientific basis']
                ])),
                html.Tbody(ev_rows if ev_rows else [
                    html.Tr([html.Td(
                        'No tagged events. Run lulc_event_tagger.py.',
                        colSpan=6,
                        style={'color': COLORS['text_muted'], 'padding': '20px 14px',
                               'fontSize': '13px'}
                    )])
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ], style=CARD_STYLE)

        acc_banner = html.Div([
            html.Div([
                html.Span('Classification methodology:  ',
                          style={'color': COLORS['text_muted'], 'fontSize': '12px'}),
                html.Span('Rabi — Supervised Random Forest · Sentinel-2 · 138 ground-truth points · ',
                          style={'color': COLORS['text'], 'fontSize': '12px'}),
                html.Span(f"{rabi['model_accuracy']}% OOB accuracy",
                          style={'color': COLORS['success'], 'fontWeight': '600',
                                 'fontSize': '12px'}),
                html.Span('   |   Kharif — SAR+Optical Fusion · Sentinel-2 + Sentinel-1 · ',
                          style={'color': COLORS['text'], 'fontSize': '12px'}),
                html.Span(f"{kharif['model_accuracy']}% OOB accuracy",
                          style={'color': COLORS['success'], 'fontWeight': '600',
                                 'fontSize': '12px'}),
            ])
        ], style={**CARD_STYLE, 'marginBottom': '20px',
                  'borderLeft': f"4px solid {COLORS['success']}"})

        return html.Div([
            html.Div('Land Cover Monitor', style={
                'fontSize': '20px', 'fontWeight': '700',
                'color': COLORS['text'], 'marginBottom': '6px'
            }),
            html.Div(
                '2 seasonal LULC maps  ·  19,616 ha  ·  Sentinel-2 + SAR fusion  ·  '
                '138 ground-truth points  ·  PAU Ludhiana MSc Research',
                style={'color': COLORS['text_muted'], 'fontSize': '12px',
                       'marginBottom': '20px'}
            ),
            lc_cards, acc_banner,
            html.Div([dcc.Graph(figure=fig_compare, config={'displayModeBar': False})],
                     style={**CARD_STYLE, 'marginBottom': '20px'}),
            html.Div([dcc.Graph(figure=fig_rot, config={'displayModeBar': False})],
                     style={**CARD_STYLE, 'marginBottom': '20px'}),
            ev_table,
            html.Div([
                html.Div([
                    html.Span('🗺 Land Cover Map',
                              style={'color': COLORS['text'], 'fontSize': '14px',
                                     'fontWeight': '600'}),
                    html.Span(
                        '  ·  Khamanon Block  ·  Toggle seasons in legend  ·  Hover for class detail',
                        style={'color': COLORS['text_muted'], 'fontSize': '12px'}
                    )
                ], style={'marginBottom': '14px'}),
                dcc.Graph(
                    id='lulc-map',
                    figure=_lulc_map_cache,
                    config={'displayModeBar': True,
                            'modeBarButtonsToRemove': ['select2d', 'lasso2d']}
                )
            ], style={**CARD_STYLE, 'marginTop': '20px'}),
        ])

    return html.Div('Tab not found')


# ============================================
# SOIL MAP CALLBACK
# ============================================

@callback(
    Output('soil-map-v2',   'figure'),
    Output('soil-stats-v2', 'children'),
    Input('soil-prop', 'value'),
    Input('refresh',   'n_intervals')
)
def update_map(prop, n):
    _, grid, _, _, st, _, _, _, _, _ = load_all()
    s2_date = st.get('sentinel2_date', '—')

    cmaps = {
        'pH': 'RdYlGn_r', 'OC': 'YlGn', 'EC': 'OrRd',
        'available_N': 'Blues', 'available_P': 'Purples',
        'K2O': 'BuGn', 'CEC': 'PuBu', 'bulk_density': 'YlOrBr', 'CaCO3': 'RdPu'
    }
    units = {
        'pH': 'pH', 'OC': '%', 'EC': 'dS/m',
        'available_N': 'kg/ha', 'available_P': 'kg/ha',
        'K2O': 'kg/ha', 'CEC': 'meq/100g', 'bulk_density': 'g/L', 'CaCO3': '%'
    }

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    grid_lon, grid_lat = transformer.transform(grid['easting'].values, grid['northing'].values)

    vals = grid[prop]
    fig  = go.Figure(data=go.Densitymap(
        lat=grid_lat, lon=grid_lon,
        z=vals, radius=3,
        colorscale=cmaps.get(prop, 'Viridis'),
        showscale=True,
        colorbar=dict(
            title=dict(text=units.get(prop, ''), font=dict(color=COLORS['text'])),
            tickfont=dict(color=COLORS['text']), thickness=12
        ),
        hovertemplate=(
            prop.replace('_', ' ').title() + ': %{z:.3f} ' +
            units.get(prop, '') + '<extra></extra>'
        )
    ))
    fig.update_layout(
        map=dict(style='dark', center=dict(lat=30.795, lon=76.352), zoom=11),
        paper_bgcolor=COLORS['card'], height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text=(prop.replace('_', ' ').title() + ' (' + units.get(prop, '') + ')'
                  + '  ·  S2: ' + s2_date),
            font=dict(size=12, color=COLORS['text'])
        )
    )

    context = {
        'pH'         : 'Mean 8.39 — strongly alkaline. Values >8.5 risk micronutrient lock-up.',
        'OC'         : 'Mean 0.43% — critically low. Healthy soil needs >0.75%.',
        'EC'         : 'Mean 0.10 dS/m — no salinity stress. Safe for all crops.',
        'available_N': 'Mean 225 kg/ha — moderate. High spatial variability detected.',
        'available_P': 'Mean 30.5 kg/ha — moderate. PAU optimal: 25-35 kg/ha.',
        'K2O'        : 'Mean 71 kg/ha — some deficiency zones detected (<55 kg/acre).',
        'CEC'        : 'Mean 14 meq/100g — typical for loamy alluvial Punjab soils.',
        'bulk_density': 'Mean 234 g/L — some compaction zones above 260 g/L.',
        'CaCO3'      : 'Mean 0.76% — locks up P and micronutrients in high zones.'
    }

    stats = [
        html.Div(prop.replace('_', ' ').title(),
                 style={'color': COLORS['text'], 'fontWeight': '700',
                        'fontSize': '15px', 'marginBottom': '16px'})
    ]

    for label, value in [
        ('Mean',    f"{vals.mean():.3f} {units.get(prop, '')}"),
        ('Min',     f"{vals.min():.3f}"),
        ('Max',     f"{vals.max():.3f}"),
        ('Std Dev', f"{vals.std():.3f}"),
        ('Points',  f"{len(vals):,}"),
        ('S2 Date', s2_date)
    ]:
        stats.append(html.Div([
            html.Div(label, style={
                'color': COLORS['text_muted'], 'fontSize': '10px',
                'fontWeight': '600', 'textTransform': 'uppercase',
                'letterSpacing': '0.6px', 'marginBottom': '2px'
            }),
            html.Div(value, style={
                'color': COLORS['text'], 'fontSize': '13px', 'fontWeight': '500',
                'marginBottom': '12px', 'paddingBottom': '12px',
                'borderBottom': f"1px solid {COLORS['border']}"
            })
        ]))

    stats.append(html.Div(
        context.get(prop, ''),
        style={
            'color': COLORS['accent'], 'fontSize': '11px', 'lineHeight': '1.6',
            'backgroundColor': COLORS['accent'] + '11', 'padding': '10px',
            'borderRadius': '6px', 'marginTop': '4px'
        }
    ))

    return fig, stats


# ============================================
# RUN EXECUTION
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