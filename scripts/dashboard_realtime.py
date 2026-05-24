# ============================================
# DIGITAL TWIN - KHAMANON BLOCK
# Script 10: Real-Time Dashboard
# Shows last update time + live status
# ============================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import json
import os
from datetime import datetime

print("=" * 55)
print("  REAL-TIME DASHBOARD — KHAMANON BLOCK")
print("=" * 55)

base = os.path.dirname(os.path.abspath(__file__))

# ============================================
# LOAD DATA FUNCTION
# Called every time dashboard refreshes
# Always reads latest files from disk
# ============================================

def load_data():
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
    validation = pd.read_csv(
        os.path.join(base,'..','data',
                     'model_validation_real.csv')
    )

    status_path = os.path.join(
        base,'..','data','last_update.json'
    )
    if os.path.exists(status_path):
        with open(status_path,'r') as f:
            status = json.load(f)
    else:
        status = {
            'last_run'      : 'Never',
            'sentinel2_date': 'Unknown',
            'images_used'   : 0,
            'ndvi_mean'     : 0,
            'alerts'        : [],
            'status'        : 'No update yet'
        }

    return soil, grid, ndvi, validation, status

soil, grid, ndvi_df, validation, status = load_data()

print(f"Data loaded.")
print(f"Last update: {status['last_run']}")

# ============================================
# CONFIG
# ============================================

soil_properties = {
    'pH'          : {'unit':'pH units','cmap':'RdYlGn_r'},
    'OC'          : {'unit':'%',       'cmap':'YlGn'    },
    'EC'          : {'unit':'dS/m',    'cmap':'OrRd'    },
    'available_N' : {'unit':'kg/ha',   'cmap':'Blues'   },
    'available_P' : {'unit':'kg/ha',   'cmap':'Purples' },
    'K2O'         : {'unit':'kg/ha',   'cmap':'BuGn'    },
    'CEC'         : {'unit':'meq/100g','cmap':'PuBu'    },
    'bulk_density': {'unit':'g/L',     'cmap':'YlOrBr'  },
    'CaCO3'       : {'unit':'%',       'cmap':'RdPu'    }
}

zone_colors = {
    'Healthy Cropland (North)'    : 'green',
    'Stressed Cropland (Central)' : 'orange',
    'Peri-urban SE'               : 'red',
    'Vegetation West'             : 'blue'
}

# ============================================
# ALERT COLOR
# ============================================

def alert_color(status_text):
    if status_text == 'SUCCESS':
        return '#2ecc71'
    elif status_text == 'No update yet':
        return '#e67e22'
    else:
        return '#e74c3c'

# ============================================
# APP LAYOUT
# ============================================

app = Dash(__name__)

app.layout = html.Div([

    # HEADER
    html.Div([
        html.Div([
            html.H1(
                'Digital Twin — Khamanon Block',
                style={'color':'white','margin':'0',
                       'fontSize':'24px',
                       'fontWeight':'bold'}
            ),
            html.P(
                'Fatehgarh Sahib, Punjab  |  '
                'Real-Time Sentinel-2  |  '
                '208 cLHS Points  |  '
                'RF Soil Prediction',
                style={'color':'#bdc3c7',
                       'margin':'4px 0 0 0',
                       'fontSize':'12px'}
            )
        ], style={'display':'inline-block',
                  'width':'70%',
                  'verticalAlign':'middle'}),

        # LIVE STATUS PANEL
        html.Div([
            html.Div(id='status-panel')
        ], style={'display':'inline-block',
                  'width':'28%',
                  'verticalAlign':'middle',
                  'textAlign':'right'})

    ], style={
        'backgroundColor':'#2c3e50',
        'padding':'15px 30px'
    }),

    # AUTO-REFRESH every 30 seconds
    dcc.Interval(
        id='refresh-interval',
        interval=30 * 1000,
        n_intervals=0
    ),

    # TABS
    dcc.Tabs(
        id='main-tabs', value='tab-status',
        children=[
            dcc.Tab(label='🔴 Live Status',
                    value='tab-status',
                    selected_style={
                        'fontWeight':'bold',
                        'borderTop':'3px solid #e74c3c',
                        'backgroundColor':'#ecf0f1'
                    }),
            dcc.Tab(label='Soil Maps',
                    value='tab-soil',
                    selected_style={
                        'fontWeight':'bold',
                        'borderTop':'3px solid #27ae60',
                        'backgroundColor':'#ecf0f1'
                    }),
            dcc.Tab(label='Crop Monitoring',
                    value='tab-crop',
                    selected_style={
                        'fontWeight':'bold',
                        'borderTop':'3px solid #27ae60',
                        'backgroundColor':'#ecf0f1'
                    }),
            dcc.Tab(label='Sample Points',
                    value='tab-samples',
                    selected_style={
                        'fontWeight':'bold',
                        'borderTop':'3px solid #27ae60',
                        'backgroundColor':'#ecf0f1'
                    }),
            dcc.Tab(label='Soil Summary',
                    value='tab-summary',
                    selected_style={
                        'fontWeight':'bold',
                        'borderTop':'3px solid #27ae60',
                        'backgroundColor':'#ecf0f1'
                    }),
        ]
    ),

    html.Div(id='tab-content',
             style={'padding':'20px 30px',
                    'backgroundColor':'#f8f9fa',
                    'minHeight':'80vh'})

], style={'fontFamily':'Segoe UI, Arial, sans-serif'})


# ============================================
# STATUS PANEL CALLBACK
# Updates every 30 seconds
# ============================================

@callback(
    Output('status-panel','children'),
    Input('refresh-interval','n_intervals')
)
def update_status(n):
    _, _, _, _, st = load_data()

    color = alert_color(st['status'])
    alerts = st.get('alerts', [])

    items = [
        html.Div([
            html.Span('● ',
                      style={'color': color,
                             'fontSize':'16px'}),
            html.Span(
                st['status'],
                style={'color':'white',
                       'fontWeight':'bold',
                       'fontSize':'13px'}
            )
        ]),
        html.Div(
            f"Last update: {st['last_run']}",
            style={'color':'#bdc3c7',
                   'fontSize':'11px',
                   'marginTop':'3px'}
        ),
        html.Div(
            f"S2 date: {st['sentinel2_date']} "
            f"| Images: {st.get('images_used',0)}",
            style={'color':'#bdc3c7',
                   'fontSize':'11px'}
        ),
        html.Div(
            f"NDVI: {st.get('ndvi_mean',0):.3f}",
            style={'color':'#bdc3c7',
                   'fontSize':'11px'}
        )
    ]

    if alerts:
        items.append(html.Div(
            f"⚠️ {len(alerts)} alert(s)",
            style={'color':'#e74c3c',
                   'fontWeight':'bold',
                   'fontSize':'12px',
                   'marginTop':'4px'}
        ))

    return html.Div(items)


# ============================================
# TAB CONTENT CALLBACK
# ============================================

@callback(
    Output('tab-content','children'),
    Input('main-tabs','value'),
    Input('refresh-interval','n_intervals')
)
def render_tab(tab, n):

    soil, grid, ndvi_df, validation, st = load_data()
    zones = [c for c in ndvi_df.columns
             if c != 'month']

    # ==========================================
    # TAB 1: LIVE STATUS
    # ==========================================
    if tab == 'tab-status':

        alerts = st.get('alerts', [])
        changes = st.get('spectral_changes', {})

        # Status card
        color  = alert_color(st['status'])

        status_cards = [
            html.Div([
                html.H2(
                    f"● {st['status']}",
                    style={'color': color,
                           'margin':'0'}
                ),
                html.P(
                    f"Last run: {st['last_run']}",
                    style={'color':'#7f8c8d',
                           'margin':'5px 0'}
                ),
                html.P(
                    f"Sentinel-2 date: "
                    f"{st['sentinel2_date']}",
                    style={'color':'#7f8c8d',
                           'margin':'5px 0'}
                ),
                html.P(
                    f"Images used: "
                    f"{st.get('images_used',0)}",
                    style={'color':'#7f8c8d',
                           'margin':'5px 0'}
                ),
            ], style={
                'backgroundColor':'white',
                'padding':'20px',
                'borderRadius':'8px',
                'boxShadow':'0 2px 8px rgba(0,0,0,0.1)',
                'marginBottom':'20px'
            })
        ]

        # Spectral index change cards
        index_cards = []
        for idx_name in ['NDVI','NDBI','SAVI','BSI']:
            if idx_name in changes:
                c = changes[idx_name]
                delta = c['delta']
                d_color = (
                    '#2ecc71' if delta > 0
                    else '#e74c3c'
                )
                arrow = '↑' if delta > 0 else '↓'
                index_cards.append(
                    html.Div([
                        html.H4(idx_name,
                                style={'margin':'0',
                                       'color':'#2c3e50'}),
                        html.P(
                            f"{c['old']} → {c['new']}",
                            style={'margin':'5px 0',
                                   'fontSize':'18px',
                                   'fontWeight':'bold'}
                        ),
                        html.P(
                            f"{arrow} {abs(delta):.4f}",
                            style={'margin':'0',
                                   'color': d_color,
                                   'fontWeight':'bold'}
                        )
                    ], style={
                        'backgroundColor':'white',
                        'padding':'15px',
                        'borderRadius':'8px',
                        'boxShadow':
                            '0 2px 8px rgba(0,0,0,0.1)',
                        'display':'inline-block',
                        'width':'22%',
                        'marginRight':'2%',
                        'textAlign':'center'
                    })
                )

        # Alert panel
        if alerts:
            alert_items = [
                html.Div([
                    html.Span(
                        f"⚠️ [{a['severity']}] "
                        f"{a['type']}: ",
                        style={'fontWeight':'bold',
                               'color':'#e74c3c'}
                    ),
                    html.Span(a['message'])
                ], style={'marginBottom':'8px'})
                for a in alerts
            ]
            alert_panel = html.Div([
                html.H4('Active Alerts',
                        style={'color':'#e74c3c'}),
                html.Div(alert_items)
            ], style={
                'backgroundColor':'#fdf2f2',
                'padding':'15px',
                'borderRadius':'8px',
                'border':'1px solid #e74c3c',
                'marginTop':'20px'
            })
        else:
            alert_panel = html.Div([
                html.H4(
                    '✅ No Active Alerts',
                    style={'color':'#27ae60',
                           'margin':'0'}
                ),
                html.P(
                    'All soil and crop indicators '
                    'within normal range.',
                    style={'color':'#7f8c8d',
                           'margin':'5px 0 0 0'}
                )
            ], style={
                'backgroundColor':'#eafaf1',
                'padding':'15px',
                'borderRadius':'8px',
                'border':'1px solid #27ae60',
                'marginTop':'20px'
            })

        # Soil predictions change table
        soil_cols = ['pH','OC','EC','K2O',
                     'available_P','available_N',
                     'CEC','bulk_density','CaCO3']

        pred_stats = []
        for col in soil_cols:
            if col in grid.columns:
                pred_stats.append({
                    'Property': col,
                    'Current Mean': round(
                        grid[col].mean(), 3),
                    'Min': round(
                        grid[col].min(), 3),
                    'Max': round(
                        grid[col].max(), 3)
                })

        fig_pred = go.Figure(data=[go.Table(
            header=dict(
                values=['Soil Property',
                        'Predicted Mean',
                        'Min', 'Max'],
                fill_color='#2c3e50',
                font=dict(color='white',size=11),
                align='center'
            ),
            cells=dict(
                values=[
                    [p['Property']
                     for p in pred_stats],
                    [p['Current Mean']
                     for p in pred_stats],
                    [p['Min']
                     for p in pred_stats],
                    [p['Max']
                     for p in pred_stats]
                ],
                fill_color=[
                    ['#ecf0f1' if i%2==0
                     else 'white'
                     for i in range(len(pred_stats))]
                ],
                align='center',
                font=dict(size=11)
            )
        )])
        fig_pred.update_layout(
            title='Current Soil Predictions '
                  '(Updated with Latest S2 Data)',
            height=380
        )

        return html.Div([
            html.H3('Live System Status',
                    style={'color':'#2c3e50',
                           'marginBottom':'5px'}),
            html.P(
                f'Dashboard auto-refreshes every '
                f'30 seconds. Run realtime_updater.py '
                f'to fetch new Sentinel-2 data.',
                style={'color':'#7f8c8d',
                       'marginBottom':'15px'}
            ),
            html.Div(status_cards),
            html.Div(index_cards),
            alert_panel,
            html.Br(),
            dcc.Graph(figure=fig_pred)
        ])

    # ==========================================
    # TAB 2: SOIL MAPS
    # ==========================================
    elif tab == 'tab-soil':
        return html.Div([
            html.H3('Predicted Soil Property Maps',
                    style={'color':'#2c3e50',
                           'marginBottom':'5px'}),
            html.P(
                f"Updated: {st['last_run']} | "
                f"S2: {st['sentinel2_date']} | "
                f"{len(grid):,} prediction points",
                style={'color':'#7f8c8d',
                       'marginBottom':'15px',
                       'fontSize':'12px'}
            ),
            html.Div([
                html.Label(
                    'Select Soil Property:',
                    style={'fontWeight':'bold',
                           'marginRight':'10px'}
                ),
                dcc.Dropdown(
                    id='soil-dropdown',
                    options=[
                        {'label':
                         f"{p.replace('_',' ').title()}"
                         f" ({soil_properties[p]['unit']})",
                         'value': p}
                        for p in soil_properties
                    ],
                    value='pH',
                    clearable=False,
                    style={'width':'350px',
                           'display':'inline-block',
                           'verticalAlign':'middle'}
                )
            ], style={'marginBottom':'20px'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='soil-map',
                              style={'height':'520px'})
                ], style={'width':'70%',
                          'display':'inline-block',
                          'verticalAlign':'top'}),
                html.Div([
                    html.H4('Block Statistics'),
                    html.Div(id='soil-stats')
                ], style={
                    'width':'28%',
                    'display':'inline-block',
                    'verticalAlign':'top',
                    'marginLeft':'2%',
                    'backgroundColor':'white',
                    'padding':'15px',
                    'borderRadius':'8px',
                    'boxShadow':
                        '0 2px 8px rgba(0,0,0,0.1)'
                })
            ])
        ])

    # ==========================================
    # TAB 3: CROP MONITORING
    # ==========================================
    elif tab == 'tab-crop':

        fig_ts = go.Figure()

        for zone in zones:
            if zone in zone_colors:
                fig_ts.add_trace(go.Scatter(
                    x    = ndvi_df['month'],
                    y    = ndvi_df[zone],
                    name = zone,
                    mode = 'lines+markers',
                    line = dict(
                        color=zone_colors[zone],
                        width=2.5
                    ),
                    marker=dict(size=9)
                ))

        fig_ts.add_vrect(
            x0='Jan-2025', x1='Mar-2025',
            fillcolor='gold', opacity=0.12,
            annotation_text='Rabi 2024-25'
        )
        fig_ts.add_vrect(
            x0='Jun-2025', x1='Oct-2025',
            fillcolor='lightgreen', opacity=0.12,
            annotation_text='Kharif 2025'
        )
        fig_ts.add_vrect(
            x0='Jan-2026', x1='Mar-2026',
            fillcolor='gold', opacity=0.12,
            annotation_text='Rabi 2025-26'
        )
        fig_ts.add_hline(
            y=0.40,
            line_dash='dash',
            line_color='red',
            annotation_text='Stress Threshold'
        )
        fig_ts.update_layout(
            title='Real NDVI Time Series — '
                  'Khamanon Block',
            xaxis_title='Month',
            yaxis_title='NDVI',
            yaxis=dict(range=[0,1.0]),
            plot_bgcolor='white',
            height=450,
            legend=dict(
                orientation='h',
                yanchor='bottom', y=-0.35,
                xanchor='left', x=0
            )
        )

        # Current NDVI status
        current_ndvi = st.get('ndvi_mean', 0)
        ndvi_status  = (
            '🔴 STRESS' if current_ndvi < 0.35
            else '🟡 MODERATE' if current_ndvi < 0.50
            else '🟢 HEALTHY'
        )

        return html.Div([
            html.H3('Crop Growth Monitoring',
                    style={'color':'#2c3e50',
                           'marginBottom':'5px'}),
            html.Div([
                html.Span('Current NDVI Status: ',
                          style={'fontWeight':'bold'}),
                html.Span(
                    f"{ndvi_status} "
                    f"(mean={current_ndvi:.3f})",
                    style={'fontSize':'14px'}
                )
            ], style={
                'backgroundColor':'white',
                'padding':'10px 15px',
                'borderRadius':'6px',
                'marginBottom':'15px',
                'boxShadow':
                    '0 2px 6px rgba(0,0,0,0.08)'
            }),
            dcc.Graph(figure=fig_ts)
        ])

    # ==========================================
    # TAB 4: SAMPLE POINTS
    # ==========================================
    elif tab == 'tab-samples':

        fig = go.Figure()
        fig.add_trace(go.Scattermap(
            lat  = soil['latitude'],
            lon  = soil['longitude'],
            mode = 'markers',
            marker=dict(
                size=8,
                color=soil['pH'],
                colorscale='RdYlGn_r',
                colorbar=dict(title='pH',
                              thickness=15),
                showscale=True
            ),
            text=soil.apply(
                lambda r:
                f"ID: {r['sample_id']}<br>"
                f"pH: {r['pH']:.2f}<br>"
                f"OC: {r['OC']:.3f}%<br>"
                f"EC: {r['EC']:.3f} dS/m<br>"
                f"N: {r['available_N']:.1f} kg/ha",
                axis=1
            ),
            hoverinfo='text',
            name='cLHS Points'
        ))
        fig.update_layout(
            map=dict(
                style ='open-street-map',
                center=dict(lat=30.795,lon=76.352),
                zoom  =11
            ),
            title='208 cLHS Sample Points '
                  '(coloured by pH)',
            height=580,
            margin=dict(l=0,r=0,t=50,b=0)
        )

        return html.Div([
            html.H3('cLHS Sample Points Map'),
            html.P(
                '208 real field samples. '
                'Hover any point for soil values.',
                style={'color':'#7f8c8d',
                       'marginBottom':'15px'}
            ),
            dcc.Graph(figure=fig)
        ])

    # ==========================================
    # TAB 5: SOIL SUMMARY
    # ==========================================
    elif tab == 'tab-summary':

        soil_cols = ['pH','OC','EC','K2O',
                     'available_P','available_N',
                     'CEC','bulk_density','CaCO3']

        colors = ['#3498db','#2ecc71','#e74c3c',
                  '#9b59b6','#f39c12','#1abc9c',
                  '#e67e22','#34495e','#e91e63']

        fig_box = go.Figure()
        for col, color in zip(soil_cols, colors):
            fig_box.add_trace(go.Box(
                y=soil[col],
                name=col.replace('_',' ').title(),
                marker_color=color,
                boxmean=True
            ))
        fig_box.update_layout(
            title='Soil Property Distribution — '
                  '208 Real cLHS Samples',
            height=400,
            showlegend=False,
            plot_bgcolor='white'
        )

        summary = soil[soil_cols].describe().round(3)
        fig_tbl = go.Figure(data=[go.Table(
            header=dict(
                values=['Stat'] + [
                    c.replace('_',' ').title()
                    for c in soil_cols
                ],
                fill_color='#2c3e50',
                font=dict(color='white',size=10),
                align='center'
            ),
            cells=dict(
                values=[
                    summary.index.tolist()
                ] + [
                    summary[c].tolist()
                    for c in soil_cols
                ],
                fill_color=[
                    ['#ecf0f1' if i%2==0
                     else 'white'
                     for i in range(len(summary))]
                ],
                align='center',
                font=dict(size=10)
            )
        )])
        fig_tbl.update_layout(
            title='Descriptive Statistics',
            height=350
        )

        return html.Div([
            html.H3('Soil Data Summary'),
            dcc.Graph(figure=fig_box),
            dcc.Graph(figure=fig_tbl)
        ])

    return html.Div('Tab not found')


# ============================================
# SOIL MAP CALLBACK
# ============================================

@callback(
    Output('soil-map',  'figure'),
    Output('soil-stats','children'),
    Input('soil-dropdown','value'),
    Input('refresh-interval','n_intervals')
)
def update_map(prop, n):
    _, grid, _, _, st = load_data()
    config = soil_properties[prop]
    vals   = grid[prop]

    fig = go.Figure(data=go.Densitymap(
        lat       = grid['northing'],
        lon       = grid['easting'],
        z         = vals,
        radius    = 15,
        colorscale= config['cmap'],
        showscale = True,
        colorbar  = dict(title=config['unit'],
                         thickness=15),
        hovertemplate=(
            f"{prop.replace('_',' ').title()}"
            f": %{{z:.3f}} {config['unit']}"
            "<extra></extra>"
        )
    ))
    fig.update_layout(
        map=dict(
            style ='open-street-map',
            center=dict(lat=30.795,lon=76.352),
            zoom  =11
        ),
        title=dict(
            text=(
                f"Predicted "
                f"{prop.replace('_',' ').title()} "
                f"({config['unit']}) — "
                f"Updated {st['sentinel2_date']}"
            ),
            font=dict(size=12)
        ),
        height=500,
        margin=dict(l=0,r=0,t=50,b=0)
    )

    stats = [
        ('Mean',   f"{vals.mean():.3f} {config['unit']}"),
        ('Min',    f"{vals.min():.3f} {config['unit']}"),
        ('Max',    f"{vals.max():.3f} {config['unit']}"),
        ('Std',    f"{vals.std():.3f} {config['unit']}"),
        ('Points', f"{len(vals):,}"),
        ('S2 Date',st['sentinel2_date']),
    ]

    divs = [
        html.Div([
            html.Span(f"{l}: ",
                      style={'fontWeight':'bold',
                             'fontSize':'12px',
                             'color':'#7f8c8d'}),
            html.Span(v,
                      style={'fontSize':'12px',
                             'color':'#2c3e50'})
        ], style={'marginBottom':'8px',
                  'borderBottom':'1px solid #ecf0f1',
                  'paddingBottom':'6px'})
        for l, v in stats
    ]

    return fig, divs


# ============================================
# RUN
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  REAL-TIME DIGITAL TWIN DASHBOARD")
    print("=" * 55)
    print("\nOpen browser and go to:")
    print("  http://127.0.0.1:8050")
    print("\nDashboard auto-refreshes every 30 sec.")
    print("Run realtime_updater.py to fetch")
    print("new Sentinel-2 data anytime.")
    print("\nPress Ctrl+C to stop.")
    print("=" * 55 + "\n")
    app.run(debug=False, port=8050)