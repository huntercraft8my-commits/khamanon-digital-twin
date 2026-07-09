# ============================================
# DSS - KHAMANON BLOCK
# Script 5: Interactive Dashboard
# ============================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import os

# ============================================
# LOAD ALL DATA
# ============================================

base = os.path.dirname(os.path.abspath(__file__))

soil_samples  = pd.read_csv(
    os.path.join(base, '..', 'data', 'soil_samples_fake.csv')
)
prediction_grid = pd.read_csv(
    os.path.join(base, '..', 'data', 'prediction_grid.csv')
)
ndvi_df = pd.read_csv(
    os.path.join(base, '..', 'data', 'ndvi_timeseries.csv')
)

# ============================================
# SOIL PROPERTIES CONFIG
# ============================================

soil_properties = {
    'pH'            : {'unit': 'pH units', 'cmap': 'RdYlGn_r'},
    'organic_carbon': {'unit': '%',        'cmap': 'YlGn'    },
    'EC'            : {'unit': 'dS/m',     'cmap': 'OrRd'    },
    'available_N'   : {'unit': 'kg/ha',    'cmap': 'Blues'   },
    'available_P'   : {'unit': 'kg/ha',    'cmap': 'Purples' },
    'available_K'   : {'unit': 'kg/ha',    'cmap': 'BuGn'    },
    'CEC'           : {'unit': 'meq/100g', 'cmap': 'PuBu'    }
}

lulc_colors = {
    'Cropland'  : '#2ecc71',
    'Built-up'  : '#e74c3c',
    'Fallow'    : '#f39c12',
    'Vegetation': '#27ae60',
    'Water body': '#3498db'
}

# ============================================
# INITIALIZE DASH APP
# ============================================

app = Dash(__name__)

app.layout = html.Div([

    # ---- HEADER ----
    html.Div([
        html.H1(
            'DSS — Khamanon Block',
            style={
                'color'     : 'white',
                'margin'    : '0',
                'fontSize'  : '26px',
                'fontWeight': 'bold'
            }
        ),
        html.P(
            'Fatehgarh Sahib, Punjab | '
            'Soil Properties + Crop Monitoring | '
            'Fake Data Demo',
            style={
                'color'    : '#bdc3c7',
                'margin'   : '5px 0 0 0',
                'fontSize' : '13px'
            }
        )
    ], style={
        'backgroundColor': '#2c3e50',
        'padding'        : '20px 30px',
        'marginBottom'   : '0px'
    }),

    # ---- TAB BAR ----
    dcc.Tabs(
        id='main-tabs',
        value='tab-soil',
        children=[
            dcc.Tab(
                label='Soil Property Maps',
                value='tab-soil',
                style={'fontWeight': 'bold'},
                selected_style={
                    'fontWeight'     : 'bold',
                    'borderTop'      : '3px solid #27ae60',
                    'backgroundColor': '#ecf0f1'
                }
            ),
            dcc.Tab(
                label='Crop Monitoring',
                value='tab-crop',
                style={'fontWeight': 'bold'},
                selected_style={
                    'fontWeight'     : 'bold',
                    'borderTop'      : '3px solid #27ae60',
                    'backgroundColor': '#ecf0f1'
                }
            ),
            dcc.Tab(
                label='Sample Points Map',
                value='tab-samples',
                style={'fontWeight': 'bold'},
                selected_style={
                    'fontWeight'     : 'bold',
                    'borderTop'      : '3px solid #27ae60',
                    'backgroundColor': '#ecf0f1'
                }
            ),
            dcc.Tab(
                label='Soil Summary',
                value='tab-summary',
                style={'fontWeight': 'bold'},
                selected_style={
                    'fontWeight'     : 'bold',
                    'borderTop'      : '3px solid #27ae60',
                    'backgroundColor': '#ecf0f1'
                }
            ),
        ],
        style={'marginBottom': '0px'}
    ),

    # ---- TAB CONTENT ----
    html.Div(id='tab-content',
             style={'padding': '20px 30px',
                    'backgroundColor': '#f8f9fa',
                    'minHeight': '80vh'})

], style={'fontFamily': 'Segoe UI, Arial, sans-serif'})


# ============================================
# CALLBACKS — update content when tab changes
# ============================================

@callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab(tab):

    # ==========================================
    # TAB 1: SOIL PROPERTY MAPS
    # ==========================================
    if tab == 'tab-soil':
        return html.Div([

            html.H3('Predicted Soil Property Maps',
                    style={'color': '#2c3e50',
                           'marginBottom': '5px'}),
            html.P(
                'Select a soil property to view its '
                'predicted spatial distribution across '
                'Khamanon Block.',
                style={'color': '#7f8c8d',
                       'marginBottom': '15px'}
            ),

            # Dropdown to select soil property
            html.Div([
                html.Label('Select Soil Property:',
                           style={'fontWeight': 'bold',
                                  'marginRight': '10px'}),
                dcc.Dropdown(
                    id='soil-property-dropdown',
                    options=[
                        {'label': f"{p.replace('_',' ').title()} "
                                  f"({soil_properties[p]['unit']})",
                         'value': p}
                        for p in soil_properties
                    ],
                    value='pH',
                    clearable=False,
                    style={'width': '350px',
                           'display': 'inline-block',
                           'verticalAlign': 'middle'}
                )
            ], style={'marginBottom': '20px'}),

            # Map + stats side by side
            html.Div([
                # Map
                html.Div([
                    dcc.Graph(id='soil-map',
                              style={'height': '500px'})
                ], style={'width': '70%',
                          'display': 'inline-block',
                          'verticalAlign': 'top'}),

                # Stats panel
                html.Div([
                    html.H4('Block Statistics',
                            style={'color': '#2c3e50',
                                   'marginTop': '10px'}),
                    html.Div(id='soil-stats-panel')
                ], style={
                    'width'          : '28%',
                    'display'        : 'inline-block',
                    'verticalAlign'  : 'top',
                    'marginLeft'     : '2%',
                    'backgroundColor': 'white',
                    'padding'        : '15px',
                    'borderRadius'   : '8px',
                    'boxShadow'      : '0 2px 8px rgba(0,0,0,0.1)'
                })
            ])
        ])

    # ==========================================
    # TAB 2: CROP MONITORING
    # ==========================================
    elif tab == 'tab-crop':

        # NDVI time series chart
        fig_ts = go.Figure()

        zone_config = [
            ('ndvi_healthy',
             'Healthy Cropland (North)',    'green',  'circle'),
            ('ndvi_stressed',
             'Stressed Cropland (Central)', 'orange', 'square'),
            ('ndvi_urban',
             'Peri-urban / Built-up (SE)',  'red',    'triangle-up'),
            ('ndvi_vegetation',
             'Vegetation / Scrubland',      'blue',   'diamond'),
        ]

        for col, label, color, symbol in zone_config:
            fig_ts.add_trace(go.Scatter(
                x    = ndvi_df['month'],
                y    = ndvi_df[col],
                name = label,
                mode = 'lines+markers',
                line = dict(color=color, width=2.5),
                marker = dict(symbol=symbol, size=8)
            ))

        # Season bands
        fig_ts.add_vrect(
            x0='Jan', x1='Apr',
            fillcolor='gold', opacity=0.12,
            annotation_text='Wheat Season',
            annotation_position='top left'
        )
        fig_ts.add_vrect(
            x0='Jun', x1='Nov',
            fillcolor='lightgreen', opacity=0.12,
            annotation_text='Rice Season',
            annotation_position='top left'
        )
        fig_ts.add_hline(
            y=0.40,
            line_dash='dash',
            line_color='red',
            annotation_text='Stress Threshold',
            annotation_position='right'
        )

        fig_ts.update_layout(
            title=dict(
                text='NDVI Time Series — Khamanon Block (2025–2026)',
                font=dict(size=15)
            ),
            xaxis_title='Month',
            yaxis_title='NDVI',
            yaxis=dict(range=[0, 0.90]),
            legend=dict(
                orientation='h',
                yanchor='bottom', y=-0.35,
                xanchor='left',   x=0
            ),
            plot_bgcolor ='white',
            paper_bgcolor='white',
            height=420
        )

        # Crop calendar table
        crop_calendar = pd.DataFrame({
            'Month'      : ndvi_df['month'],
            'Crop Stage' : [
                'Wheat growing', 'Wheat peak', 'Wheat peak',
                'Wheat maturity', 'Post-harvest bare',
                'Tillage / bare', 'Rice transplanting',
                'Rice growing', 'Rice peak',
                'Rice maturity', 'Rice harvest / wheat sow',
                'Wheat establishing'
            ],
            'Healthy NDVI'  : ndvi_df['ndvi_healthy'],
            'Stressed NDVI' : ndvi_df['ndvi_stressed'],
            'Stress Alert'  : [
                '⚠️' if s < 0.40 else '✅'
                for s in ndvi_df['ndvi_stressed']
            ]
        })

        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=list(crop_calendar.columns),
                fill_color='#2c3e50',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[
                    crop_calendar[c]
                    for c in crop_calendar.columns
                ],
                fill_color=[
                    ['#fff9e6'
                     if i % 2 == 0
                     else 'white'
                     for i in range(12)]
                ],
                align='left',
                font=dict(size=11)
            )
        )])
        fig_table.update_layout(
            title='Crop Calendar — Khamanon Block',
            height=420
        )

        return html.Div([
            html.H3('Crop Growth Monitoring',
                    style={'color': '#2c3e50',
                           'marginBottom': '5px'}),
            html.P(
                'NDVI time series tracking crop growth, '
                'stress detection, and seasonal patterns '
                'across Khamanon Block zones.',
                style={'color'        : '#7f8c8d',
                       'marginBottom' : '15px'}
            ),
            dcc.Graph(figure=fig_ts),
            html.Br(),
            dcc.Graph(figure=fig_table)
        ])

    # ==========================================
    # TAB 3: SAMPLE POINTS MAP
    # ==========================================
    elif tab == 'tab-samples':

        soil_samples['color'] = soil_samples['LULC'].map(
            lulc_colors
        )

        fig_map = go.Figure()

        for lulc_class, color in lulc_colors.items():
            subset = soil_samples[
                soil_samples['LULC'] == lulc_class
            ]
            if len(subset) == 0:
                continue
            fig_map.add_trace(go.Scattermap(
                lat  = subset['latitude'],
                lon  = subset['longitude'],
                mode = 'markers',
                marker=dict(size=12, color=color),
                name = lulc_class,
                text = subset.apply(
                    lambda r:
                    f"ID: {r['sample_id']}<br>"
                    f"LULC: {r['LULC']}<br>"
                    f"pH: {r['pH']}<br>"
                    f"OC: {r['organic_carbon']}%<br>"
                    f"EC: {r['EC']} dS/m<br>"
                    f"N: {r['available_N']} kg/ha",
                    axis=1
                ),
                hoverinfo='text'
            ))

        fig_map.update_layout(
            map=dict(
                style  ='open-street-map',
                center = dict(lat=30.45, lon=76.345),
                zoom   = 11
            ),
            title=dict(
                text='cLHS Sample Points — Khamanon Block',
                font=dict(size=15)
            ),
            height=580,
            legend=dict(
                orientation='v',
                x=0.01, y=0.99,
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='gray',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )

        return html.Div([
            html.H3(
                'cLHS Sample Points Map',
                style={'color': '#2c3e50',
                       'marginBottom': '5px'}
            ),
            html.P(
                'All 50 fake cLHS sample points plotted '
                'on Khamanon Block. Hover over any point '
                'to see its soil properties.',
                style={'color'      : '#7f8c8d',
                       'marginBottom': '15px'}
            ),
            dcc.Graph(figure=fig_map),
            html.P(
                '💡 Hover over any point to see pH, '
                'OC, EC, and N values at that location.',
                style={
                    'color'        : '#27ae60',
                    'fontWeight'   : 'bold',
                    'marginTop'    : '10px'
                }
            )
        ])

    # ==========================================
    # TAB 4: SOIL SUMMARY
    # ==========================================
    elif tab == 'tab-summary':

        # Box plots for all soil properties
        fig_box = go.Figure()

        props   = ['pH', 'organic_carbon', 'EC',
                   'available_N', 'available_P',
                   'available_K']
        colors2 = ['#3498db', '#2ecc71', '#e74c3c',
                   '#9b59b6', '#f39c12', '#1abc9c']

        for prop, color in zip(props, colors2):
            fig_box.add_trace(go.Box(
                y    = soil_samples[prop],
                name = prop.replace('_', ' ').title(),
                marker_color = color,
                boxmean      = True
            ))

        fig_box.update_layout(
            title ='Soil Property Distribution — All 50 Sample Points',
            yaxis_title = 'Value',
            height      = 420,
            showlegend  = False,
            plot_bgcolor = 'white'
        )

        # LULC pie chart
        lulc_counts = soil_samples['LULC'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels = lulc_counts.index,
            values = lulc_counts.values,
            marker = dict(
                colors=[lulc_colors[l]
                        for l in lulc_counts.index]
            ),
            hole   = 0.4
        )])
        fig_pie.update_layout(
            title ='LULC Distribution — Khamanon Block',
            height= 380
        )

        # Summary stats table
        summary = soil_samples[
            ['pH', 'organic_carbon', 'EC',
             'available_N', 'available_P', 'available_K']
        ].describe().round(3)

        fig_stats = go.Figure(data=[go.Table(
            header=dict(
                values=['Statistic'] + [
                    c.replace('_', ' ').title()
                    for c in summary.columns
                ],
                fill_color ='#2c3e50',
                font       = dict(color='white', size=11),
                align      = 'left'
            ),
            cells=dict(
                values=[summary.index.tolist()] + [
                    summary[c].tolist()
                    for c in summary.columns
                ],
                fill_color=[
                    ['#ecf0f1'
                     if i % 2 == 0
                     else 'white'
                     for i in range(len(summary))]
                ],
                align='left',
                font=dict(size=11)
            )
        )])
        fig_stats.update_layout(
            title ='Descriptive Statistics — Soil Properties',
            height= 350
        )

        return html.Div([
            html.H3(
                'Soil Data Summary',
                style={'color'       : '#2c3e50',
                       'marginBottom': '5px'}
            ),
            html.P(
                'Statistical summary and distribution '
                'of all soil properties across '
                'Khamanon Block sample points.',
                style={'color'        : '#7f8c8d',
                       'marginBottom' : '15px'}
            ),
            html.Div([
                html.Div(
                    [dcc.Graph(figure=fig_box)],
                    style={'width'  : '65%',
                           'display': 'inline-block',
                           'verticalAlign': 'top'}
                ),
                html.Div(
                    [dcc.Graph(figure=fig_pie)],
                    style={'width'        : '33%',
                           'display'      : 'inline-block',
                           'verticalAlign': 'top',
                           'marginLeft'   : '2%'}
                )
            ]),
            dcc.Graph(figure=fig_stats)
        ])

    return html.Div('Tab not found')


# ============================================
# CALLBACK: Update soil map when
# dropdown selection changes
# ============================================

@callback(
    Output('soil-map',        'figure'),
    Output('soil-stats-panel','children'),
    Input('soil-property-dropdown', 'value')
)
def update_soil_map(selected_property):

    config = soil_properties[selected_property]
    vals   = prediction_grid[selected_property]

    fig = go.Figure(data=go.Densitymap(
        lat       = prediction_grid['latitude'],
        lon       = prediction_grid['longitude'],
        z         = vals,
        radius    = 18,
        colorscale= config['cmap'],
        showscale = True,
        colorbar  = dict(
            title=config['unit'],
            thickness=15
        ),
        hovertemplate=(
            f"{selected_property.replace('_',' ').title()}"
            f": %{{z:.3f}} {config['unit']}<br>"
            "Lat: %{lat:.4f}<br>"
            "Lon: %{lon:.4f}<extra></extra>"
        )
    ))

    fig.update_layout(
        map=dict(
            style ='open-street-map',
            center=dict(lat=30.45, lon=76.345),
            zoom  =11
        ),
        title=dict(
            text=(
                f"Predicted {selected_property.replace('_',' ').title()}"
                f" ({config['unit']}) — Khamanon Block"
            ),
            font=dict(size=13)
        ),
        height=480,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Stats panel
    stats_items = [
        ('Mean',    f"{vals.mean():.3f} {config['unit']}"),
        ('Min',     f"{vals.min():.3f} {config['unit']}"),
        ('Max',     f"{vals.max():.3f} {config['unit']}"),
        ('Std Dev', f"{vals.std():.3f} {config['unit']}"),
        ('Samples', '2,500 grid points'),
    ]

    stats_divs = []
    for label, value in stats_items:
        stats_divs.append(html.Div([
            html.Span(
                label + ': ',
                style={'color'     : '#7f8c8d',
                       'fontSize'  : '13px',
                       'fontWeight': 'bold'}
            ),
            html.Span(
                value,
                style={'color'   : '#2c3e50',
                       'fontSize': '13px'}
            )
        ], style={'marginBottom': '12px',
                  'borderBottom': '1px solid #ecf0f1',
                  'paddingBottom': '10px'}))

    # Punjab soil health context
    context = {
        'pH'            :
            '⚠️ Punjab soils average pH 7.5–8.5. '
            'Values above 8.0 indicate alkalinity stress.',
        'organic_carbon':
            '⚠️ Punjab OC critically low (<0.5%). '
            'Healthy soil needs >0.75%.',
        'EC'            :
            'EC below 1.0 dS/m is safe for most crops. '
            'Above 2.0 indicates salinity stress.',
        'available_N'   :
            'Optimal N: 200–280 kg/ha. '
            'Continuous rice-wheat depletes N rapidly.',
        'available_P'   :
            'Optimal P: 20–35 kg/ha for Punjab soils.',
        'available_K'   :
            'K generally adequate in alluvial soils '
            'of central Punjab.',
        'CEC'           :
            'CEC 10–20 meq/100g is typical for '
            'loamy Punjab soils.'
    }

    stats_divs.append(html.Div(
        context[selected_property],
        style={
            'backgroundColor': '#eafaf1',
            'padding'        : '10px',
            'borderRadius'   : '6px',
            'fontSize'       : '12px',
            'color'          : '#1e8449',
            'marginTop'      : '10px'
        }
    ))

    return fig, stats_divs


# ============================================
# RUN THE DASHBOARD
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  DSS DASHBOARD STARTING")
    print("=" * 50)
    print("\nOpen your browser and go to:")
    print("  http://127.0.0.1:8050")
    print("\nPress Ctrl+C in this window to stop.")
    print("=" * 50 + "\n")
    app.run(debug=False, port=8050)