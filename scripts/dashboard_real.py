# ============================================
# DSS - KHAMANON BLOCK
# Script 8: Real Data Dashboard
# ============================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import os

print("=" * 55)
print("  PHASE 5 — REAL DATA DASHBOARD")
print("  Khamanon Block DSS")
print("=" * 55)

# ============================================
# LOAD ALL REAL DATA
# ============================================

base = os.path.dirname(os.path.abspath(__file__))

soil    = pd.read_csv(
    os.path.join(base, '..', 'data',
                 'soil_data_validated.csv')
)
grid    = pd.read_csv(
    os.path.join(base, '..', 'data',
                 'real_prediction_grid.csv')
)
ndvi_df = pd.read_csv(
    os.path.join(base, '..', 'data',
                 'ndvi_processed.csv')
)
validation = pd.read_csv(
    os.path.join(base, '..', 'data',
                 'model_validation_real.csv')
)

print(f"\nSoil samples    : {len(soil)}")
print(f"Prediction grid : {len(grid)}")
print(f"NDVI months     : {len(ndvi_df)}")

# ============================================
# CONFIG
# ============================================

soil_properties = {
    'pH'          : {'unit': 'pH units', 'cmap': 'RdYlGn_r'},
    'OC'          : {'unit': '%',        'cmap': 'YlGn'    },
    'EC'          : {'unit': 'dS/m',     'cmap': 'OrRd'    },
    'available_N' : {'unit': 'kg/ha',    'cmap': 'Blues'   },
    'available_P' : {'unit': 'kg/ha',    'cmap': 'Purples' },
    'K2O'         : {'unit': 'kg/ha',    'cmap': 'BuGn'    },
    'CEC'         : {'unit': 'meq/100g', 'cmap': 'PuBu'    },
    'bulk_density': {'unit': 'g/L',      'cmap': 'YlOrBr'  },
    'CaCO3'       : {'unit': '%',        'cmap': 'RdPu'    }
}

zone_colors = {
    'Healthy Cropland (North)'    : 'green',
    'Stressed Cropland (Central)' : 'orange',
    'Peri-urban SE'               : 'red',
    'Vegetation West'             : 'blue'
}

zone_markers = {
    'Healthy Cropland (North)'    : 'circle',
    'Stressed Cropland (Central)' : 'square',
    'Peri-urban SE'               : 'triangle-up',
    'Vegetation West'             : 'diamond'
}

# ============================================
# APP LAYOUT
# ============================================

app = Dash(__name__)

app.layout = html.Div([

    # HEADER
    html.Div([
        html.H1(
            'DSS — Khamanon Block',
            style={'color':'white','margin':'0',
                   'fontSize':'26px',
                   'fontWeight':'bold'}
        ),
        html.P(
            'Fatehgarh Sahib, Punjab  |  '
            'Soil + Crop Monitoring  |  '
            'Real Sentinel-2 + Field Data  |  '
            '208 cLHS Sample Points',
            style={'color':'#bdc3c7',
                   'margin':'5px 0 0 0',
                   'fontSize':'12px'}
        )
    ], style={
        'backgroundColor':'#2c3e50',
        'padding'        :'20px 30px'
    }),

    # TABS
    dcc.Tabs(
        id='main-tabs', value='tab-soil',
        children=[
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
            dcc.Tab(label='Model Accuracy',
                    value='tab-accuracy',
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
# TAB CONTENT CALLBACK
# ============================================

@callback(
    Output('tab-content','children'),
    Input('main-tabs','value')
)
def render_tab(tab):

    # ==========================================
    # TAB 1: SOIL MAPS
    # ==========================================
    if tab == 'tab-soil':
        return html.Div([
            html.H3('Predicted Soil Property Maps',
                    style={'color':'#2c3e50',
                           'marginBottom':'5px'}),
            html.P(
                'Spatial prediction across Khamanon Block '
                'using Random Forest + Sentinel-2 + '
                'terrain covariates. '
                f'Grid: {len(grid):,} prediction points.',
                style={'color':'#7f8c8d',
                       'marginBottom':'15px'}
            ),
            html.Div([
                html.Label('Select Soil Property:',
                           style={'fontWeight':'bold',
                                  'marginRight':'10px'}),
                dcc.Dropdown(
                    id='soil-dropdown',
                    options=[
                        {'label':
                         f"{p.replace('_',' ').title()} "
                         f"({soil_properties[p]['unit']})",
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
                    html.H4('Block Statistics',
                            style={'color':'#2c3e50',
                                   'marginTop':'10px'}),
                    html.Div(id='soil-stats')
                ], style={
                    'width':'28%',
                    'display':'inline-block',
                    'verticalAlign':'top',
                    'marginLeft':'2%',
                    'backgroundColor':'white',
                    'padding':'15px',
                    'borderRadius':'8px',
                    'boxShadow':'0 2px 8px rgba(0,0,0,0.1)'
                })
            ])
        ])

    # ==========================================
    # TAB 2: CROP MONITORING
    # ==========================================
    elif tab == 'tab-crop':

        zones = [c for c in ndvi_df.columns
                 if c != 'month']

        fig_ts = go.Figure()

        for zone in zones:
            if zone in zone_colors:
                fig_ts.add_trace(go.Scatter(
                    x    = ndvi_df['month'],
                    y    = ndvi_df[zone],
                    name = zone,
                    mode = 'lines+markers',
                    line = dict(
                        color=zone_colors.get(zone,'gray'),
                        width=2.5
                    ),
                    marker=dict(
                        symbol=zone_markers.get(
                            zone,'circle'),
                        size=9
                    )
                ))

        # Season bands
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
            line_dash='dash', line_color='red',
            annotation_text='Stress Threshold',
            annotation_position='right'
        )

        fig_ts.update_layout(
            title=dict(
                text='Real NDVI Time Series — '
                     'Khamanon Block (2025–2026)',
                font=dict(size=14)
            ),
            xaxis_title='Month',
            yaxis_title='NDVI',
            yaxis=dict(range=[0, 1.0]),
            legend=dict(
                orientation='h',
                yanchor='bottom', y=-0.35,
                xanchor='left', x=0
            ),
            plot_bgcolor ='white',
            paper_bgcolor='white',
            height=450
        )

        # Stress summary table
        stress_data = []
        for _, row in ndvi_df.iterrows():
            r = [row['month']]
            for zone in zones:
                if zone in row:
                    v = row[zone]
                    r.append(f"{v:.3f}")
            stress_data.append(r)

        fig_tbl = go.Figure(data=[go.Table(
            header=dict(
                values=['Month'] + [
                    z.split('(')[0].strip()
                    for z in zones
                ],
                fill_color='#2c3e50',
                font=dict(color='white', size=11),
                align='center'
            ),
            cells=dict(
                values=[
                    [r[0] for r in stress_data]
                ] + [
                    [r[i+1] for r in stress_data]
                    for i in range(len(zones))
                ],
                fill_color=[
                    ['#f8f9fa'
                     if i % 2 == 0 else 'white'
                     for i in range(len(stress_data))]
                ],
                align='center',
                font=dict(size=11)
            )
        )])
        fig_tbl.update_layout(
            title='Monthly NDVI Values by Zone',
            height=500
        )

        return html.Div([
            html.H3('Crop Growth Monitoring',
                    style={'color':'#2c3e50',
                           'marginBottom':'5px'}),
            html.P(
                'Real Sentinel-2 NDVI extracted for '
                '4 zones across Khamanon Block. '
                '15 months of data (Jan 2025 – Mar 2026).',
                style={'color':'#7f8c8d',
                       'marginBottom':'15px'}
            ),
            dcc.Graph(figure=fig_ts),
            html.Br(),
            dcc.Graph(figure=fig_tbl)
        ])

    # ==========================================
    # TAB 3: SAMPLE POINTS MAP
    # ==========================================
    elif tab == 'tab-samples':

        fig = go.Figure()

        fig.add_trace(go.Scattermap(
            lat  = soil['latitude'],
            lon  = soil['longitude'],
            mode = 'markers',
            marker=dict(
                size  = 8,
                color = soil['pH'],
                colorscale = 'RdYlGn_r',
                colorbar   = dict(
                    title='pH',
                    thickness=15
                ),
                showscale=True
            ),
            text = soil.apply(
                lambda r:
                f"ID: {r['sample_id']}<br>"
                f"pH: {r['pH']:.2f}<br>"
                f"OC: {r['OC']:.3f}%<br>"
                f"EC: {r['EC']:.3f} dS/m<br>"
                f"N:  {r['available_N']:.1f} kg/ha<br>"
                f"K:  {r['K2O']:.1f} kg/ha",
                axis=1
            ),
            hoverinfo='text',
            name='cLHS Points'
        ))

        fig.update_layout(
            map=dict(
                style ='open-street-map',
                center=dict(lat=30.795,
                            lon=76.352),
                zoom  =11
            ),
            title=dict(
                text='208 cLHS Sample Points — '
                     'Khamanon Block (coloured by pH)',
                font=dict(size=14)
            ),
            height=600,
            margin=dict(l=0,r=0,t=50,b=0)
        )

        return html.Div([
            html.H3('cLHS Sample Points Map',
                    style={'color':'#2c3e50',
                           'marginBottom':'5px'}),
            html.P(
                '208 field sample points plotted on '
                'Khamanon Block. Colour = pH value. '
                'Hover over any point for soil details.',
                style={'color':'#7f8c8d',
                       'marginBottom':'15px'}
            ),
            dcc.Graph(figure=fig),
            html.P(
                '💡 Hover any point → see pH, OC, EC, '
                'N and K values at that location.',
                style={'color':'#27ae60',
                       'fontWeight':'bold',
                       'marginTop':'10px'}
            )
        ])

    # ==========================================
    # TAB 4: MODEL ACCURACY
    # ==========================================
    elif tab == 'tab-accuracy':

        props  = validation['soil_property'].tolist()
        r2vals = validation['R2'].tolist()
        cvvals = validation['CV_R2'].tolist()
        rmse   = validation['RMSE'].tolist()

        fig_acc = go.Figure()

        fig_acc.add_trace(go.Bar(
            x    = props,
            y    = r2vals,
            name = 'Test R²',
            marker_color='#3498db',
            text = [f'{v:.3f}' for v in r2vals],
            textposition='outside'
        ))
        fig_acc.add_trace(go.Bar(
            x    = props,
            y    = cvvals,
            name = 'CV R² (5-fold)',
            marker_color='#2ecc71',
            text = [f'{v:.3f}' for v in cvvals],
            textposition='outside'
        ))
        fig_acc.add_hline(
            y=0.5, line_dash='dash',
            line_color='orange',
            annotation_text='Moderate (0.5)'
        )
        fig_acc.add_hline(
            y=0.3, line_dash='dash',
            line_color='red',
            annotation_text='Weak (0.3)'
        )

        fig_acc.update_layout(
            barmode='group',
            title='RF Model Accuracy — All Soil Properties',
            yaxis_title='R² Score',
            yaxis=dict(range=[-0.6, 1.0]),
            height=420,
            plot_bgcolor='white'
        )

        fig_rmse = go.Figure(data=[go.Table(
            header=dict(
                values=['Soil Property',
                        'Test R²', 'CV R²',
                        'RMSE', 'MAE'],
                fill_color='#2c3e50',
                font=dict(color='white', size=11),
                align='center'
            ),
            cells=dict(
                values=[
                    validation['soil_property'],
                    validation['R2'].round(3),
                    validation['CV_R2'].round(3),
                    validation['RMSE'].round(4),
                    validation['MAE'].round(4)
                ],
                fill_color=[
                    ['#ecf0f1' if i%2==0
                     else 'white'
                     for i in range(len(validation))]
                ],
                align='center',
                font=dict(size=11)
            )
        )])
        fig_rmse.update_layout(
            title='Detailed Validation Metrics',
            height=380
        )

        return html.Div([
            html.H3('RF Model Accuracy',
                    style={'color':'#2c3e50',
                           'marginBottom':'5px'}),
            html.P(
                'Random Forest model performance for '
                '9 soil properties. '
                'Practice data — real data will '
                'show improved R².',
                style={'color':'#7f8c8d',
                       'marginBottom':'15px'}
            ),
            dcc.Graph(figure=fig_acc),
            dcc.Graph(figure=fig_rmse)
        ])

    # ==========================================
    # TAB 5: SOIL SUMMARY
    # ==========================================
    elif tab == 'tab-summary':

        soil_cols = ['pH','OC','EC','K2O',
                     'available_P','available_N',
                     'CEC','bulk_density','CaCO3']

        fig_box = go.Figure()
        colors  = ['#3498db','#2ecc71','#e74c3c',
                   '#9b59b6','#f39c12','#1abc9c',
                   '#e67e22','#34495e','#e91e63']

        for col, color in zip(soil_cols, colors):
            fig_box.add_trace(go.Box(
                y    = soil[col],
                name = col.replace('_',' ').title(),
                marker_color = color,
                boxmean      = True
            ))

        fig_box.update_layout(
            title='Soil Property Distribution — '
                  '208 Real cLHS Samples',
            yaxis_title='Value',
            height=420,
            showlegend=False,
            plot_bgcolor='white'
        )

        summary = soil[soil_cols].describe().round(3)

        fig_tbl = go.Figure(data=[go.Table(
            header=dict(
                values=['Statistic'] + [
                    c.replace('_',' ').title()
                    for c in soil_cols
                ],
                fill_color='#2c3e50',
                font=dict(color='white', size=10),
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
            title='Descriptive Statistics — '
                  'Real Soil Data',
            height=350
        )

        return html.Div([
            html.H3('Soil Data Summary',
                    style={'color':'#2c3e50',
                           'marginBottom':'5px'}),
            html.P(
                'Statistical distribution of all '
                '9 soil properties from 208 real '
                'cLHS sample points, Khamanon Block.',
                style={'color':'#7f8c8d',
                       'marginBottom':'15px'}
            ),
            dcc.Graph(figure=fig_box),
            dcc.Graph(figure=fig_tbl)
        ])

    return html.Div('Tab not found')


# ============================================
# SOIL MAP + STATS CALLBACK
# ============================================

@callback(
    Output('soil-map',  'figure'),
    Output('soil-stats','children'),
    Input('soil-dropdown','value')
)
def update_soil_map(prop):

    config = soil_properties[prop]
    vals   = grid[prop]

    fig = go.Figure(data=go.Densitymap(
        lat       = grid['northing'],
        lon       = grid['easting'],
        z         = vals,
        radius    = 15,
        colorscale= config['cmap'],
        showscale = True,
        colorbar  = dict(
            title    = config['unit'],
            thickness= 15
        ),
        hovertemplate=(
            f"{prop.replace('_',' ').title()}"
            f": %{{z:.3f}} {config['unit']}"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        map=dict(
            style ='open-street-map',
            center=dict(lat=30.795, lon=76.352),
            zoom  =11
        ),
        title=dict(
            text=(
                f"Predicted "
                f"{prop.replace('_',' ').title()} "
                f"({config['unit']}) — "
                f"Khamanon Block"
            ),
            font=dict(size=13)
        ),
        height=500,
        margin=dict(l=0,r=0,t=50,b=0)
    )

    # Stats panel
    stats = [
        ('Mean',    f"{vals.mean():.3f} {config['unit']}"),
        ('Min',     f"{vals.min():.3f} {config['unit']}"),
        ('Max',     f"{vals.max():.3f} {config['unit']}"),
        ('Std Dev', f"{vals.std():.3f} {config['unit']}"),
        ('Points',  f"{len(vals):,} grid points"),
        ('Samples', '208 cLHS field points'),
    ]

    context = {
        'pH'          :
            '⚠️ Mean pH 8.39 indicates strong '
            'alkalinity across Khamanon Block. '
            'Values above 8.5 risk micronutrient '
            'deficiency.',
        'OC'          :
            '⚠️ Mean OC 0.43% is critically low. '
            'Healthy soil needs >0.75%. '
            'Indicates long-term organic matter depletion.',
        'EC'          :
            '✓ EC below 0.15 dS/m — no significant '
            'salinity stress in Khamanon Block.',
        'available_N' :
            'Available N ranges 174–268 kg/ha. '
            'High spatial variability suggests '
            'uneven fertiliser application.',
        'available_P' :
            'Available P ranges 16–57 kg/ha. '
            'Central bank recommendation is '
            '25–35 kg/ha for wheat.',
        'K2O'         :
            'K2O ranges 44–124 kg/ha. '
            'Alluvial soils of Punjab generally '
            'have adequate potassium.',
        'CEC'         :
            'CEC 10–20 meq/100g typical for '
            'loamy alluvial soils of central Punjab.',
        'bulk_density':
            'Bulk density 188–277 g/L. '
            'Values above 260 suggest compaction '
            'from heavy machinery.',
        'CaCO3'       :
            'CaCO3 presence contributes to '
            'alkaline pH. High values lock up '
            'phosphorus and micronutrients.'
    }

    divs = []
    for label, value in stats:
        divs.append(html.Div([
            html.Span(
                label + ': ',
                style={'color':'#7f8c8d',
                       'fontWeight':'bold',
                       'fontSize':'13px'}
            ),
            html.Span(
                value,
                style={'color':'#2c3e50',
                       'fontSize':'13px'}
            )
        ], style={'marginBottom':'10px',
                  'borderBottom':'1px solid #ecf0f1',
                  'paddingBottom':'8px'}))

    divs.append(html.Div(
        context.get(prop, ''),
        style={
            'backgroundColor':'#eafaf1',
            'padding'        :'10px',
            'borderRadius'   :'6px',
            'fontSize'       :'12px',
            'color'          :'#1e8449',
            'marginTop'      :'10px'
        }
    ))

    return fig, divs


# ============================================
# RUN
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  DSS DASHBOARD — REAL DATA")
    print("=" * 55)
    print("\nOpen browser and go to:")
    print("  http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop.")
    print("=" * 55 + "\n")
    app.run(debug=False, port=8050)