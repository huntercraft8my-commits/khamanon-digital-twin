# ================================================================
# Khamanon Field Operations Monitor — Dashboard Tab
# Run  : python field_ops_viz.py
# Open : http://127.0.0.1:8051
# ================================================================

import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import json

# ── COLORS — same as dashboard_v2.py ────────────────────────────
C = {
    'primary'   : '#1a1f2e',
    'secondary' : '#252b3b',
    'accent'    : '#00d4aa',
    'warning'   : '#f59e0b',
    'danger'    : '#ef4444',
    'success'   : '#10b981',
    'text'      : '#f1f5f9',
    'muted'     : '#94a3b8',
    'card'      : '#1e2535',
    'border'    : '#2d3748',
    'bg'        : '#1a1f2e',
}

EVENT_COLOR = {
    'HARVESTING'        : '#f59e0b',
    'STUBBLE_BURNING'   : '#ef4444',
    'RICE_TRANSPLANTING': '#00d4aa',
    'FIELD_FLOODING'    : '#3b82f6',
    'PLOUGHING'         : '#f97316',
}

EVENT_LABEL = {
    'HARVESTING'        : 'Wheat/Rice Harvest',
    'STUBBLE_BURNING'   : 'Stubble Burning',
    'RICE_TRANSPLANTING': 'Rice Transplanting',
    'FIELD_FLOODING'    : 'Field Flooding',
    'PLOUGHING'         : 'Ploughing',
}

EVENT_ICON = {
    'HARVESTING'        : '🌾',
    'STUBBLE_BURNING'   : '🔥',
    'RICE_TRANSPLANTING': '🌱',
    'FIELD_FLOODING'    : '💧',
    'PLOUGHING'         : '🚜',
}

CARD = {
    'backgroundColor': C['card'],
    'borderRadius'   : '12px',
    'padding'        : '20px',
    'border'         : f"1px solid {C['border']}",
}

# ── LOAD DATA ────────────────────────────────────────────────────
ts = pd.read_csv('../data/multiindex_timeseries_clean.csv')
ts['date'] = pd.to_datetime(ts['date'])

ev = pd.read_csv('../data/field_events.csv')
ev['date'] = pd.to_datetime(ev['date'])

with open('../data/field_ops_status.json') as f:
    st = json.load(f)


# ── CHART 1: Multi-index time series with event markers ──────────
def make_timeseries_chart():
    fig = go.Figure()

    # Season background bands
    bands = [
        ('2025-01-01', '2025-05-15', 'Rabi wheat 2024-25', '#1D9E75'),
        ('2025-06-01', '2025-11-30', 'Kharif rice 2025',   '#378ADD'),
        ('2025-12-01', '2026-05-15', 'Rabi wheat 2025-26', '#1D9E75'),
    ]
    for x0, x1, label, col in bands:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=col,
                      opacity=0.07, layer='below', line_width=0)

    # NDVI
    fig.add_trace(go.Scatter(
        x=ts['date'], y=ts['NDVI'], name='NDVI',
        mode='lines', line=dict(color='#10b981', width=2.5),
        hovertemplate='<b>NDVI</b>: %{y:.3f}<br>%{x|%d %b %Y}<extra></extra>'
    ))
    # NBR
    fig.add_trace(go.Scatter(
        x=ts['date'], y=ts['NBR'], name='NBR',
        mode='lines', line=dict(color='#ef4444', width=1.8, dash='dash'),
        hovertemplate='<b>NBR</b>: %{y:.3f}<br>%{x|%d %b %Y}<extra></extra>'
    ))
    # BSI
    fig.add_trace(go.Scatter(
        x=ts['date'], y=ts['BSI'], name='BSI',
        mode='lines', line=dict(color='#f59e0b', width=1.8, dash='dot'),
        hovertemplate='<b>BSI</b>: %{y:.3f}<br>%{x|%d %b %Y}<extra></extra>'
    ))

    # Zero line
    fig.add_hline(y=0, line_color='rgba(255,255,255,0.12)',
                  line_width=1, line_dash='solid')

    # Event vertical markers
    for _, row in ev.iterrows():
        col   = EVENT_COLOR.get(row['event'], C['muted'])
        label = EVENT_LABEL.get(row['event'], row['event'])
        fig.add_vline(
            x=str(row['date'].date()),
            line_color=col, line_width=1.5,
            line_dash='dot', opacity=0.85
        )
        fig.add_annotation(
            x=row['date'], y=0.95, yref='paper',
            text=f"<b>{label}</b>",
            showarrow=False, textangle=-90,
            font=dict(color=col, size=10),
            xanchor='left', yanchor='top'
        )

    fig.update_layout(
        plot_bgcolor=C['bg'], paper_bgcolor=C['bg'],
        font=dict(color=C['text']),
        height=380,
        margin=dict(l=50, r=20, t=50, b=60),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            orientation='h', x=0, y=-0.18,
            font=dict(size=12)
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            tickformat='%b %Y', tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            title='Index value',
            range=[-0.25, 0.90],
            zeroline=False,
        ),
        hovermode='x unified',
        title=dict(
            text='Spectral indices — 16 months · Sentinel-2',
            font=dict(size=13, color=C['muted']),
            x=0, pad=dict(l=10)
        )
    )
    return fig


# ── CHART 2: Event timeline ───────────────────────────────────────
def make_timeline_chart():
    fig = go.Figure()

    y_pos = {
        'HARVESTING'        : 3,
        'STUBBLE_BURNING'   : 2,
        'RICE_TRANSPLANTING': 1,
        'FIELD_FLOODING'    : 0,
        'PLOUGHING'         : 0,
    }
    y_tick_labels = ['Other', 'Rice Transplant', 'Stubble Burning', 'Harvest']

    for etype in EVENT_COLOR:
        subset = ev[ev['event'] == etype]
        if subset.empty:
            continue
        sizes = [18 if c == 'HIGH' else 13 for c in subset['confidence']]
        fig.add_trace(go.Scatter(
            x=subset['date'],
            y=[y_pos.get(etype, 0)] * len(subset),
            mode='markers',
            name=EVENT_LABEL.get(etype, etype),
            marker=dict(
                color=EVENT_COLOR[etype],
                size=sizes,
                symbol='diamond',
                line=dict(color='white', width=1.5)
            ),
            hovertemplate=(
                f"<b>{EVENT_LABEL.get(etype, etype)}</b><br>"
                "%{x|%d %b %Y}<br>"
                "<extra></extra>"
            )
        ))

    # Horizontal axis line
    fig.add_shape(type='line',
        x0=ts['date'].min(), x1=ts['date'].max(),
        y0=-0.6, y1=-0.6,
        line=dict(color=C['border'], width=1))

    # Season label annotations on timeline
    fig.add_annotation(x='2025-02-15', y=3.4, text='Rabi 2024-25',
        showarrow=False, font=dict(color='#1D9E75', size=10))
    fig.add_annotation(x='2025-08-15', y=3.4, text='Kharif 2025',
        showarrow=False, font=dict(color='#378ADD', size=10))
    fig.add_annotation(x='2026-02-15', y=3.4, text='Rabi 2025-26',
        showarrow=False, font=dict(color='#1D9E75', size=10))

    fig.update_layout(
        plot_bgcolor=C['bg'], paper_bgcolor=C['bg'],
        font=dict(color=C['text']),
        height=240,
        margin=dict(l=130, r=20, t=30, b=50),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            tickformat='%b %Y',
            range=[ts['date'].min(), ts['date'].max()],
        ),
        yaxis=dict(
            tickvals=[0, 1, 2, 3],
            ticktext=y_tick_labels,
            gridcolor='rgba(255,255,255,0.05)',
            range=[-0.7, 3.7],
            tickfont=dict(size=11)
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            orientation='h', x=0, y=-0.25,
            font=dict(size=11)
        ),
        title=dict(
            text='Field event timeline — Jan 2025 to May 2026',
            font=dict(size=13, color=C['muted']),
            x=0, pad=dict(l=10)
        )
    )
    return fig


# ── STATUS CARDS ─────────────────────────────────────────────────
last_ev   = ev.iloc[-1]
last_label= EVENT_LABEL.get(last_ev['event'], last_ev['event'])
last_date = last_ev['date'].strftime('%b %d, %Y')
last_color= EVENT_COLOR.get(last_ev['event'], C['accent'])

def stat_card(title, value, sub, color):
    return html.Div([
        html.P(title, style={
            'color': C['muted'], 'fontSize': '11px', 'margin': '0 0 8px 0',
            'textTransform': 'uppercase', 'letterSpacing': '1px'
        }),
        html.P(value, style={
            'color': color, 'fontSize': '22px',
            'fontWeight': '600', 'margin': '0 0 4px 0', 'lineHeight': '1.2'
        }),
        html.P(sub, style={
            'color': C['muted'], 'fontSize': '12px', 'margin': '0'
        }),
    ], style={**CARD, 'flex': '1'})


# ── EVENT LOG TABLE ───────────────────────────────────────────────
def event_table():
    rows = []
    for _, row in ev.iterrows():
        col   = EVENT_COLOR.get(row['event'], C['muted'])
        label = EVENT_LABEL.get(row['event'], row['event'])
        icon  = EVENT_ICON.get(row['event'], '')
        conf_col = C['success'] if row['confidence'] == 'HIGH' else C['warning']
        rows.append(html.Tr([
            html.Td(row['date'].strftime('%d %b %Y'),
                    style={'color': C['muted'], 'padding': '10px 14px',
                           'whiteSpace': 'nowrap', 'fontSize': '13px'}),
            html.Td(
                html.Span(f"{icon}  {label}", style={
                    'backgroundColor': col + '22',
                    'color': col,
                    'padding': '4px 12px',
                    'borderRadius': '20px',
                    'fontSize': '12px',
                    'fontWeight': '500',
                }),
                style={'padding': '10px 14px'}
            ),
            html.Td(
                html.Span(row['confidence'], style={
                    'color': conf_col, 'fontWeight': '500', 'fontSize': '13px'
                }),
                style={'padding': '10px 14px'}
            ),
            html.Td(row['note'],
                    style={'color': C['muted'], 'padding': '10px 14px',
                           'fontSize': '12px'}),
        ], style={'borderBottom': f"1px solid {C['border']}"}))

    return html.Table(
        [html.Thead(html.Tr([
            html.Th(h, style={
                'color': C['muted'], 'fontSize': '11px',
                'textTransform': 'uppercase', 'letterSpacing': '1px',
                'padding': '10px 14px', 'textAlign': 'left',
                'borderBottom': f"1px solid {C['border']}"
            })
            for h in ['Date', 'Field Operation', 'Confidence', 'Scientific basis']
        ])),
         html.Tbody(rows)],
        style={'width': '100%', 'borderCollapse': 'collapse'}
    )


# ── APP LAYOUT ───────────────────────────────────────────────────
app = dash.Dash(__name__, title='Khamanon Field Ops Monitor')

app.layout = html.Div([

    # ── Header ──
    html.Div([
        html.Div([
            html.H2('⬡ Khamanon DSS',
                    style={'color': C['accent'], 'margin': '0',
                           'fontSize': '20px', 'fontWeight': '600'}),
            html.P('Field Operations Monitor · Fatehgarh Sahib, Punjab · Real-Time Sentinel-2',
                   style={'color': C['muted'], 'margin': '4px 0 0 0', 'fontSize': '12px'}),
        ]),
        html.Div([
            html.Span('● LIVE', style={'color': C['success'],
                      'fontSize': '12px', 'fontWeight': '600'}),
            html.Span(f"  S2: {st['latest_date']}  |  NDVI: {st['latest_NDVI']:.3f}  |  "
                      f"NBR: {st['latest_NBR']:.3f}",
                      style={'color': C['muted'], 'fontSize': '12px'}),
        ], style={'textAlign': 'right'}),
    ], style={
        'backgroundColor': C['primary'],
        'padding': '16px 28px',
        'display': 'flex', 'justifyContent': 'space-between',
        'alignItems': 'center',
        'borderBottom': f"1px solid {C['border']}",
        'position': 'sticky', 'top': '0', 'zIndex': '100'
    }),

    # ── Main content ──
    html.Div([

        # ── Status cards ──
        html.Div([
            stat_card('Current Phase',
                      'Post-Harvest Fallow', 'May 2026 · bare soil',
                      C['warning']),
            stat_card('Last Detected Event',
                      last_label, last_date,
                      last_color),
            stat_card('Next Expected',
                      'Rice Transplanting', 'June – July 2026',
                      C['success']),
            stat_card('S2 Images Processed',
                      str(st['total_images']), 'Jan 2025 → May 2026',
                      C['accent']),
            stat_card('Events Detected',
                      str(st['total_events']), '2 full crop cycles',
                      C['accent']),
        ], style={
            'display': 'flex', 'gap': '16px',
            'marginBottom': '20px', 'flexWrap': 'wrap'
        }),

        # ── Time series chart ──
        html.Div([
            dcc.Graph(figure=make_timeseries_chart(),
                      config={'displayModeBar': False})
        ], style={**CARD, 'marginBottom': '20px'}),

        # ── Event timeline ──
        html.Div([
            dcc.Graph(figure=make_timeline_chart(),
                      config={'displayModeBar': False})
        ], style={**CARD, 'marginBottom': '20px'}),

        # ── Event log table ──
        html.Div([
            html.H3('Detected field operations log',
                    style={'color': C['text'], 'fontSize': '14px',
                           'fontWeight': '500', 'margin': '0 0 16px 0'}),
            event_table(),
        ], style=CARD),

    ], style={'padding': '24px 28px',
              'backgroundColor': C['primary'],
              'minHeight': '100vh'}),

], style={
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    'backgroundColor': C['primary'],
})

if __name__ == '__main__':
    app.run(debug=False, port=8051)