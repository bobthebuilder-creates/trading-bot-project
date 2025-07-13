"""
Basic dashboard for monitoring the trading bot
"""
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TradingDashboard:
    """Basic dashboard for monitoring trading bot performance"""
    
    def __init__(self, port: int = 8050):
        self.app = dash.Dash(__name__)
        self.port = port
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Trading Bot Dashboard", style={'textAlign': 'center'}),
            
            # Performance metrics
            html.Div([
                html.H3("Performance Metrics"),
                html.Div(id='performance-metrics')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            # Current positions
            html.Div([
                html.H3("Current Positions"),
                html.Div(id='current-positions')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            # Price chart
            html.Div([
                html.H3("Price Chart"),
                dcc.Graph(id='price-chart')
            ]),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('performance-metrics', 'children'),
             Output('current-positions', 'children'),
             Output('price-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # TODO: Get real data from trading bot
            metrics = html.Div([
                html.P("Total PnL: $0.00"),
                html.P("Daily PnL: $0.00"),
                html.P("Win Rate: 0%"),
                html.P("Sharpe Ratio: 0.0")
            ])
            
            positions = html.Div([
                html.P("No active positions")
            ])
            
            # Empty chart for now
            figure = {
                'data': [],
                'layout': {
                    'title': 'Price Chart (Coming Soon)',
                    'xaxis': {'title': 'Time'},
                    'yaxis': {'title': 'Price'}
                }
            }
            
            return metrics, positions, figure
    
    def run(self, debug: bool = True):
        """Run the dashboard"""
        logger.info(f"Starting dashboard on port {self.port}")
        self.app.run_server(debug=debug, port=self.port, host='0.0.0.0')
