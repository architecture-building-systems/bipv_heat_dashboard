import os
import ast
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go

THIS_FOLDER = Path(__file__).parent.resolve()
DATA_DIR = os.path.join(THIS_FOLDER, 'data')

def list_feather_files():
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.feather')]

def get_experiment_log_file():
    """Get the single experiment log file"""
    pkl_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
    if len(pkl_files) == 0:
        return None
    elif len(pkl_files) == 1:
        return pkl_files[0]
    else:
        # Multiple files found, return the first one or could implement priority logic
        return pkl_files[0]

def load_experiment_log():
    """Load the experiment log file"""
    log_file = get_experiment_log_file()
    if not log_file:
        return None
    try:
        with open(os.path.join(DATA_DIR, log_file), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading experiment log {log_file}: {e}")
        return None

def parse_experiment_time(date_str, time_str):
    """Convert date and time strings to datetime object"""
    try:
        # Parse date like '210725' as DDMMYY to '25-07-21' -> '2021-07-25'
        if len(date_str) == 6:
            day = date_str[:2]
            month = date_str[2:4]
            year = '20' + date_str[4:6]
            date_formatted = f"{year}-{month}-{day}"
        else:
            date_formatted = date_str
        
        # Combine date and time
        datetime_str = f"{date_formatted} {time_str}"
        return pd.to_datetime(datetime_str)
    except:
        return None

def get_feather_filename_from_experiment(experiment_code):
    """Get feather filename from experiment code"""
    return f"{experiment_code}.feather"

def feather_file_exists(experiment_code):
    """Check if feather file exists for the experiment"""
    feather_filename = get_feather_filename_from_experiment(experiment_code)
    return os.path.exists(os.path.join(DATA_DIR, feather_filename))

def load_df(filename):
    df = pd.read_feather(os.path.join(DATA_DIR, filename))
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df = df[numeric_cols]
    df = df.interpolate().resample('1min').mean()
    return df

def get_unit(col):
    # For MultiIndex columns like ('Comfort Cube', 'metabolic_rate-mets')
    # The unit is in the second element after the '-'
    if isinstance(col, tuple):
        # Get the second level (child) which contains the measurement-unit
        measurement_with_unit = col[1] if len(col) > 1 else col[0]
    else:
        measurement_with_unit = col
    
    if '-' in str(measurement_with_unit):
        unit = str(measurement_with_unit).split('-')[-1].strip()
        return unit
    return 'Unknown'

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('BIPV Heat Dashboard', style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Top section - 20% height, split into two 50/50 columns
    html.Div([
        # Left column - Experiment selection
        html.Div([
            html.H3('Experiment Selection'),
            html.Label('Select experiment:'),
            dcc.Dropdown(id='experiment-dropdown'),
            html.Br(),
            html.Button('Download Raw Data', id='download-button', n_clicks=0, 
                       style={'marginTop': '10px', 'width': '100%'}),
            dcc.Download(id='download-data')
        ], style={
            'width': '50%', 
            'display': 'inline-block', 
            'verticalAlign': 'top',
            'border': '1px solid #000'
        }),
        
        # Right column - Experiment information
        html.Div([
            html.H3('Experiment Information'),
            html.Div(id='file-status', style={'marginBottom': '10px'}),
            html.Div(id='experiment-details', style={'fontSize': '14px', 'lineHeight': '1.4'}),
        ], style={
            'width': '50%', 
            'display': 'inline-block', 
            'verticalAlign': 'top',
            'border': '1px solid #000'
        }),
    ], style={
        'height': '20vh',
        'overflow': 'auto',
        'marginBottom': '5px'
    }),
    
    # Bottom section - 80% height, split into left (plot) and right (controls)
    html.Div([
        # Left side - Plot area (70%)
        html.Div([
            html.Div(id='unit-warning', style={'color': 'red', 'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Graph(id='timeseries-plot', style={'height': '68vh'}),
        ], style={
            'width': '70%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'border': '1px solid #000'
        }),
        
        # Right side - Controls and Legend (30%)
        html.Div([
            html.H4('Series Selection'),
            html.Label('Select series (multiple allowed):'),
            dcc.Dropdown(id='series-dropdown', multi=True),
            html.Br(),
            html.H4('Legend'),
            html.Div(id='custom-legend', style={'fontSize': '12px', 'lineHeight': '1.6'}),
        ], style={
            'width': '28%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'border': '1px solid #000',
            'height': '70vh',
            'overflow': 'auto'
        }),
    ], style={
        'height': '70vh'
    })
])

@app.callback(
    Output('download-data', 'data'),
    Input('download-button', 'n_clicks'),
    State('experiment-dropdown', 'value'),
    prevent_initial_call=True
)
def download_experiment_data(n_clicks, selected_experiment):
    if not selected_experiment:
        return None
    
    feather_filename = get_feather_filename_from_experiment(selected_experiment)
    feather_path = os.path.join(DATA_DIR, feather_filename)
    
    if not os.path.exists(feather_path):
        return None
    
    return dcc.send_file(feather_path)

@app.callback(
    Output('experiment-dropdown', 'options'),
    Output('experiment-dropdown', 'value'),
    Input('experiment-dropdown', 'id')  # Triggers on app load
)
def update_experiment_options(_):
    experiment_log = load_experiment_log()
    if not experiment_log:
        return [{'label': 'No experiment log found', 'value': None}], None
    
    # Only show experiments that have corresponding feather files
    available_experiments = []
    for exp_code in experiment_log.keys():
        if feather_file_exists(exp_code):
            exp_data = experiment_log[exp_code]
            label = f"{exp_code} ({exp_data.get('Date', 'Unknown date')})"
            available_experiments.append({'label': label, 'value': exp_code})
    
    if not available_experiments:
        return [{'label': 'No experiments with data files found', 'value': None}], None
    
    return available_experiments, None

@app.callback(
    Output('series-dropdown', 'options'),
    Output('series-dropdown', 'value'),
    Output('file-status', 'children'),
    Output('experiment-details', 'children'),
    Input('experiment-dropdown', 'value')
)
def update_series_options(selected_experiment):
    if not selected_experiment:
        return [], [], '', ''
    
    feather_filename = get_feather_filename_from_experiment(selected_experiment)
    
    if not feather_file_exists(selected_experiment):
        error_msg = f'Error: Data file {feather_filename} not found for experiment {selected_experiment}'
        return [], [], error_msg, ''
    
    try:
        df = load_df(feather_filename)
        options = [{'label': str(col), 'value': str(col)} for col in df.columns]
        status = f'Loaded data from: {feather_filename}'
        
        # Get experiment details
        experiment_log = load_experiment_log()
        exp_details = []
        if experiment_log and selected_experiment in experiment_log:
            exp_data = experiment_log[selected_experiment]
            exp_details = [
                html.P([html.Strong('Date: '), exp_data.get('Date', 'Unknown')]),
                html.P([html.Strong('Monitor: '), exp_data.get('Monitor name', 'Unknown')]),
                html.P([html.Strong('IR: '), f"{exp_data.get('IR', 'N/A')}%"]),
                html.P([html.Strong('Warm White: '), f"{exp_data.get('WW', 'N/A')}%"]),
                html.P([html.Strong('Cold White: '), f"{exp_data.get('CW', 'N/A')}%"]),
                html.P([html.Strong('Notes: '), exp_data.get('Notes', 'None')], style={'fontStyle': 'italic'})
            ]
        
        return options, [options[0]['value']] if options else [], status, exp_details
    except Exception as e:
        error_msg = f'Error loading data from {feather_filename}: {e}'
        return [], [], error_msg, ''

@app.callback(
    Output('custom-legend', 'children'),
    Input('series-dropdown', 'value'),
    Input('experiment-dropdown', 'value')
)
def update_custom_legend(selected_series, selected_experiment):
    if not (selected_series and selected_experiment):
        return ''
    
    feather_filename = get_feather_filename_from_experiment(selected_experiment)
    
    if not feather_file_exists(selected_experiment):
        return ''
    
    try:
        df = load_df(feather_filename)
        
        # Convert stringified tuples back to tuples if needed
        cols = []
        for s in selected_series:
            try:
                col = ast.literal_eval(s)
            except Exception:
                col = s
            if col in df.columns:
                cols.append(col)
        
        if not cols:
            return ''
        
        # Group columns by unit for legend styling
        unit_map = {}
        for col in cols:
            unit = get_unit(col)
            if unit not in unit_map:
                unit_map[unit] = []
            unit_map[unit].append(col)
        
        units = list(unit_map.keys())
        legend_items = []
        
        # Primary axis (solid lines)
        if len(units) > 0:
            legend_items.append(html.H5(f'Primary Axis ({units[0]})', style={'margin': '10px 0 5px 0', 'color': '#333'}))
            for col in unit_map[units[0]]:
                legend_items.append(
                    html.Div([
                        html.Span('━━', style={'color': '#1f77b4', 'marginRight': '5px', 'fontSize': '16px'}),
                        html.Span(str(col), style={'fontSize': '11px'})
                    ], style={'marginBottom': '3px'})
                )
        
        # Secondary axis (dashed lines)
        if len(units) > 1:
            legend_items.append(html.H5(f'Secondary Axis ({units[1]})', style={'margin': '15px 0 5px 0', 'color': '#333'}))
            for col in unit_map[units[1]]:
                legend_items.append(
                    html.Div([
                        html.Span('╌╌', style={'color': '#ff7f0e', 'marginRight': '5px', 'fontSize': '16px'}),
                        html.Span(str(col), style={'fontSize': '11px'})
                    ], style={'marginBottom': '3px'})
                )
        
        return legend_items
        
    except Exception:
        return ''

@app.callback(
    Output('timeseries-plot', 'figure'),
    Output('unit-warning', 'children'),
    Input('experiment-dropdown', 'value'),
    Input('series-dropdown', 'value')
)
def update_plot(selected_experiment, selected_series):
    if not (selected_experiment and selected_series):
        return {}, ''
    
    feather_filename = get_feather_filename_from_experiment(selected_experiment)
    
    if not feather_file_exists(selected_experiment):
        return {}, f'Error: Data file not found for experiment {selected_experiment}'
    
    df = load_df(feather_filename)
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Convert stringified tuples back to tuples if needed, and filter only existing columns
    cols = []
    for s in selected_series:
        try:
            col = ast.literal_eval(s)
        except Exception:
            col = s
        if col in df.columns:
            cols.append(col)
    
    if not cols:
        return {}, ''
    
    # Group columns by unit
    unit_map = {}
    for col in cols:
        unit = get_unit(col)
        if unit not in unit_map:
            unit_map[unit] = []
        unit_map[unit].append(col)
    
    units = list(unit_map.keys())
    warning = ''
    if len(units) > 2:
        warning = f'Warning: More than two units selected ({", ".join(units)}). Only the first two will be plotted.'
        units = units[:2]
        # Keep only columns from first two units
        cols = []
        for unit in units:
            cols.extend(unit_map[unit])
    
    # Create figure with secondary y-axis if needed
    use_secondary = len(units) > 1
    fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])
    
    # Plot first unit on primary y-axis
    for col in unit_map[units[0]]:
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df[col], 
                mode='lines', 
                name=str(col),
                line=dict(width=2)
            ),
            secondary_y=False
        )
    
    # Plot second unit on secondary y-axis with dashed lines
    if use_secondary:
        for col in unit_map[units[1]]:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col], 
                    mode='lines', 
                    name=str(col),
                    line=dict(dash='dash', width=2)
                ),
                secondary_y=True
            )
        
        # Set y-axes titles
        fig.update_yaxes(title_text=f"Primary ({units[0]})", secondary_y=False)
        fig.update_yaxes(title_text=f"Secondary ({units[1]})", secondary_y=True)
    else:
        fig.update_yaxes(title_text=f"Value ({units[0]})", secondary_y=False)
    
    # Add experiment phase lines
    experiment_log = load_experiment_log()
    if experiment_log and selected_experiment in experiment_log:
        exp_data = experiment_log[selected_experiment]
        date = exp_data.get('Date', '')
        
        # Define the phases and their colors
        phases = [
            ('Start time', 'green', 'solid'),
            ('Start Warmup', 'orange', 'dash'),
            ('Start measurement', 'red', 'solid'),
            ('End measurement', 'red', 'dash'),
            ('End cool down', 'blue', 'solid')
        ]
        
        for phase_name, color, line_style in phases:
            if phase_name in exp_data:
                phase_time = parse_experiment_time(date, exp_data[phase_name])
                if phase_time:
                    # Add vertical line using add_shape instead of add_vline
                    fig.add_shape(
                        type="line",
                        x0=phase_time, x1=phase_time,
                        y0=0, y1=1,
                        yref="paper",
                        line=dict(
                            color=color,
                            width=2,
                            dash=line_style
                        )
                    )
                    # Add annotation for the phase name
                    fig.add_annotation(
                        x=phase_time,
                        y=1.02,
                        yref="paper",
                        text=phase_name,
                        showarrow=False,
                        textangle=-45,
                        font=dict(size=10, color=color)
                    )
    
    # Update layout
    fig.update_layout(
        title=f"Experiment: {selected_experiment}",
        xaxis_title="Time",
        showlegend=False  # Hide default legend, we have custom legend on the right
    )
    
    return fig, warning

# Expose Flask server for WSGI
server = app.server

if __name__ == '__main__':
    app.run(debug=True) 