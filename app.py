import os
import ast
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go



DATA_DIR = 'data'

def list_feather_files():
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.feather')]

def load_df(filename):
    df = pd.read_feather(os.path.join(DATA_DIR, filename))
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df = df[numeric_cols]
    df = df.interpolate().resample('1min').mean()
    return df

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('BIPV Heat Dashboard'),
    html.Label('Select data file:'),
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': f, 'value': f} for f in list_feather_files()],
        value=None,
        placeholder='Choose a feather file',
    ),
    html.Br(),
    html.Label('Select series (multiple allowed):'),
    dcc.Dropdown(id='series-dropdown', multi=True),
    html.Br(),
    dcc.Graph(id='timeseries-plot'),
])

@app.callback(
    Output('series-dropdown', 'options'),
    Output('series-dropdown', 'value'),
    Input('file-dropdown', 'value')
)
def update_series_options(selected_file):
    if not selected_file:
        return [], []
    df = load_df(selected_file)
    options = [{'label': str(col), 'value': str(col)} for col in df.columns]
    return options, [options[0]['value']] if options else []

@app.callback(
    Output('timeseries-plot', 'figure'),
    Input('file-dropdown', 'value'),
    Input('series-dropdown', 'value')
)
def update_plot(selected_file, selected_series):
    if not (selected_file and selected_series):
        return {}
    df = load_df(selected_file)
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
        return {}
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=str(col)))
    fig.update_layout(
        legend_title_text='Series',
        xaxis_title='Time',
        yaxis_title='Value',
        showlegend=True  # Always show legend, even for one trace
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)

server = app.server  # Expose the Flask server for WSGI 