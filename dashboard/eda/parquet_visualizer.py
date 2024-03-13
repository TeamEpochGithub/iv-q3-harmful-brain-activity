import glob
import plotly.graph_objects as go
import os
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd

# Define function to visualize Parquet files
def create_parquet_visualizer(app, file_path):

    # Define layout
    layout = html.Div([
        html.H1('EEG Visualizer'),
        html.P('EEG id'),
        dcc.Dropdown(
            id='parquet-dropdown',
            options=[],
            value=None
        ),
        html.P('Feature'),
        dcc.Dropdown(
            id='parquet-column-selector',
            options=['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG'],
            value='Fp1'
        ),
        dcc.Graph(id='parquet-graph'),
        dcc.Graph(id='label-graph'),
    ])
    
    # Register callbacks
    @app.callback(
        Output('parquet-dropdown', 'options'),
        [Input('parquet-dropdown', 'value')]
    )
    def update_dropdown(selected_file):
        parquet_files = glob.glob(file_path)[:100]
        options = [{'label': file, 'value': file} for file in parquet_files]
        return options
    
    @app.callback(
        Output('parquet-graph', 'figure'),
        [Input('parquet-dropdown', 'value'), Input('parquet-column-selector', 'value')]
    )
    def update_graph(selected_file, column):
        if selected_file is None:
            return {}
        
        df = pd.read_parquet(selected_file)

        # Assuming your dataframe has columns 'x' and 'y', you can create a scatter plot
        trace = go.Line(
            x=df.index,
            y=df[column],
            marker=dict(color='blue')  # You can customize marker properties
        )
        
        layout = {
            'title': f'Visualization of {selected_file}',
            'xaxis': {'title': 'Step (200Hz)'},
            'yaxis': {'title': 'µV'}
        }
        
        return {'data': [trace], 'layout': layout}
    
    @app.callback(
        Output('label-graph', 'figure'),
        [Input('parquet-dropdown', 'value')]
    )
    def update_label_graph(selected_file):
        if selected_file is None:
            return {}
        
        df = pd.read_csv('./data/raw/train.csv')

        # Split selected_file on /
        file_name = selected_file.split('/')[-1]
        eeg_id = file_name[:-8]
        
        matching = df[df['eeg_id'] == eeg_id]

        columns = list(filter(lambda col: col.endswith('_vote'), matching.columns))
        trace = go.Figure()

        for i, x in matching.iterrows():
            print(x)
            bar = go.Bar(
                x= columns,
                y = x[columns]
            )
            trace.add_trace(bar)


        layout = {
            'title' : f'Labels of {eeg_id}',
        }

        return {'data': [trace], 'layout': layout}

    return app, layout