import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from dashboard.eda.eda_layout import create_eda_layout
from dashboard.eda.parquet_visualizer import create_parquet_visualizer

# Initialize the Dash app
app = dash.Dash(__name__)

parquet_app, parquet_layout = create_parquet_visualizer(app=app, file_path='./data/raw/train_eegs/*')

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Q3 - Harmful Brain Activity', style={'textAlign': 'center'}),
    
    dcc.Tabs(id='tabs', value='eda', children=[
        dcc.Tab(label='EDA', value='eda'),
        dcc.Tab(label='Model', value='model'),
        dcc.Tab(label='Predictions', value='predictions')
    ]),
    
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'eda':
        eda_app, eda_layout = create_eda_layout(app=app)
        return eda_layout 

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
