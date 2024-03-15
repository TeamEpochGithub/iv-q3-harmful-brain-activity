"""Module containing the functions to visualize parquet files."""
import glob
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


# Define function to visualize Parquet files
def create_parquet_visualizer(app: Dash, file_path: str) -> tuple[Any, Any]:  # noqa: C901
    """Create a parquet visualizer element in the dashboard.

    :param app: The dashboard app
    :param file_path: The path to read the parquet files
    :return: Updated app and layout
    """
    # Define layout
    layout = html.Div(
        [
            html.H1("EEG Visualizer"),
            html.P("EEG id"),
            dcc.Dropdown(
                id="parquet-dropdown",
                options=[],
                value=None,
            ),
            html.P("Feature"),
            dcc.Dropdown(
                id="parquet-column-selector",
                options=["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"],
                value="Fp1",
            ),
            dcc.Graph(id="parquet-graph"),
            dcc.Graph(id="labels-graph"),
        ],
    )

    # Register callbacks
    @app.callback(
        Output("parquet-dropdown", "options"),
        [Input("parquet-dropdown", "value")],
    )
    def update_dropdown(_: str) -> list[dict[str, str]]:
        parquet_files = glob.glob(file_path)
        return [{"label": file, "value": file} for file in parquet_files]

    @app.callback(
        Output("parquet-graph", "figure"),
        [Input("parquet-dropdown", "value"), Input("parquet-column-selector", "value")],
    )
    def update_graph(selected_file: str, column: str) -> dict[str, Any]:
        if selected_file is None:
            return {}

        eeg_df = pd.read_parquet(selected_file)

        # Split selected_file on '/'
        file_name = selected_file.split("/")[-1]
        # Assuming the eeg_id is the file name without the last 8 characters (e.g., extension)
        eeg_id = file_name[:-8]

        # Ensure eeg_id is compared as the correct type; casting to int might be necessary
        # Adjust this part according to your 'eeg_id' column data type
        try:
            matching_eeg_id = int(eeg_id)
        except ValueError:
            # Handle the case where eeg_id cannot be converted to int
            # ("eeg_id cannot be converted to an integer.")
            return {}

        train_df = pd.read_csv("./data/raw/train.csv")
        matching = train_df[train_df["eeg_id"] == matching_eeg_id]

        # Create a Plotly figure
        fig = go.Figure()

        # Assuming 'column' is defined somewhere in your function to specify which column to plot
        scatter = go.Scatter(
            x=eeg_df.index,
            y=eeg_df[column],
            mode="lines",
            marker={"color": "blue"},
            name="Data",
        )
        fig.add_trace(scatter)

        # Add vertical lines for each eeg_label_offset_seconds
        for offset in matching["eeg_label_offset_seconds"]:
            fig.add_shape(type="line", x0=(offset + 25) * 200, y0=0, x1=(offset + 25) * 200, y1=1, xref="x", yref="paper", line={"color": "Red", "width": 2})

        fig.update_layout(
            title=f"Visualization of {eeg_id}",
            xaxis={"title": "Step (200Hz)"},
            yaxis={"title": "µV"},
            showlegend=True,
        )

        return fig

    @app.callback(
        Output("labels-graph", "figure"),
        [Input("parquet-dropdown", "value")],
    )
    def update_label_graph(selected_file: str) -> dict[str, Any]:
        if selected_file is None:
            return {}

        train_df = pd.read_csv("./data/raw/train.csv")

        # Split selected_file on '/'
        file_name = selected_file.split("/")[-1]
        # Assuming the eeg_id is the file name without the last 8 characters (e.g., extension)
        eeg_id = file_name[:-8]

        # Ensure eeg_id is compared as the correct type; casting to int might be necessary
        # Adjust this part according to your 'eeg_id' column data type
        try:
            matching_eeg_id = int(eeg_id)
        except ValueError:
            # Handle the case where eeg_id cannot be converted to int
            # ("eeg_id cannot be converted to an integer.")
            return {}

        matching = train_df[train_df["eeg_id"] == matching_eeg_id]

        if matching.empty:
            # (f"No matching records found for eeg_id: {eeg_id}")
            return {}

        columns = list(filter(lambda col: col.endswith("_vote"), matching.columns))
        trace = go.Figure()

        for _, x in matching.iterrows():
            # Print statement removed to clean up output, uncomment if needed for debugging
            # print(x[columns].to_numpy())
            bar = go.Bar(
                x=columns,
                y=x[columns].to_numpy(),
                name=f"Offset: {200 * (x['eeg_label_offset_seconds'] + 25)}",  # Optional: name each bar for clarity
            )
            trace.add_trace(bar)

        layout = {
            "title": f"Labels of {eeg_id}",
        }

        # Adjusted the return to provide a figure directly, which is more common with Plotly usage
        trace.update_layout(layout)
        return trace

    return app, layout
