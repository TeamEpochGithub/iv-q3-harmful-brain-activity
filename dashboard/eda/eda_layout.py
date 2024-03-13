from dashboard.eda.parquet_visualizer import create_parquet_visualizer


def create_eda_layout(app):
    return create_parquet_visualizer(app, file_path='./data/raw/train_eegs/*')