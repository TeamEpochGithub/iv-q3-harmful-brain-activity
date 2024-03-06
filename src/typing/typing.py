"""Common type definitions for the project."""

import pandas as pd

# XData is a tuple of EEG data, spectrogram data, and metadata
XData = tuple[dict[str, pd.DataFrame] | None, dict[str, pd.DataFrame] | None, pd.DataFrame]
