import pandas as pd
import numpy as np

def read_initial_conditions(path):
    """
    Reads initial pollutant concentration data from a CSV file.

    Expected CSV columns:
        x  - position along the river (m)
        C  - concentration (µg/m³)
    """
    return pd.read_csv(path)

def interpolate_to_grid(df, grid_x):
    """
    Placeholder for interpolation of CSV data onto the model grid.

    Parameters:
        df      : DataFrame with columns ['x', 'C']
        grid_x  : 1D numpy array of model grid positions

    Returns:
        1D numpy array of interpolated concentrations on grid_x.
    """
    # To be implemented by Initial Conditions Lead
    pass

