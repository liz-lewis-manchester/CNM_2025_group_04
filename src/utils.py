import pandas as pd
import numpy as np

def read_initial_conditions(path: str) -> pd.DataFrame: 
    """
    Reads initial pollutant concentration data from a CSV file.

    Expected CSV columns:
        x  - position along the river (m)
        C  - concentration (µg/m³)

    Parameters
    ---------
    path : str 
        Path to the CSV file (e.g. 'src/data/initial_conditions.csv').

    Returns
    -------
    df : pandas.DataFrame 
         Data Frame with columns ['x', 'C']
    """

    # read the CSV file
    df = pd.read_csv(path)

    # Rename the first two columns to standard name 'x' and 'C'
    original_cols = df.columns
    if len(original_cols) < 2:
        raise ValueError("Initial conditions CSV must have at least two columns.")
    
    df = df.rename(columns={
        original_cols[0]: "x",
        original_cols[1]: "C"
    })

    # Keep only the two relevant columns and ensure they are numeric
    df = df(
        [["x", "C"]]
        .astype(float)
        .dropna(subset=["x", "C"])
        .sort_values("x")
    )

    if df.emty:
        raise ValueError("Initial conditions DataFrame is empty after cleaning.")
    return df


def interpolate_to_grid(df: pd.DataFrame, grid_x: np.ndarray) -> np.ndarray:
    """
    Interpolates initial-condition data from the CSV onto the model grid.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
            x  - positions along the river (m)
            C  - concentration at those positions (µg/m³)
    grid_x : np.ndarray
        1D numpy array of model grid positions (m).

    Returns
    -------
    C_grid : np.ndarray
        1D numpy array of interpolated concentrations on grid_x (µg/m³),
        representing C(x, t=0) on the model grid.
    """

    # Extract data as numpy arrays
    x_data = df["x"].to_numpy(dtype=float)
    C_data = df["C"].to_numpy(dtype=float)

    # Linear interpolation from measurement points to model grid
    # Values outside the measured range are set to zero
    C_grid = np.interp(
        grid_x,
        x_data,
        C_data,
        left=0.0,
        right=0.0
    )

    return C_grid

def load_initial_condition_on_grid(path: str, grid_x: np.ndarray) -> np.ndarray:
    """
    Convenience function that reads the CSV and returns C(x, 0) on the model grid.

    Parameters
    ----------
    path : str
        Path to the initial-conditions CSV file.
    grid_x : np.ndarray
        1D numpy array of model grid positions (m).

    Returns
    -------
    C0 : np.ndarray
        Initial concentration profile on the model grid, C(x, t=0).
    """
   
    df_ic = read_initial_conditions(path)
    C0 = interpolate_to_grid(df_ic, grid_x)
    return C0
   



