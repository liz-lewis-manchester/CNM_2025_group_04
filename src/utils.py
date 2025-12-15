import pandas as pd
import numpy as np


def read_initial_conditions(path: str) -> pd.DataFrame:
    """
    Reads initial pollutant concentration data from a CSV file.

    Expected CSV columns:
        x  - position along the river (m)
        C  - concentration (µg/m^3)

    Parameters
    ----------
    path : str
        Showing path to the CSV file (e.g. 'src/data/initial_conditions.csv').

    Returns
    -------
    df : pandas.DataFrame
        Cleaned DataFrame with columns ['x', 'C'], sorted by x.
    """

    # Reads the CSV file
    df = pd.read_csv(path)

    # Renaming first two columns to standard names
    original_cols = list(df.columns)
    if len(original_cols) < 2:
        raise ValueError("Initial conditions CSV must have at least two columns.")

    df = df.rename(columns={
        original_cols[0]: "x",
        original_cols[1]: "C"
    })

    # Keeping only relevant columns, ensure numeric, drop NaNs, sort
    df = (
        df[["x", "C"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna(subset=["x", "C"])
        .sort_values("x")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Initial conditions DataFrame is empty after cleaning.")

    return df


def interpolate_to_grid(df: pd.DataFrame, grid_x: np.ndarray) -> np.ndarray:
    """
    This part interpolates initial-condition data from measurement points
    onto the computational model grid.

    Linear interpolation is used. Values outside the measured
    domain are set to zero.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
            x  - positions along the river (m)
            C  - concentration at those positions (µg/m^3)
    grid_x : np.ndarray
        1D numpy array of model grid positions (m).

    Returns
    -------
    C_grid : np.ndarray
        Initial concentration profile on the model grid, C(x, t=0).
    """

    x_data = df["x"].to_numpy(dtype=float)
    C_data = df["C"].to_numpy(dtype=float)

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
    This part is the convenience wrapper that reads the initial-condition CSV file
    and interpolates it directly onto the model grid.

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
    return interpolate_to_grid(df_ic, grid_x)
