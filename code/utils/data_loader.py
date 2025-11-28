"""
Data loading wrapper for oddball with float32 conversion for PyOD compatibility.

This module provides a drop-in replacement for nonconform.utils.data.load()
that uses oddball as the backend but maintains the same behavior:
- Auto-converts numeric data to float32 (required by PyOD)
- Returns DataFrames for setup=False
- Returns numpy arrays for setup=True
"""

import numpy as np
import pandas as pd
from oddball import load as oddball_load
from typing import Union, Tuple


def load(
    dataset,
    setup: bool = False,
    seed: int = None,
    **kwargs
) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load dataset from oddball with float32 conversion for PyOD compatibility.

    Args:
        dataset: Dataset enum from oddball.Dataset
        setup: If True, returns (x_train, x_test, y_test). If False, returns DataFrame.
        seed: Random seed for setup mode
        **kwargs: Additional arguments passed to oddball.load()

    Returns:
        If setup=False: pandas DataFrame with float32 numeric columns
        If setup=True: tuple of (x_train, x_test, y_test) as float32 numpy arrays
    """
    if setup:
        # Load with setup=True (train/test split)
        if seed is not None:
            kwargs['seed'] = seed

        x_train, x_test, y_test = oddball_load(dataset, setup=True, **kwargs)

        # Convert to numpy arrays if they're DataFrames, then to float32 for PyOD compatibility
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values.astype(np.float32)
        else:
            x_train = x_train.astype(np.float32)

        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.values.astype(np.float32)
        else:
            x_test = x_test.astype(np.float32)

        # y_test should remain as-is (integers)
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        return x_train, x_test, y_test

    else:
        # Load as DataFrame
        df = oddball_load(dataset, as_dataframe=True, **kwargs)

        # Convert all numeric columns except 'Class' to float32
        for col in df.columns:
            if col != 'Class' and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(np.float32)

        return df
