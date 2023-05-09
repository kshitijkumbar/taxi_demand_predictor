from datetime import datetime
from typing import Tuple

import pandas as pd

def trainTestSplit(
    df: pd.DataFrame,
    cutoff_date: datetime,
    tgt_col_name: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate and return train test split in terms of pandas dataframes/series

    Args:
        df (pd.DataFrame): _description_
        cutoff_date (datetime): _description_
        tgt_col_name (str): _description_

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: _description_
    """
    
    train_data = df[df.pickup_hr < cutoff_date].reset_index(drop=True)
    test_data = df[df.pickup_hr >= cutoff_date].reset_index(drop=True)
    
    X_train = train_data.drop(columns=[tgt_col_name])
    X_test = test_data.drop(columns=[tgt_col_name])
    y_train = train_data[tgt_col_name]
    y_test = test_data[tgt_col_name]
    
    
    return X_train, y_train, X_test, y_test