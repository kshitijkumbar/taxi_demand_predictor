from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline

import pandas as pd
import lightgbm as lgb

def avgRidesPerMonth(X: pd.DataFrame) -> pd.DataFrame:
    """Adds one column with average rides from weekly data over a
     month

    Args:
        X (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    X["avg_rides_per_month"] = (X[f'rides_prev_{1*7*24}_hr'] + \
                                X[f'rides_prev_{2*7*24}_hr'] + \
                                X[f'rides_prev_{3*7*24}_hr'] + \
                                X[f'rides_prev_{4*7*24}_hr']) * 0.25

    return X

# Another way to add a transformation
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        
        # Add numerical columns from datetime column
        X_['hour'] = X_['pickup_hr'].dt.hour
        X_['day_of_week'] = X_['pickup_hr'].dt.weekday
           
        return X_.drop(columns=['pickup_hr'])
    
def getPipeline(
   **hyperparams 
) -> Pipeline:
    
    # Only transformation
    add_feat_avg_rides_month = FunctionTransformer(
    avgRidesPerMonth, validate=False
    )
    
    # Another transformation method
    add_temporal_feats = TemporalFeatureEngineer()
    
    return make_pipeline(
                add_feat_avg_rides_month,
                add_temporal_feats,
                lgb.LGBMRegressor(**hyperparams)
            )