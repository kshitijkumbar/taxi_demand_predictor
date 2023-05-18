from datetime import datetime, timedelta

import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config 

def getHsfsProject() -> hopsworks.project.Project:
    
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    
def getFeatStore() -> FeatureStore:
    
    project = getHsfsProject()
    return project.get_feature_store()

def getModelPredictions(model, features: pd.DataFrame) -> pd.DateFrame:
    """"""
    predictions = model.predict(features)
    
    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)
    
    return results

def loadFeatBatchFromStore(
    current_date: datetime,
) -> pd.DataFrame:
    
    feature_store = getFeatStore()
    
    n_features = config.N_FEATURES
    
    # Read time-series data from the feature store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=28)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1))
    )
    
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, 
                                                    fetch_data_to)]
    
    # Validate we are not missing data in the feature store
    location_ids = ts_data["pickup_loc_id"].unique()
    assert len(ts_data) == n_features*len(location_ids), "Time-series is not complete"
    
    # Sort data by location and time
    ts_data.sort_values(by=['pickup_loc_id', 'pickup_hour'], inplace=True)
    print(f"{ts_data=}")
        
    # Transpose time-series data as a feature vector for each location id
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_loc_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=["pickup_hour"])
        x[i, :] = ts_data_i['rides'].values
        
    features = pd.DataFrame(
        x,
        columns=[f"rides_previous_{i+1}_hour" for i in reversed(range(n_features))]        
    )
    features["pickup_hour"] = current_date
    features["pickup_loc_id"] = location_ids
    
    return features

def loadModelFromRegistry():
    import joblib
    from pathlib import Path
    
    project = getHsfsProject()
    
    model_registry = project.get_model_registry()
    
    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / "model.pkl")
    
    return model 
