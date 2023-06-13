import os
from dotenv import load_dotenv

from src.paths import PARENT_DIR

# load key-value pairs from .env file located in  the parent directory
load_dotenv(PARENT_DIR / ".env")

HOPSWORKS_PROJECT_NAME = "ml_prod_pipline"

try:
    HOPSWORKS_API_KEY  = os.environ["HOPSWORKS_API_KEY"]
except:
    raise Exception("Create an .env file in the project root dir with the HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "ts_hourly_feat_group"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "ts_hourly_feat_view"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 3

#Number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28

# added for monitoring purposes
FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group'
FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view'
FEATURE_VIEW_MONITORING = 'predictions_vs_actuals_for_monitoring_feature_view'
