from typing import Optional
import hsfs
import hopsworks

import src.config as config 

def getFeatureStore() -> hsfs.feature_store.FeatureStore:
    """Connect to Hopsworks and return feature store pointer

    Returns:
        hsfs.feature_store.FeatureStore: pointer to feature store
    """
    
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    
    return project.get_feature_store()

def getFeatureGroup(
    name: str,
    version: Optional[int] = 1
) -> hsfs.feature_group.FeatureGroup:
    """Connects to the feature store and returns a pointer to the given feature group 'name'

    Args:
        name (str): name of feature group
        version (Optional[int], optional): Version number

    Returns:
        hsfs.feature_group.FeatureGroup: pointer to feature group
    """
    
    return getFeatureStore().get_feature_group(
        name=name,
        version=version
    )
    