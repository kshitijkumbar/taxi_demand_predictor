from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

def downloadRawFile(year: int, month: int) -> Path:
    """
    Downloads Parquet file with historical taxi rides for given month and year combimation
    """
    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    response = requests.get(URL)
    
    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}_{month:02d}.parquet"
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f"{URL} is not available")

def getValidRawData(
    rides: pd.DataFrame,
    year: int,
    month: int
) -> pd.DataFrame:
    """
    Returns valid subset of raw data based on input year and month    
    """
    
    month_start = f"{year}-{month:02d}-01"
    month_end = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-{1:02d}-01"
    rides = rides [(rides.pickup_datetime >= month_start)
                   &(rides.pickup_datetime <= month_end)]
    
    return rides

def loadRawData(
    year: int,
    months: Optional[List[int]] = None
) -> pd.DataFrame:
    
    rides = pd.DataFrame()
    
    if months is None:
        # 
        months = list(range(1,13))
    elif isinstance(months, int):
        months = [months]
    
    for month in months:
        local_file = RAW_DATA_DIR / f"rides_{year}_{month:02d}.parquet"
        
        if not local_file.exists():
            try:
                print(f"File for {year}_{month:02d} not found, Downloading file...")
                downloadRawFile(year, month)
            except:
                print(f"File for {year}_{month:02d} not available")
                continue
        else:
            print(f"File for {year}_{month:02d} found locally...")
            
        # Load raw file into pandas df
        month_rides = pd.read_parquet(local_file)
        
        # Rename columns
        month_rides = month_rides[['tpep_pickup_datetime', 'PULocationID']]
        month_rides.rename(columns={
            'tpep_pickup_datetime' : 'pickup_datetime',
            'PULocationID' : 'pickup_loc_id'
        }, inplace=True)
        
        # Get valid data within time-range
        month_rides = getValidRawData(month_rides, year, month)
        
        # Append to main dataset
        rides = pd.concat([rides, month_rides])
    
    rides = rides[['pickup_datetime', 'pickup_loc_id']]
    
    return rides

def processRawData(
    rides: pd.DataFrame,
) -> pd.DataFrame:
    """"""
    
    # Get all rides per location and pickup hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg_rides = rides.groupby(['pickup_hour', 'pickup_loc_id']).size().reset_index()
    agg_rides.rename(columns={0:'rides'}, inplace=True)
    
    # Add Rows for (locations, pickup_hours) with 0 rides
    agg_rides_all_slots = addMissingSlots(agg_rides)
    
    return agg_rides_all_slots
     

def addMissingSlots(aggregrate_rides: pd.DataFrame) -> pd.DataFrame:

    location_ids = aggregrate_rides['pickup_loc_id'].unique()
    full_range = pd.date_range(
        aggregrate_rides['pickup_hour'].min(),
        aggregrate_rides['pickup_hour'].max(),
        freq='H' 
    )
    output = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # Keep rides for specific id
        aggregrate_rides_i = aggregrate_rides.loc[aggregrate_rides.pickup_loc_id == location_id, ['pickup_hour', 'rides']]
        
        
        aggregrate_rides_i.set_index('pickup_hour', inplace=True)
        aggregrate_rides_i.index = pd.DatetimeIndex(aggregrate_rides_i.index)
        aggregrate_rides_i = aggregrate_rides_i.reindex(full_range, fill_value=0)
        
        aggregrate_rides_i['pickup_loc_id'] = location_id
        
        output = pd.concat([output, aggregrate_rides_i])
    
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})
    
    return output

def getCutoffIndices(
    data: pd.DataFrame,
    n_features: int,
    step_size: int
) -> list:
    stop_pos = len(data) - 1
    
    subseq_first_idx = 0
    subseq_mid_idx = n_features
    subseq_last_idx = n_features + 1
    indices = []
    
    while subseq_last_idx <= stop_pos:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        
        subseq_first_idx    += step_size
        subseq_mid_idx      += step_size
        subseq_last_idx     += step_size
    
    return indices

def processData2FeatTgt(
    ts_data: pd.DataFrame,
    input_feat_len: int,
    step_size: int
    ) -> pd.DataFrame:
    n_features = input_feat_len
    
    
    loc_ids = ts_data['pickup_loc_id'].unique()
    feat_df = pd.DataFrame()
    tgt_df = pd.DataFrame()
    for loc_id in tqdm(loc_ids):     
        
        ts_data_one_loc = ts_data.loc[ts_data.pickup_loc_id == loc_id, :].reset_index(drop=True)

        indices = getCutoffIndices(
            ts_data_one_loc,
            n_features,
            step_size
        )

        n_examples = len(indices)

        x = np.ndarray(shape=(n_examples, n_features), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []

        for i,idx in enumerate(indices):
            x[i,:] = ts_data_one_loc.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_loc.iloc[idx[1]:idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_loc.iloc[idx[1]]['pickup_hour'])
        feat_1_loc = pd.DataFrame(
            x,
            columns=[f"rides_prev_{i+1}_hr" for i in reversed(range(n_features))]
        )
        feat_1_loc['pickup_hr'] = pickup_hours
        feat_1_loc['location_id'] = loc_id
        
        tgt_1_loc = pd.DataFrame(y, columns=[f"tgt_rides_nxt_hr"])
        
        feat_df = pd.concat([feat_df, feat_1_loc])
        tgt_df = pd.concat([tgt_df, tgt_1_loc])
        feat_df.reset_index(drop=True, inplace=True)
        tgt_df.reset_index(drop=True, inplace=True)
        
    return feat_df,tgt_df
        