import zipfile

from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd

# plotting libs
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    loadFeatBatchFromStore,
    loadPredsFromStore,
)

from src.paths import DATA_DIR
from src.plot import plotSample

st.set_page_config (layout="wide")

# Title
current_date = pd.to_datetime(datetime.utcnow()).floor('H')
st.title(f"Taxi demand prediction")
st.header(f"{current_date} UTC")


progress_bar = st.sidebar.header("Work in progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 6

def loadShapeDataFile():
    
    # Get file
    URL = f"https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    response = requests.get(URL)
    path = DATA_DIR / f"taxi_zones.zip"
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f"{URL} is not available")
    
    # Unzip file
    with zipfile.ZipFile(path, "r") as zipf:
        zipf.extractall(DATA_DIR / "taxi_zones")
    
    # Load and return shape file
    return gpd.read_file(DATA_DIR / "taxi_zones/taxi_zones.shp").to_crs("epsg:4326")

@st.cache_data
def _LoadBatchOfFeatsFromStore(current_date: datetime) -> pd.DataFrame:
    """
    Wrapped version of load Feature  batch for streamlit caching

    Args:
        current_date (datetime): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    return loadFeatBatchFromStore(current_date)

@st.cache_data
def _LoadPredsFromStore(
        from_pickup_hr: datetime,
        to_pickup_hr: datetime,
    ) -> pd.DataFrame:
    """
    Wrapped version of load predictions from store for streamlit caching

    Args:
        from_pickup_hr (datetime): _description_
        to_pickup_hr (datetime): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return loadPredsFromStore(from_pickup_hr, to_pickup_hr)

with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = loadShapeDataFile()
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1/N_STEPS)
    
with st.spinner(text="Fetching model predictions from the store"):
    preds_df = _LoadPredsFromStore(
        from_pickup_hr=current_date - timedelta(hours=1),
        to_pickup_hr=current_date
    )
    st.sidebar.write("Model predictions have arrived")
    progress_bar.progress(2/N_STEPS)
    
next_hr_preds_ready = False if preds_df[preds_df.pickup_hr == current_date].empty else True
prev_hr_preds_ready = False if preds_df[preds_df.pickup_hr == (current_date - timedelta(hours=1))].empty else True
                    
if next_hr_preds_ready:
    preds_df = preds_df[preds_df.pickup_hr == current_date]                    
elif:
    preds_df = preds_df[preds_df.pickup_hr == (current_date - timedelta(hours=1))]
    current_date = current_date - timedelta(hours=1)
    st.subheader('Most recent data is not yet available. Using last hour predictions')
else:
    raise Exception('Features are not available for last two hours. Error with feature pipeline')

with st.spinner(text="Preparing data to plot"):
    
    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        f = float(val - minval) / (maxval - minval)
        return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
    
    df = pd.merge(geo_df, preds_df, right_on="pickup_loc_id", left_on="LocationID") 
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    progress_bar.progress(3/N_STEPS)
    
with st.spinner(text="Generating NYC Map"):
    
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=-73.9712,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )
    
    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlights=True,
        pickable=True,
    )
    
    tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b>Predicted rides:</b>{predicted_demand}"}
    
    r=pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )
    
    st.pydeck_chart(r)
    progress_bar.progress(4/N_STEPS)
    
with st.spinner(text="Fetching bathc of features used in the last run"):
    feats_df = _LoadBatchOfFeatsFromStore(current_date)
    st.sidebar.write('Inference feratures fetched from the store')
    progress_bar.progress(5/N_STEPS)
    
    
with st.spinner(text="Plotting time-series data"):
    row_indices = np.argsort(preds_df['predicted_demand'].values)[::-1]
    n_to_plot = 10
    
    # Plot each time-series with the prediction
    for row_id in row_indices[:n_to_plot]:
        fig = plotSample(
            features=preds_df,
            eg_id=row_id,
            # targets=results["predicted_demand"],
            predictions=pd.Series(preds_df["predicted_demand"])
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=100)
        
    progress_bar.progress(6/N_STEPS)
    
        