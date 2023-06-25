import zipfile

from datetime import datetime

import requests
import numpy as np
import pandas as pd

# plotting libs
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    loadFeatBatchFromStore,
    loadModelFromRegistry,
    getModelPredictions
)

from src.paths import DATA_DIR
from src.plot import plotSample

st.set_page_config (layout="wide")

# Title
current_date = pd.to_datetime(datetime.utcnow()).floor('H')
st.title(f"Taxi demand prediction")
st.header(f"{current_date}")


progress_bar = st.sidebar.header("Work in progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 7 

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

with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = loadShapeDataFile()
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1/N_STEPS)
    
with st.spinner(text="Fetching batch of inference data"):
    features = loadFeatBatchFromStore(current_date)
    st.sidebar.write("Inference features fetched from store")
    progress_bar.progress(2/N_STEPS)
    print(f"{features}")
    
with st.spinner(text="Loading ML model from the registry"):
    model = loadModelFromRegistry()
    st.sidebar.write("ML model was loaded from the registry")
    progress_bar.progress(3/N_STEPS)


with st.spinner(text="Preparing data to plot"):
    results = getModelPredictions(model, features)
    st.sidebar.write("Model predictions are here")
    progress_bar.progress(4/N_STEPS)

with st.spinner(text="Preparing data to plot"):
    
    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        f = float(val - minval) / (maxval - minval)
        return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
    
    df = pd.merge(geo_df, results, right_on="pickup_loc_id", left_on="LocationID") 
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    progress_bar.progress(5/N_STEPS)
    
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
    progress_bar.progress(6/N_STEPS)
    
with st.spinner(text="Plotting time-series data"):
    row_indices = np.argsort(results['predicted_demand'].values)[::-1]
    n_to_plot = 10
    
    # Plot each time-series with the prediction
    for row_id in row_indices[:n_to_plot]:
        fig = plotSample(
            features=features,
            eg_id=row_id,
            # targets=results["predicted_demand"],
            predictions=pd.Series(results["predicted_demand"])
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=100)
        
    progress_bar.progress(7/N_STEPS)
    
        