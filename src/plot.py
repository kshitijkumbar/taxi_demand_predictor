from typing import Optional
from  datetime import timedelta
import pandas as pd
import plotly.express as px

def plotSample(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    eg_id: int,
    predictions: Optional[pd.Series] =None
):
    """"""
    feats = features.iloc[eg_id] 
    tgt = targets.iloc[eg_id] 

    ts_columns = [col for col in features.columns if col.startswith('rides_prev_')]
    ts_values = [feats[col] for col in ts_columns] + [tgt]
    
    ts_dates = pd.date_range(
        feats['pickup_hr'] - timedelta(hours=len(ts_columns)),
        feats['pickup_hr'],
        freq='H'
    )
    
    # Plot for past values
    title = f"Pick up hour = {feats['pickup_hr']},\
            location_id = {feats['pickup_loc_id']}"
    
    fig = px.line(
        x=ts_dates,
        y=ts_values,
        template='plotly_dark',
        markers=True,
        title=title
    )
    
    # Plotting target values
    
    fig.add_scatter(
        x=ts_dates[-1:],
        y=[tgt],
        line_color='green',
        mode='markers',
        marker_size=10,
        name='actual value' 
    )
    
    # Plot predictions if available
    if predictions is not None:
        preds = predictions.iloc[eg_id]
        fig.add_scatter(
            x=ts_dates[-1:],
            y=[preds],
            line_color='red',
            mode='markers',
            marker_symbol='x',
            marker_size=15,
            name='prediction'
        )
    return fig
        