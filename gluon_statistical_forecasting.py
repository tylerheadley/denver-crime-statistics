import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import numpy as np
from gluonts.model.predictor import Predictor
from gluonts.dataset.pandas import PandasDataset
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.field_names import FieldName
from typing import List

from custom_forecasters import (
    SampleDistributionForecaster,
    PredictLastForecaster,
    PredictMeanForecaster,
    PredictMedianForecaster,
    PredictQuantileForecaster,
    SimpleMovingAverage,
    ExponentiallyWeightedMovingAverage,
    CustomForecaster
)

custom_forecaster = CustomForecaster(
    freq="1H",
    forecast_model=SampleDistributionForecaster,
    prediction_length=24
)

df= pd.read_csv('data/resampled_crime_data.csv', parse_dates=True)

# Define the cutoff timestamp
cutoff_timestamp = pd.to_datetime('2019-01-23 00:00:00')

# Convert 'timestamp' column to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter the DataFrame
filtered_df = df[df['timestamp'] < cutoff_timestamp]

filtered_df['target'] = filtered_df['target'].astype(np.float32)
filtered_df.head()

filtered_df = filtered_df.set_index('timestamp')

from gluonts.dataset.pandas import PandasDataset

dataset = PandasDataset.from_long_dataframe(filtered_df, target="target", item_id="item_id")

forecast_it = custom_forecaster.predict(dataset)

# Convert forecasts to a pandas DataFrame for plotting or further analysis
forecasts_df = []

for forecast in forecast_it:
    forecast_entry = pd.DataFrame({
        "timestamp": pd.date_range(start=forecast.start_date, periods=custom_forecaster.prediction_length, freq="1H"),
        "prediction": forecast.samples.mean(axis=0),
        "item_id": forecast.item_id
    })
    forecasts_df.append(forecast_entry)

# Concatenate all forecasts for different items
all_forecasts_df = pd.concat(forecasts_df)

# Plot the forecasts
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for item_id in all_forecasts_df['item_id'].unique():
    item_forecast = all_forecasts_df[all_forecasts_df['item_id'] == item_id]
    plt.plot(item_forecast['timestamp'], item_forecast['prediction'], label=f'Item {item_id}')

plt.xlabel('Timestamp')
plt.ylabel('Prediction')
plt.title('Time Series Forecasts')
plt.legend()
plt.show()
