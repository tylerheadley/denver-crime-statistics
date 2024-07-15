import numpy as np
import pandas as pd
import datetime

df = pd.read_csv("data/cleaned_data/crime.csv", encoding='latin-1')

df['first_occurrence_date'] = pd.to_datetime(df['first_occurrence_date']) #, unit='ms')
df['last_occurrence_date'] = pd.to_datetime(df['last_occurrence_date']) # , unit='ms')

# create a numpy tensor |Time bins| x |Neighborhoods| x |Crime types|

# time_resolution is a tunable parameter represented in hours
min_datetime = df['first_occurrence_date'].min()
max_datetime = df['first_occurrence_date'].max()

print("Time range:", min_datetime, "-", max_datetime)

time_resolution = input("Enter the temporal resolution ('hour', 'day', 'week', 'month', 'year', or an integer in hours): ")
output_filename = input("Choose output filename (ending in .npz): ")

# Calculate the total duration
total_duration = max_datetime - min_datetime
duration_in_hours = total_duration.total_seconds() // 3600

# Convert total duration to the desired resolution
if time_resolution == 'hour':
    bin_size_hours = 1
elif time_resolution == 'day':
    bin_size_hours = 24
elif time_resolution == 'week':
    bin_size_hours = 168
elif time_resolution == 'month':
    bin_size_hours = 365 * 24 / 12
elif time_resolution == 'year':
    bin_size_hours = 365 * 24
else:
    try:
        bin_size_hours = int(time_resolution)
    except TypeError:
        raise ValueError("Unsupported resolution. Use 'hour', 'day', 'week', 'month', or 'year'.")

# Calculate the number of bins
num_time_bins = int(duration_in_hours / bin_size_hours) + 1

print("Number of time bins:", num_time_bins)


neighborhoods = sorted(df['neighborhood_id'].unique())
num_neighborhoods = len(neighborhoods)

# Create a dictionary mapping each unique crime to an integer index
neighborhood_indexes = {}
current_index = 0
for neighborhood in neighborhoods:
    neighborhood_indexes[neighborhood] = current_index
    current_index += 1

print("Crime type indexes:", neighborhood_indexes)

print("Number of neighborhoods (nodes):", num_neighborhoods)


crime_types = df['offense_code'].unique()
num_crime_types = len(crime_types)

print("Number of crime types:", num_crime_types)

# Create a dictionary mapping each unique crime to an integer index
crime_type_indexes = {}
current_index = 0
for crime_type in crime_types:
    crime_type_indexes[crime_type] = current_index
    current_index += 1

print("Crime type indexes:", crime_type_indexes)

# Initialize the 4D tensor
tensor = np.zeros((num_time_bins, num_neighborhoods, num_crime_types), dtype=int)

print("Tensor created with dimensions:", tensor.shape)

# Populate the tensor
for _, row in df.iterrows():
    time_bin = int((row['first_occurrence_date'] - min_datetime).total_seconds() // (bin_size_hours * 3600))
    neighborhood = neighborhood_indexes[row['neighborhood_id']]
    crime_type = crime_type_indexes[row['offense_code']]
    tensor[time_bin, neighborhood, crime_type] += 1

print("Tensor entries populated with crime counts.")

# Save the tensor as a compressed .npz file
np.savez_compressed('data/tensors/' + output_filename, tensor=tensor)
