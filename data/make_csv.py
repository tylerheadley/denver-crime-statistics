import pandas as pd

# Step 1: Read the CSV file into a Pandas DataFrame
df = pd.read_csv('data/cleaned_data/crime.csv', encoding='latin-1')

# Step 2: Convert the 'first_occurrence_date' column to a datetime format
df['first_occurrence_date'] = pd.to_datetime(df['first_occurrence_date'])

# Step 3: Determine the complete time range
start_time = df['first_occurrence_date'].min()
end_time = df['first_occurrence_date'].max()
time_range = pd.date_range(start=start_time, end=end_time, freq='h')

# Step 4: Initialize the new DataFrame for the resampled data
resampled_data = []

# Loop through each neighborhood and perform hourly resampling
neighborhood_ids = df['neighborhood_id'].unique()

for neighborhood_id in neighborhood_ids:
    # Filter the DataFrame for the current neighborhood_id
    neighborhood_df = df[df['neighborhood_id'] == neighborhood_id]

    # Set the datetime column as the DataFrame index
    neighborhood_df.set_index('first_occurrence_date', inplace=True)

    # Perform the hourly resampling and reindex to the complete time range
    resampled_neighborhood_df = neighborhood_df.resample('h').size().reindex(time_range, fill_value=0).reset_index(name='target')

    # Add the neighborhood_id as item_id
    resampled_neighborhood_df['item_id'] = neighborhood_id

    # Append to the list of resampled data
    resampled_data.append(resampled_neighborhood_df)

# Step 5: Concatenate all the resampled data
final_df = pd.concat(resampled_data)

# Rename the index column to 'timestamp'
final_df.rename(columns={'index': 'timestamp'}, inplace=True)

# Display the final DataFrame
print(final_df)

# Save the DataFrame to a CSV file if needed
final_df.to_csv('data/resampled_crime_data.csv', index=False)
