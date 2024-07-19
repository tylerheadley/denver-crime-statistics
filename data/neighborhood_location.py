import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import geopandas as gpd


df = pd.read_csv("data/cleaned_data/crime.csv", encoding='latin-1', parse_dates=True)

means = df.groupby('neighborhood_id').agg({'geo_lon': 'median', 'geo_lat': 'median'}).reset_index()

neighborhoods = pd.read_csv("data/cleaned_data/neighborhoods.csv")
neighborhoods.rename(columns={'NBHD_NAME': 'neighborhood_id'}, inplace=True)
df_merged = neighborhoods.merge(means, on='neighborhood_id', how='left')

df_merged.drop(columns=['NBHD_ID', 'TYPOLOGY', 'NOTES'], inplace=True)
print(df_merged.head())
df_merged.to_csv('data/nbhd_centers.csv', index=False)

# Convert the the_geom column to a GeoSeries
df_merged['geometry'] = gpd.GeoSeries.from_wkt(df_merged['the_geom'])

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot the neighborhoods
gdf.plot(ax=ax, color='lightblue', edgecolor='black')

# Plot the centroids
ax.scatter(df_merged['geo_lon'], df_merged['geo_lat'], color='red', marker='o')

# Add labels for each neighborhood
for idx, row in df_merged.iterrows():
    plt.text(row['geo_lon'], row['geo_lat'], row['neighborhood_id'], fontsize=6, ha='center')

# Add titles and labels
plt.title('Map of Neighborhoods with Centroids')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Show the plot
plt.show()

exit()

# Create a scatter plot of the mean coordinates
plt.figure(figsize=(8, 6))
plt.scatter(means['geo_lon'], means['geo_lat'], color='blue', marker='o')

# Add labels for each point
for i, row in means.iterrows():
    plt.text(row['geo_lon'], row['geo_lat'], row['neighborhood_id'], fontsize=6, ha='right')

# Add titles and labels
plt.title('Scatter Plot of Mean Geo Coordinates')
plt.xlabel('Mean Geo_X')
plt.ylabel('Mean Geo_Y')
plt.grid(True)

# Show the plot
plt.show()
