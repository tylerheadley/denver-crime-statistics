import pandas as pd
import datetime

#########################
# load data into pandas #
#########################

crime_df = pd.read_csv('data/raw_data/crime.csv', encoding='latin-1')
offense_codes_df = pd.read_csv('data/raw_data/offense_codes.csv', encoding='latin-1')
neighborhoods_df = pd.read_csv('data/raw_data/denver_statistical_neighborhoods_20240711.csv', delimiter=';')

#################
# data cleaning #
#################

crime_df.columns = crime_df.columns.str.lower()

crime_df['first_occurrence_date'] = pd.to_datetime(crime_df['first_occurrence_date']) #, unit='ms')
crime_df['last_occurrence_date'] = pd.to_datetime(crime_df['last_occurrence_date']) # , unit='ms')

# Replace spaces with dashes and capitalize all words
crime_df['neighborhood_id'] = crime_df['neighborhood_id'].str.replace('-', ' ').str.title()

# Replace 'Cbd' with 'CBD' in neighborhood_id column
crime_df['neighborhood_id'] = crime_df['neighborhood_id'].replace('Cbd', 'CBD')
crime_df['neighborhood_id'] = crime_df['neighborhood_id'].replace('Dia', 'DIA')
neighborhoods_df['NBHD_NAME'] = neighborhoods_df['NBHD_NAME'].str.replace('Stapleton', 'Central Park')

neighborhoods_df['NBHD_NAME'] = neighborhoods_df['NBHD_NAME'].str.replace('- ', '')

nan_count = crime_df['neighborhood_id'].isna().sum()
print("Number of NaN values before dropping:", nan_count)

# Drop rows with NaN values in neighborhood_id column
crime_df.dropna(subset=['neighborhood_id'], inplace=True)

# Count NaN values again to verify
nan_count_after_drop = crime_df['neighborhood_id'].isna().sum()
print("Number of NaN values after dropping:", nan_count_after_drop)

crime_df.sort_values(by='neighborhood_id')
neighborhoods_df.sort_values(by='NBHD_NAME')

#####################
# write data to csv #
#####################

crime_df.to_csv("data/cleaned_data/crime.csv")
offense_codes_df.to_csv("data/cleaned_data/offense_codes.csv") 
neighborhoods_df.to_csv("data/cleaned_data/neighborhoods.csv")

