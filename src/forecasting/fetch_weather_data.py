
#!pip install openmeteo-requests requests-cache retry-requests pandas
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry


print("1. Configuring Open-Meteo API client...")
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"

# THE FIX: Start fetching from February 28th to create a "padding buffer"
# This ensures we have data prior to March 1st for the timezone shift and interpolation.
params = {
    "latitude": 51.2277,
    "longitude": 6.7735,
    "start_date": "2024-02-28", 
    "end_date": "2026-03-01",
    "hourly": "temperature_2m",
    "timezone": "UTC"
}

print(f"2. Fetching padded temperature data from {params['start_date']} to {params['end_date']}...")
responses = openmeteo.weather_api(url, params=params)
hourly = responses[0].Hourly()

# Create the raw dataframe in UTC
hourly_data = {
    "Time Point": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "Temperature": hourly.Variables(0).ValuesAsNumpy()
}
df_weather = pd.DataFrame(data=hourly_data)

print("3. Aligning timezones and interpolating over boundaries...")
# Convert to fixed UTC+1 (+01:00) to match your heat demand target
df_weather['Time Point'] = pd.to_datetime(df_weather['Time Point']).dt.tz_convert('+01:00')
df_weather.set_index('Time Point', inplace=True)

# Interpolate NOW, while we still have the February padding data to anchor the line
df_weather['Temperature'] = df_weather['Temperature'].interpolate(method='linear')

print("4. Enforcing strict target grid (Trimming the padding)...")
# Generate the exact requested grid
target_grid = pd.date_range(
    start='2024-03-01 00:00:00', 
    end='2026-03-01 23:00:00', 
    freq='h', 
    tz='+01:00',
    name='Time Point'
)

# Reindex to force the dataframe to match this grid exactly (drops the Feb padding)
df_weather = df_weather.reindex(target_grid)

# Final safety net for any internal API gaps
df_weather['Temperature'] = df_weather['Temperature'].interpolate(method='linear').bfill()

# Convert back to standard columns and format to strict string layout
df_weather = df_weather.reset_index()
df_weather['Time Point'] = df_weather['Time Point'].dt.strftime('%Y-%m-%dT%H:%M:%S.000000%z')

# Save to CSV
output_csv = 'RawData_ExternalTemperature.csv'
df_weather.to_csv(output_csv, index=False)

print(f"\nSuccess! Data formatted and saved to '{output_csv}'.")
print("Preview of the flawlessly formatted data (Notice 00:00 is now filled!):")
print(df_weather.head())

