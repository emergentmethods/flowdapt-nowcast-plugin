
import pandas as pd
import geopy.distance as gd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Tuple
from openmeteo_py import Hourly, Options, OWmanager  # Daily,

from flowdapt.lib.logger import get_logger

logger = get_logger(__name__)


def get_neighbor_grid(cities_df, cities, m, r):
    """
    Gets a grid of m x m points around each city in cities
    """

    if isinstance(cities, list):
        top_cities = cities_df.head(100)
        main_cities = top_cities[top_cities['city'].isin(cities)]
    elif isinstance(cities, int):
        main_cities = cities_df.head(cities)

    cities_grid = pd.DataFrame(columns=['lat', 'lng', 'lat_km', 'lng_km',
                                        'lat_min', 'lat_max', 'lng_min', 'lng_max',
                                        'lat_grid_km', 'lng_grid_km', 'lat_grid', 'lng_grid',
                                        'grid'])

    cities_grid['city'] = main_cities['city']
    cities_grid['lat'] = main_cities['lat']
    cities_grid['lng'] = main_cities['lng']

    # convert "lat" and "lng" to kilometers
    cities_grid['lat_km'] = (main_cities['lat'] * 111.32).round(4)
    cities_grid['lng_km'] = (main_cities['lng'] * 111.32).round(4)

    # create a grid from `radius` around each city
    cities_grid['lat_min'] = cities_grid['lat_km'] - r
    cities_grid['lat_max'] = cities_grid['lat_km'] + r
    cities_grid['lng_min'] = cities_grid['lng_km'] - r
    cities_grid['lng_max'] = cities_grid['lng_km'] + r

    # create a grid from lat_min and lng_min to lat_max and lng_max with m points
    cities_grid['lat_grid_km'] = cities_grid.apply(
        lambda x: np.linspace(x['lat_min'], x['lat_max'], m), axis=1)
    cities_grid['lng_grid_km'] = cities_grid.apply(
        lambda x: np.linspace(x['lng_min'], x['lng_max'], m), axis=1)

    # convert lat_grid_km and lng_grid_km to lat and lng
    cities_grid['lat_grid'] = cities_grid['lat_grid_km'].apply(
        lambda x: (x / 111.32).round(4))
    cities_grid['lng_grid'] = cities_grid['lng_grid_km'].apply(
        lambda x: (x / 111.32).round(4))

    # convert lat_grid and lng_grid to a list of tuples
    cities_grid['grid'] = cities_grid.apply(
        lambda x: np.meshgrid(x['lat_grid'], x['lng_grid']), axis=1)

    # convert "grid" to a list of coordinates
    cities_grid['grid'] = cities_grid['grid'].apply(
        lambda x: np.array([x[0].flatten(), x[1].flatten()]).T)

    column_lbls = ['city', 'lat', 'lng', "state_id", "state_name", "population",
                   "density", "timezone", "neighbor_num", "city_group", "neighbor_distance"]
    neighbors = pd.DataFrame(columns=column_lbls)
    for j, city in enumerate(cities_grid['city']):
        city_df = main_cities[main_cities['city'] == city][column_lbls[:-3]]
        city_df["neighbor_num"] = 0
        city_df["city_group"] = j
        city_df["neighbor_distance"] = 0
        filtered_city_df = city_df.dropna(axis=1, how='all')
        neighbors = pd.concat(
            [neighbors, filtered_city_df],
            axis=0
        )
        coords_1 = main_cities[main_cities['city'] ==
                               city][['lat', 'lng']].values[0]
        df = pd.DataFrame(columns=column_lbls)
        for coords in cities_grid[cities_grid['city'] == city]['grid']:
            for i, coord in enumerate(coords):
                coords_2 = np.array([coord[0], coord[1]])
                distance = gd.geodesic(coords_1, coords_2).km
                if distance > 0:
                    df.loc[0, "city"] = f"{city}_n_r{i // m}_c{i % m}"
                    df["lat"] = coords_2[0]
                    df["lng"] = coords_2[1]
                    df["neighbor_num"] = neighbors["neighbor_num"].iloc[-1] + 1
                    df["city_group"] = j
                    df["neighbor_distance"] = distance
                    filtered_df = df.dropna(axis=1, how='all')
                    neighbors = pd.concat([neighbors, filtered_df], axis=0)
    neighbors = neighbors.reset_index(drop=True)

    return neighbors


def get_city_grid(selected_cities: list, neighbors: int, radius: float, path: str) -> pd.DataFrame:

    cities = pd.read_csv(path)
    # remove non continental US cities
    cities = cities[~cities["state_name"].isin(["Puerto Rico", "Alaska", "Hawaii"])]
    # get neighbors for N random cities, with M neighbors each
    sub_df = get_neighbor_grid(cities, cities=selected_cities, m=neighbors, r=radius)

    return sub_df

def get_start_end_dates(num_days: int) -> Tuple[str, str]:
    """
    Given number of days, give start and end dates
    for openmeteo api
    """
    end_date = datetime.now(tz=timezone.utc) + timedelta(days=4)
    start_date = datetime.now(tz=timezone.utc) - timedelta(days=num_days)

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def download_data(md: dict, n_bars: int = 5) -> pd.DataFrame:

    days = int(np.ceil(n_bars / 24))
    start_date, end_date = get_start_end_dates(days)
    logger.info(f"Getting data from {start_date} to {end_date} for days {days}")
    hourly = Hourly()
    hourly.all()
    hourly.temperature_2m()
    params = [
        "et0_fao_evapotranspiration", "vapor_pressure_deficit", "cape",
        "shortwave_radiation_instant", "direct_radiation_instant",
        "diffuse_radiation_instant", "direct_normal_irradiance_instant",
        "terrestrial_radiation_instant"
    ]
    hourly.hourly_params += params
    # pop "pressure_msl" from hourly params
    hourly.hourly_params = [x for x in hourly.hourly_params if x not in [
        "pressure_msl", "evapotranspiration"]]

    # daily = Daily()
    latitude = md["latitude"]
    longitude = md["longitude"]
    params = {
        "current_weather": True, "past_days": 0,
        "latitude": latitude, "longitude": longitude
    }
    options = Options(**params)

    mgr = MyOWmanager(
        options=options,
        hourly=hourly,
        daily=None,  # daily.all(),
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(f"Getting weather data for {md['city']}")
    df = mgr.get_data(3)
    df = df.rename(columns={'time': 'date'})

    # slice off any future forecast values coming from openmeteo
    now = np.datetime64(datetime.now(tz=timezone.utc))
    df["date"] = pd.to_datetime(df['date'])
    df_hist = df.loc[df['date'] <= now]
    df_fc = df.loc[df['date'] > now]
    return df_hist.copy(), df_fc.copy()

def process_forecast_data(df: pd.DataFrame, now: np.datetime64) -> pd.DataFrame:
    """
    Given a dataframe of rows as dates in 1 hour intervals
    extract the forecasts for 1h, 4h, 8h, 12h, 24h, 36h and
    add them as columns to the current date
    """
    # define a list of time intervals for which to extract the forecasts
    target_base = [
        "cloudcover",
        "windspeed_10m",
        "temperature_2m"
    ]

    time_intervals = [0, 1, 2, 3, 4, 5]
    target_list = []
    # loop over the time intervals and add the corresponding forecast columns to the DataFrame
    for shift in time_intervals:
        df_shift = df[target_base].shift(-shift)
        df_shift = df_shift.add_suffix(f"-{shift+1}hr")
        df_shift = df_shift.add_prefix("&-")
        target_list.append(df_shift.columns)
        df = pd.concat([df, df_shift], axis=1)

    fts = df.loc[:, df.columns.str.startswith('&-')]
    df = pd.concat([df['date'], fts], axis=1)

    # ensure the dates are forecasts made on the current hour
    df['date'] -= timedelta(hours=1)

    df = df.rename(columns={'date': 'dates'})
    df = df.iloc[0:1, :].reset_index(drop=True)
    return df

class MyOWmanager(OWmanager):
    """
    Inherit from openmeteo_py so that we can add the start and end
    date to the payload
    """

    def __init__(self, options, hourly=None, daily=None, start_date: str = "", end_date: str = ""):
        super().__init__(options, hourly=hourly, daily=daily)
        self.extra_payload = {}
        self.extra_payload["start_date"] = start_date
        self.extra_payload["end_date"] = end_date
        self.extra_payload["cell_selection"] = "nearest"
        self.extra_payload["models"] = "gfs_seamless"
        extra_payload = "&".join("%s=%s" % (k, v)
                                 for k, v in self.extra_payload.items())
        self.payload = f"{self.payload}&{extra_payload}"  # type: ignore
