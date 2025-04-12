import random

import numpy as np

def rand_action():
    '''
    Return random action in np.array
    '''
    r1 = random.randint(0, 1)
    r2 = random.randint(0, 1)
    orien = np.array([r1, 1 - r1]) 
    symb = 1 if r2 == 1 else -1
    return orien * symb

# [-120, -90] -> [0, 1]
def rssi_normalize(r):
    """
    Normalize rssi from -90 to -120
    """
    if r == -200:
        return -1
    else:
        return min(1, max(0, (r + 120) / 30))
def lon_lat_to_xy(lon, lat, origin_lon, origin_lat):
    """
    Roughly simulate the conversion of longitude and latitude to x and y coordinates (in meters) in a 2D plane based on the origin point

    lon: longitude of the point
    lat: latitude of the point
    origin_lon: longitude of the origin point
    origin_lat: latitude of the origin point

    return: the x and y coordinates of the point in meters
    """
    local_x = (lon - origin_lon) * 111139 * np.cos(np.radians(origin_lat))
    local_y = (lat - origin_lat) * 111139
    return local_x, local_y

def xy_to_lon_lat(x, y, origin_lon, origin_lat):
    """
    Roughly simulate the conversion of x and y coordinates (in meters) to longitude and latitude in a 2D plane based on the origin point

    x: x coordinate of the point in meters
    y: y coordinate of the point in meters
    origin_lon: longitude of the origin point
    origin_lat: latitude of the origin point

    return: the longitude and latitude of the point
    """
    lon = x / (111139 * np.cos(np.radians(origin_lat))) + origin_lon
    lat = y / 111139 + origin_lat
    return lat, lon

def get_lon_lat_limits(origin_lon, origin_lat, radius):
    """
    Get the minimum and maximum longitude and latitude based on the origin point and the search radius

    origin_lon: longitude of the origin point
    origin_lat: latitude of the origin point
    radius: search radius in meters

    return: the minimum and maximum longitude and latitude
    """
    return {
        'min_lon': origin_lon - radius / (111139 * np.cos(np.radians(origin_lat))),
        'max_lon': origin_lon + radius / (111139 * np.cos(np.radians(origin_lat))),
        'min_lat': origin_lat - radius / 111139,
        'max_lat': origin_lat + radius / 111139
    }