"""
This module contains useful information about GTFS data format for use in altering and using gtfs data
"""
from __future__ import annotations

import json
import requests
import os

# using dot env to keep my API key private but if I messed up and it's actually visible please do not use!!!!
from dotenv import load_dotenv
load_dotenv("api_keys.env")
transit_land_api_key = os.getenv("TRANSIT_LAND_API_KEY")

# yes bad to use import *, but general_tools designed to be suuuuupper safe
from general_tools import *


# main code block
route_types:dict = {
    "0": "tram",
    "1": "metro",
    "2": "rail",
    "3": "bus",
    "4": "ferry",
    "5": "cable_car",
    "6": "gondola",
    "7": "funicular",
    "11": "trolleybus",
    "12": "monorail"
}
# in case want to get route type number based on common name for route type (e.g., "bus" -> "3")
route_types_reverse_lookup:dict = {value: key for key, value in route_types.items()}

class TransitAgency:
    """ 
    Transit agency class compatible with transit land API response. For use in public transit data model and for 
    representing transit stuff in python
    """
    def __init__(self, agency_email, agency_fare_url, agency_id, agency_lang, agency_name, agency_phone,
                 agency_timezone, agency_url, feed_version, geometry, id, onestop_id, operator, places):
        self.agency_email = agency_email
        self.agency_fare_url = agency_fare_url
        self.agency_id = agency_id
        self.agency_lang = agency_lang
        self.agency_name = agency_name
        self.agency_phone = agency_phone
        self.agency_timezone = agency_timezone
        self.agency_url = agency_url
        self.feed_version = feed_version
        self.geometry = geometry
        self.id = id
        self.onestop_id = onestop_id
        self.operator = operator
        self.places = places

class TransitVehicle:
    def __init__(self, vehicle_id:str, vehicle_type:str, max_speed_mph:float, avg_acceleration_rate:float,
                 avg_deceleration_rate:float=None):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.max_speed_mph = max_speed_mph
        self.avg_acceleration_rate = avg_acceleration_rate
        self.avg_deceleration_rate = avg_deceleration_rate

        # avg deceleration rate defaults to the same as avg acceleration rate
        if self.avg_deceleration_rate is None:
            self.avg_deceleration_rate = self.avg_acceleration_rate

    def calculate_acceleration_time(self):
        pass

    def calculate_braking_time(self):
        pass

    def calculate_acceleration_distance(self):
        pass

    def calculate_braking_distance(self):
        pass

    def calculate_spacing_travel_time(self, spacing_distance:float):
        pass

# need to figure out what attributes for transit route will be
class TransitRoute:
    def __init__(self, route_id:str, route_agency:str, route_type:str, transit_vehicles:list[TransitVehicle],
                 policy_standing_time_seconds:int, stations:list[TransitStation]):
        self.route_id = route_id
        self.route_agency = route_agency
        self.route_type = route_type
        self.transit_vehicles = transit_vehicles
        self.policy_standing_time_seconds = policy_standing_time_seconds
        self.stations = stations

class TransitStation:
    """
    Class for a transit station to be used in modifying GTFS data

    Attributes:
        station_id: str | the ID of the station
        station_name: str | the name of the station
        station_snake_name: str | the snake name for the station
        station_type: str | the type of the station (uses gtfs route types)
        station_latitude: float | the latitude of the station
        station_longitude: float | the longitude of the station

    """
    def __init__(self, station_id:str, station_name:str, station_type:str, station_routes: list[TransitRoute],
                 station_latitude:float, station_longitude:float):
        self.station_id = station_id
        self.station_name = station_name
        self.station_type = station_type
        self.station_routes = station_routes
        self.station_latitude = station_latitude
        self.station_longitude = station_longitude
        self.station_snake_name = create_snake_name(self.station_name)

        # initial error handling
        if len(self.station_routes) < 1:
            raise ValueError("Station must be served by at least one route")
        if station_type not in route_types:
            raise ValueError("Station type must be a valid route type as defined in the GTFS standard")

        # other atrributes
        self.neighboring_stations:dict[TransitRoute, list[TransitStation]] = {}

    def get_neighboring_stations(self):
        """
        Goes through all the routes that serve the station and returns a dictionary containing the route as the key
        and the next stations in both directions as the value
        """

        # iterate through all routes that serve station
        for route in self.station_routes:
            # double check that the station is actually served by the route
            if self not in route:
                raise Exception(f"Station {self.station_name} is not served by route {route}")

            # two cases to consider: where station has neighbors in both directions, or just one
            station_position_in_route = 0
            for idx, station in enumerate(route.stations):
                if station == self:
                    station_position_in_route = idx

            # in case where station is start point for this route
            if station_position_in_route == 0:
                self.neighboring_stations[route] = [route.stations[1], None]
            # in case where station is end point for this route
            elif station_position_in_route == len(route.stations) - 1:
                self.neighboring_stations[route] = [None, route.stations[-2]]
            # in case where station has two neighbors
            else:
                self.neighboring_stations[route] = [route.stations[station_position_in_route + 1],
                                                    route.stations[station_position_in_route - 1]]

            return self.neighboring_stations
