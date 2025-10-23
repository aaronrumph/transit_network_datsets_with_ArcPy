"""
This module is used to do all the GTFS nitty gritty work
"""
# TODO: add docstrings for every class, method, and function so that can actually publish as package
    # and have be understandable. Also adding better exception handling.

# TODO: add methods needed for specific analysis

import os

# Setting up python environment/making sure env points to extensions correctly
arcgis_bin = r"C:\Program Files\ArcGIS\Pro\bin"
arcgis_extensions = r"C:\Program Files\ArcGIS\Pro\bin\Extensions"
os.environ["PATH"] = arcgis_bin + os.pathsep + arcgis_extensions + os.pathsep + os.environ.get("PATH", "")

# if you want to add modules, MUST (!!!!!!!) come after this block (ensures extensions work in cloned environment)
import arcpy
import arcpy_init
#

# main code section
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
                 policy_standing_time_seconds:int):
        self.route_id = route_id
        self.route_agency = route_agency
        self.route_type = route_type
        self.transit_vehicles = transit_vehicles
        self.policy_standing_time_seconds = policy_standing_time_seconds

class TransitDataFeatures:
    def __init__(self, name:str):
        """
        Parent class for the transit data features needed to use GTFS with ArcGIS
        :param name: str | name of the feature
        """
        self.name = name
        self.place = None
        self.agencies = []
        self.attribute_table = None
        self.path = None
        self.feature_dataset = None
        self.gdb = None
        self.arc_project = None

    def check_if_exists(self):
        if arcpy.Exists(self.path):
            return True
        else:
            return False

    def delete_feature(self):
        if self.check_if_exists():
            arcpy.Delete_management(self.path)
        else:
            raise ValueError(f"Feature {self.path} does not exist")

class TransitStops(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("Stops")
        self.agencies = agencies

    def remove_stops(self, stops_to_remove:list):
        """
        removes the stops from the list of stops
        :param stops_to_remove: list of stops to remove                                                                  # need to figure out naming convention for stops
        :return:
        """
        pass

class LineVariantElements(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("LineVariantElements")
        self.agencies = agencies

class LVEShapes(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("LVEShapes")
        self.agencies = agencies

class Calendars(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("Calendars")
        self.agencies = agencies

class CalendarExceptions(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("CalendarExceptions")
        self.agencies = agencies

class Lines(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("Lines")
        self.agencies = agencies

class LineVariants(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("LineVariants")
        self.agencies = agencies

class Runs(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("Runs")
        self.agencies = agencies

class ScheduleElements(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("ScheduleElements")
        self.agencies = agencies

class Schedules(TransitDataFeatures):
    def __init__(self, agencies:list):
        super().__init__("Schedules")
        self.agencies = agencies


