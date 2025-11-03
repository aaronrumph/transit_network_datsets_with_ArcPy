"""
This module is used to do all the Public Transit Data Model nitty gritty work
"""
# standard library imports
from __future__ import annotations
import logging
import os
import re

# local imports
from gtfs_tools import TransitAgency
from gtfs_tools import TransitStation


# Setting up python environment/making sure env points to extensions correctly
arcgis_bin = r"C:\Program Files\ArcGIS\Pro\bin"
arcgis_extensions = r"C:\Program Files\ArcGIS\Pro\bin\Extensions"
os.environ["PATH"] = arcgis_bin + os.pathsep + arcgis_extensions + os.pathsep + os.environ.get("PATH", "")

# if you want to add modules, MUST (!!!!!!!) come after this block (ensures extensions work in cloned environment)
import arcpy
import arcpy_init
#

# type checking using TYPE_CHECKING because otherwise run into problems with circular import for\
# create_network_dataset_oop and this module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from create_network_dataset_oop import FeatureDataset

from general_tools import create_snake_name

# main code section


class TransitDataFeature:
    def __init__(self, feature_name:str, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        """
        Parent class for the transit data features needed to use GTFS with ArcGIS
        :param name: str | name of the feature
        """
        from create_network_dataset_oop import FeatureDataset

        self.feature_name = feature_name
        self.feature_dataset = feature_dataset
        self.place_name = place_name
        self.agencies = agencies
        self.attribute_table = None
        self.path = os.path.join(self.feature_dataset.path, self.feature_name)

        # these attributes are implied by the feature dataset provided
        self.gdb = self.feature_dataset.gdb
        self.arc_project = self.gdb.project

        # double checking that the feature dataset provided actually exists
        if not arcpy.Exists(self.feature_dataset.path):
            raise Exception("Feature dataset provided does not exist")

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



class Stops(TransitDataFeature):
    """ Stops class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__(place_name,"Stops", feature_dataset, agencies)

    def remove_stops(self, stops_to_remove:list[TransitStation]):
        """
        removes the stops from the list of stops
        :param stops_to_remove: list of stops to remove                                                                  # need to figure out naming convention for stops
        :return:
        """
        pass

class LineVariantElements(TransitDataFeature):
    """ Line variant elements class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__("LineVariantElements", place_name, feature_dataset, agencies)
        
class LVEShapes(TransitDataFeature):
    """ LVEShapes class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__("LVEShapes", place_name, feature_dataset, agencies)

class Calendars(TransitDataFeature):
    """ Calendars class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__("Calendars", place_name, feature_dataset, agencies)

class CalendarExceptions(TransitDataFeature):
    """ Calendar exceptions class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__("CalendarExceptions", place_name, feature_dataset, agencies)

class Lines(TransitDataFeature):
    """ Lines class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__("Lines", place_name, feature_dataset, agencies)

class LineVariants(TransitDataFeature):
    """ Line variants class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__("LineVariants", place_name, feature_dataset, agencies)
        self.agencies = agencies

class Runs(TransitDataFeature):
    """ Runs class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__("Runs", place_name, feature_dataset, agencies)
        self.agencies = agencies

class ScheduleElements(TransitDataFeature):
    """ Schedule elements class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__("ScheduleElements", place_name, feature_dataset, agencies)
        self.agencies = agencies

class Schedules(TransitDataFeature):
    """ Schedules class for use in public transit data model for arcgis"""
    def __init__(self, place_name:str, feature_dataset:FeatureDataset, agencies:list[TransitAgency]):
        super().__init__("Schedules", place_name, feature_dataset, agencies)
        self.agencies = agencies

gtfs_features_in_ptdm = [Stops, LineVariantElements, LVEShapes, Calendars, CalendarExceptions, Lines, LineVariants,
                         Runs, ScheduleElements, Schedules]

class PublicTransitDataModel:
    def __init__(self, place_name:str, agencies:list[TransitAgency], feature_dataset:FeatureDataset):

        # only importing so can use FeatureDataset class for type checking
        from create_network_dataset_oop import FeatureDataset

        self.place_name = place_name
        self.snake_name = create_snake_name(self.place_name)
        self.agencies = agencies
        self.feature_dataset = feature_dataset

        # list of the various transit data features that make up a public transit data model
        self.transit_data_features = []
        # go through all the different types of data features and instantiate them
        for transit_data_feature in gtfs_features_in_ptdm:
            data_feature_instance = transit_data_feature(self.place_name, self.feature_dataset, self.agencies)
            self.transit_data_features.append(data_feature_instance)

    def check_if_ptdm_exists(self):
        """
        Checks to see that all the data features associated with a public transit data model exist
        :return: bool | True iff ALL of the associated data features exist in the feature dataset
        """
        # going through all the necessary data features for a public transit data model and checking if they exist
        for data_feature in self.transit_data_features:

            # if ANY of the associated data features don't exist the public transit data model must be incomplete
            if not data_feature.check_if_exists():
                logging.warning(f"Data feature {data_feature.feature_name} does not exist in feature dataset "
                                f"{self.feature_dataset.name}")
                return False
        # returns True iff ALL of the associated data features exist
        return True




