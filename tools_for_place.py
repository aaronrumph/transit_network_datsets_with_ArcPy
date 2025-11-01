"""
This module exists to take all the various tools already created in the create network_dataset_oop.py file and
organize them into a class that can be used to create network datasets from OSM street network data. Also used
to get transit data for a place and use that to create a transit network dataset.
"""
import time

# yes I know it's bad practice to use import * but in this case, I've made sure that it won't cause any problems
# (can safely map namespace of create_network_dataset_oop to this module because were developed in tandem)
from spare_cnd_oop import *
from gtfs_tools import *
from general_tools import *
import transit_data_for_arcgis

# standard library modules
import logging
import os


# logging setup
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

arcgis_bin = r"C:\Program Files\ArcGIS\Pro\bin"
arcgis_extensions = r"C:\Program Files\ArcGIS\Pro\bin\Extensions"
os.environ["PATH"] = arcgis_bin + os.pathsep + arcgis_extensions + os.pathsep + os.environ.get("PATH", "")
# if you want to add modules, MUST (!!!!!!!) come after this block (ensures extensions work in cloned environment)
import arcpy
import arcpy_init
#


class Place:
    """ Class that represents a place and contains methods to create network datasets for it"""
    def __init__(self, arcgis_project:ArcProject, name=None, bound_box=None):
        if name is None and bound_box is None:
            raise ValueError("Must provide either a place or bounding box")
        # parameters
        self.arcgis_project = arcgis_project
        self.name = name
        self.bound_box = bound_box

        # important attributes not passed
        self.reference_place = ReferencePlace(place_name=self.name, bound_box=self.bound_box)
        self.snake_name = create_snake_name(self.reference_place)
        # cache folder for place
        self.cache_folder = CacheFolder(self.snake_name)

        # attributes to check whether certain things exist
        self.street_network_data_exists = False
        self.elevation_data_exists = False
        self.gdb_exists = False
        self.feature_dataset_exists = False
        self.streets_feature_classes_exists = False
        self.network_dataset_types = []
        self.network_datasets_built = {}

        # set up cache folder when initializing instance if one doesn't already exist
        if not self.cache_folder.check_if_cache_folder_exists():
            self.cache_folder.set_up_cache_folder()

        # agencies that serve the place (will be set later by the get_agencies_for_place method)
        self.agencies_that_serve_place = None

    def create_network_dataset_from_place(self, network_type="walk", use_elevation=False, full_reset=False,
                                          elevation_reset=False):                                # still need to figure out what to do with bounding box rather than place
        """
        Creates network dataset from for specified place using OSM street network data.
        :param network_type | str:
        :param use_elevation| bool:
        :param full_reset:
        :param elevation_reset:
        :return:
        """
        logging.info(f"Creating network dataset for {self.name}")

        if not use_elevation:
            # add no_z to network_type if not using elevation to work with network_types_attributes dict (see module)
            network_type += "_no_z"
        else:
            # add _z to network_type if using elevation to work with network_types_attributes dict (see module)
            network_type += "_z"

        # create StreetNetwork object for this place
        street_network_for_place = StreetNetwork(self.name, network_type=network_type)

        # prepare geodatabase and create feature dataset
        geodatabase_for_place = GeoDatabase(self.arcgis_project, street_network=street_network_for_place)
        geodatabase_for_place.set_up_gdb(reset=full_reset)
        feature_dataset_for_place = FeatureDataset(geodatabase_for_place, street_network_for_place,
                                                   network_type=network_type, reset=full_reset)
        feature_dataset_for_place.create_feature_dataset()

        # elevation handling for street network
        if not use_elevation:
            street_network_for_place.get_street_network_from_osm()
            # take street network map to feature classes
            street_feature_classes_for_place = StreetFeatureClasses(feature_dataset_for_place, street_network_for_place,
                                                                    use_elevation=False,
                                                                    reset=full_reset)
        else:
            # using elevation
            street_network_for_place.get_street_network_from_osm()

            # create new elevationmapper object for this street network and then getting elevations and nodes
            elevation_mapper_for_place = ElevationMapper(street_network=street_network_for_place,reset=elevation_reset)
            elevation_mapper_for_place.add_elevation_data_to_nodes()
            elevation_mapper_for_place.add_grades_to_edges()

            # worth using reset here just to be safe because still need to debug
            street_feature_classes_for_place = StreetFeatureClasses(feature_dataset_for_place, street_network_for_place,
                                                                    use_elevation=True,
                                                                    reset=True)

        # setting up empty street feature classes (edges_fc and nodes_fc), then populating them, then caching them
        street_feature_classes_for_place.create_empty_feature_classes()
        street_feature_classes_for_place.add_street_network_data_to_feature_classes()
        street_feature_classes_for_place.save_street_feature_classes_to_shapefile()


        # create and build network dataset from streets feature classes
        network_dataset_for_place = NetworkDataset(feature_dataset_for_place, network_type=network_type,
                                                   reset=full_reset)
        network_dataset_for_place.create_network_dataset()
        network_dataset_for_place.build_network_dataset()

    def get_agencies_for_place(self):

        """ Takes a place name of format 'city, state, country'
        and returns a list of transit agencies that serve the place

        :return: list[TransitAgency] | list of transit agencies (TransitAgencyObjects) that serve the place
        """
        # list of TransitAgency objects to be returned
        agencies_for_place = []

        # transit land's API only requires the city name (although this seems stupid)
        place_short_name = self.reference_place.place_name.split(",")[0]                                                    # fix so can use bounding box too
        transit_land_response = requests.get(f"https://transit.land/api/v2/rest/agencies?api_key={transit_land_api_key}"
                                             f"&city_name={place_short_name}")
        transit_land_response.raise_for_status()
        transit_land_data = transit_land_response.json()

        # going through the agency dicts provided by the api and using them as kwargs for TransitAgency object
        for agency_data in transit_land_data["agencies"]:
            temp_agency = TransitAgency(**agency_data)
            agencies_for_place.append(temp_agency)

        # now can set self.agencies_that_serve_place
        self.agencies_that_serve_place = agencies_for_place

        return self.agencies_that_serve_place

    def reset_place_cache(self):
        if self.cache_folder.check_if_cache_folder_exists():
            raise Exception(f"Cannot reset cache folder for place {self.snake_name}")

    def check_if_place_elevation_data_exists(self):
        pass

    def check_if_gdb_exists(self):
        pass

    def check_if_street_network_data_exists(self):
        pass

    def check_if_feature_dataset_exists(self):
        pass

    def check_if_streets_feature_classes_exists(self):
        pass


# # # # # # # # # # # # # # # # # # Testing Area :::: DO NOT REMOVE "if __name__ ..." # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    arc_package_project = ArcProject("network_dataset_package_project")
    arc_package_project.set_up_project()
    test_place = Place(arc_package_project, "Cincinnati, Ohio, USA")
    test_place.create_network_dataset_from_place(network_type="walk",
                                                 use_elevation=True, full_reset=True, elevation_reset=True)


