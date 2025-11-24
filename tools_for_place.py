"""
This module exists to take all the various tools already created in the create_network_dataset_oop.py file and
organize them into a class that can be used to create network datasets from OSM street network data. Also used
to get transit data for a place and use that to create a transit network dataset.
"""
from Geoenrichment import travel_mode

# yes I know it's bad practice to use import * but in this case, I've made sure that it won't cause any problems
# (can safely map namespace of create_network_dataset_oop to this module because were developed in tandem)
from create_network_dataset_oop import *
from gtfs_tools import *
from general_tools import *


# standard library modules
import logging
import os



# logging setup
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
# don't want to display debugging messages when running this script
logging.getLogger("requests").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)

# set up arcpy environment
arcpy_config.set_up_arcpy_env()

# if you want to add modules (non std lib), MUST (!!!!!!!) come after this block
# (ensures extensions work in cloned environment)
import arcpy
import arcpy_init
#


class Place:
    """
    Class that represents a place and contains methods to create network datasets for it

    Parameters:
        arcgis_project (ArcProject): ArcGIS project object
        place_name (str): Name of place
        bound_box (tuple): Bounding box of place | (longitude_min, latitude_min, longitude_max, latitude_max)
        geographic_scope (str): Geographic scope of place | {"place_only"", "county", "msa", "csa", "specified"}
            specified means a list of places to include in the network dataset (must be well-formed place names that
            OSM will recognize).
        scenario_id (str): Scenario ID | if you would like to keep track of the same GeoDatabase/Network Dataset
            across multiple runs, then you should use the same scenario_id for each run! By default, will be
            a random (partially modified) base64 value.
        specified_places_to_include (list[str]): List of places to include in the network dataset if using
            "specified" geographic scope

    Methods:
        use_scope_for_place(geographic_scope="place_only") -> None: Sets the geographic scope for the place
        create_network_dataset_from_place(network_type="walk", use_elevation=False, full_reset=False,
                                          elevation_reset=False) -> None: Creates network dataset for place of specified
                                          type
        get_agencies_for_place() -> None: Gets agencies that serve the place (for transit network datasets)
        generate_isochrones_for_place() -> None: Generates isochrones for the place

    """
    def __init__(self, arcgis_project:ArcProject, place_name:str | None=None,
                 bound_box:tuple[str | float, str | float, str | float, str | float] | None=None,
                 geographic_scope:str="place_only",
                 scenario_id:str=None, specified_places_to_include:list[str]=None):


        if place_name is None and bound_box is None:
            raise ValueError("Must provide either a place or bounding box")
        # parameters
        self.arcgis_project = arcgis_project
        self.place_name = place_name
        self.bound_box = bound_box
        self.geographic_scope = geographic_scope
        self.scenario_id = scenario_id
        self.specified_places_to_include = specified_places_to_include

        if self.scenario_id is None:
            self.scenario_id = generate_random_base64_value(1000000000)

        if self.bound_box:
            self.geographic_scope = "bbox"

        # important attributes not passed
        self.main_reference_place = ReferencePlace(place_name=self.place_name, bound_box=self.bound_box)
        self.snake_name = create_snake_name(self.main_reference_place)
        self.snake_name_with_scope = f"{self.snake_name}_{self.geographic_scope}"
        # cache folder for place
        self.cache_folder = CacheFolder(self.snake_name_with_scope)

        # list of reference places for creating networks
        self.reference_place_list = []

        # the corresponding gdb path:
        self.gdb_path = os.path.join(self.arcgis_project.project_dir_path, f"{self.snake_name_with_scope}.gdb")
        self.network_dataset_for_place = None

        # attributes to check whether certain things exist
        self.street_network_data_exists = False
        self.elevation_data_exists = False
        self.gdb_exists = False
        self.feature_dataset_exists = False
        self.streets_feature_classes_exists = False


        # set up cache folder when initializing instance if one doesn't already exist
        if not self.cache_folder.check_if_cache_folder_exists():
            self.cache_folder.set_up_cache_folder()

        # agencies that serve the place (will be set later by the get_agencies_for_place method)
        self.agencies_that_serve_place = None

        # always use scope for place
        self.use_scope_for_place(geographic_scope=geographic_scope)

    def use_scope_for_place(self, geographic_scope="place_only"):

        # can set the geographic scope to something new
        self.geographic_scope = geographic_scope

        # if using bounding box
        if self.bound_box:
            # the list of ReferencePlaces to use in creating the network (just the bounding box)
            reference_place_list = [self.main_reference_place]

        # if using city limits and reference place
        elif self.place_name and self.geographic_scope == "place_only":
            # the list of ReferencePlaces to use in creating the network
            reference_place_list = [self.main_reference_place]

        # if using specified places
        elif self.place_name and self.geographic_scope == "specified":
            logging.info(f"Using the provided list of other places to include {self.specified_places_to_include}")
            # check that if specified is selected that specified_places_to_include is not None
            if self.specified_places_to_include is None:
                raise ValueError("Geographic scope is 'specified', but no specified places to include were provided")
            # for each place name in specified_places_to_include, create a ReferencePlace object and add it to the list
            reference_place_list = [ReferencePlace(place_name=place_name) for
                                    place_name in self.specified_places_to_include]
            # also need to add the original place to the list (in first position so that the network is seen as being
            # 'centered' on the main place)
            reference_place_list.insert(0, self.main_reference_place)

        # if using county, msa, or csa
        elif (self.place_name and
              (self.geographic_scope == "county" or self.geographic_scope == "msa" or self.geographic_scope == "csa")):
            reference_place_list = get_reference_places_for_scope(self.place_name, self.geographic_scope)

        else:
            raise ValueError("Geographic scope not recognized, please use "
                            "'city', 'county', 'msa', 'csa', or 'specified'")

        # now set the list
        self.reference_place_list = reference_place_list
        return reference_place_list

    def create_network_dataset_from_place(self, network_type="walk", use_elevation=False, full_reset=False,
                                          elevation_reset=False) -> None:                                # still need to figure out what to do with bounding box rather than place
        """
        Creates network dataset from for specified place using OSM street network data.
        :param network_type: str | {"walk", "drive", "bike", "transit"}
        :param use_elevation: bool | whether to use elevation data for the streets in the network (in order to calculate
            grade adjusted walk-times/bike-times
        :param use_elevation: bool | Whether to use elevation when creating the network dataset. If True, will query
            USGS EPQS API to get elevations, but be warned this can be very slow (even though I made it as fast as I
            could with asyncio)
        :param full_reset: bool | WARNING!!: If True, will nuke the entire geodatabase (if it exists) for the place.
            Use with caution!
        :param elevation_reset: bool | If True, will reset the elevation data for the network dataset (if it exists).
            To be avoided because requires querying the EPQS API again, which can be very slow. However, if you are
            seeing that many edges in the network dataset have no elevation data (will see a logging message in the
            console), you may want to try this.
        :return: None | You will have to manually open ArcGIS Pro, and open the catalog to add your network dataset to
            your map
        """
        logging.info(f"Creating network dataset for {self.main_reference_place.pretty_name}")

        if not use_elevation:
            # add no_z to network_type if not using elevation to work with network_types_attributes dict (see module)
            network_type += "_no_z"
        else:
            # add _z to network_type if using elevation to work with network_types_attributes dict (see module)
            network_type += "_z"

        # create StreetNetwork object for this place
        street_network_for_place = StreetNetwork(self.geographic_scope,self.reference_place_list,
                                                 network_type=network_type)

        # prepare geodatabase and create feature dataset
        geodatabase_for_place = GeoDatabase(self.arcgis_project, street_network=street_network_for_place)
        geodatabase_for_place.set_up_gdb(reset=full_reset)
        feature_dataset_for_place = FeatureDataset(geodatabase_for_place, street_network_for_place,
                                                   scenario_id=self.scenario_id, network_type=network_type,
                                                   reset=full_reset)
        feature_dataset_for_place.create_feature_dataset()

        # elevation handling for street network
        if not use_elevation:
            street_network_for_place.get_street_network_from_osm(reset=full_reset)
            # take street network map to feature classes
            street_feature_classes_for_place = StreetFeatureClasses(feature_dataset_for_place, street_network_for_place,
                                                                    use_elevation=False,
                                                                    reset=full_reset)
        else:
            # using elevation
            street_network_for_place.get_street_network_from_osm(reset=full_reset)

            # create new ElevationMapper object for this street network and then getting elevations and nodes
            elevation_mapper_for_place = ElevationMapper(street_network=street_network_for_place,reset=elevation_reset)
            elevation_mapper_for_place.add_elevation_data_to_nodes()
            elevation_mapper_for_place.add_grades_to_edges()

            # worth using reset here just to be safe because still need to debug
            street_feature_classes_for_place = StreetFeatureClasses(feature_dataset_for_place, street_network_for_place,
                                                                    use_elevation=True,
                                                                    reset=full_reset)

        # testing switch over to gdf -> shp -> fc method
        street_feature_classes_for_place.map_street_network_to_feature_classes()

        # if using transit then need to create a Transit Network and get it setup
        if "transit" == network_type[:7]:
            transit_network_for_place = TransitNetwork(geographic_scope=self.geographic_scope,
                                                       feature_dataset=feature_dataset_for_place,
                                                       reference_place_list=[self.main_reference_place])
            # download gtfs data for place
            transit_network_for_place.unzip_gtfs_data()
            # create public transit model from downloaded GTFS data
            transit_network_for_place.create_public_transit_data_model()
            transit_network_for_place.connect_network_to_streets()

        # create and build network dataset from streets feature classes
        network_dataset_for_place = NetworkDataset(feature_dataset_for_place, network_type=network_type,
                                                   reset=full_reset)
        # set self.network_dataset_for_place
        self.network_dataset_for_place = network_dataset_for_place

        # create and build network dataset
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
        place_short_name = self.main_reference_place.place_name.split(",")[0]                                                    # fix so can use bounding box too
        transit_land_response = requests.get(f"https://transit.land/api/v2/rest/agencies?apikey={transit_land_api_key}"
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

    def generate_isochrone(self, isochrone_name:str=None, addresses:list[str] | str=None,
                           points:list[tuple[float, float]] | tuple[float, float]=None, network_type:str="walk",
                           use_elevation:bool=False, cutoffs_minutes:list[float]=None,
                           travel_direction:str="TO_FACILITIES", day_of_week:str="Today", date:str=None,
                           analysis_time:str="9:00 AM", open_on_complete=True, reset_network_dataset:bool=False):
        """
        This method creates an isochrone for a given set of addresses or points using the specified network type and
        cutoffs. This will mostly just be used to demonstrate the functionality of the network dataset. Can pass either
        well-formed address(es) (see address parameter) or tuple(s) of (latitude, longitude) (see points parameter). Can
        specify what kind of network to use; by default, will use non-elevation enabled transit network dataset.
        If one already exists for the place, it will be used. If not, a new one matching the desired analysis
        will be generated. Can choose to specify cutoffs in minutes (see cutoffs_minutes parameter) or
        let the method use the default cutoffs (10, 20, 30, 40, 50, and 60 minute cutoffs).

        :param isochrone_name: the name for the isochrone service area layer
        :param addresses: the address(es) to use for the isochrone analysis
        :param points: the points to use for the isochrone
        :param network_type: the network type to be built or used for the analysis
        :param use_elevation: whether the network should use elevation or not
        :param cutoffs_minutes: the isochrone cutoffs to be used (in minutes)
        :param travel_direction: whether to do analysis_time to or from facilities {"FROM_FACILITIES", "TO_FACILITIES"}
        :param day_of_week: the day of the week to use for the analysis {"Today", "Monday", "Tuesday", "Wednesday",
                                                                        "Thursday", "Friday", "Saturday", "Sunday"}
        :param date: date to use for the analysis (e.g. "11/15/2025")
        :param analysis_time: time to use for the analysis (e.g. "9:00 AM")
        :param open_on_complete: whether to open the project automatically after the analysis is complete
        :param reset_network_dataset: WARNING! whether to reset the network dataset before running the analysis
            (doing this will wreck any other isochrones that used the old, conflicting network dataset if there was one

        :return:
        """
        logging.info("Generating isochrone")

        # if no name for service area layer provided, generate random base64 value
        if isochrone_name is None:
            isochrone_name = generate_random_base64_value(1000000000)

        # check that travel_direction is valid
        if travel_direction not in ["FROM_FACILITIES", "TO_FACILITIES"]:
            raise ValueError("travel_direction must be one of the following: FROM_FACILITIES, TO_FACILITIES")

        # set default cutoffs to use for service area analysis
        if cutoffs_minutes is None:
            # in case network desired is transit
            if network_type == "transit":
                cutoffs_minutes = [10, 20, 30, 40, 50, 60]
            # in case network desired is walk
            elif network_type == "walk":
                cutoffs_minutes = [5, 10, 15, 20, 25, 30]
            # in case a different type of network is desired
            else:
                cutoffs_minutes = [10, 20, 30, 40, 50, 60]

        # some exception handling to make sure input parameters are valid
        # check that either address(es) or point(s) provided
        if addresses is None and points is None:
            raise ValueError("must provide either address(es) or point(s)")

        # checking that correct type(s) for addresses (valid address checking handled by get_coordinates_from_address)
        if not isinstance(addresses, list) and not isinstance(addresses, str) and addresses is not None:
            raise ValueError("addresses must be a list of strings or a single address string")

        # checking that points are valid
        if not isinstance(points, list) and not isinstance(points, tuple) and points is not None:
            raise ValueError("points must be a list of tuples or a single tuple of (latitude, longitude)")
        # in case list of points provided
        elif isinstance(points, list):
            # check that each point is a tuple.
            for point in points:
                check_if_valid_coordinate_point(point)
        # in case single point provided
        elif isinstance(points, tuple):
            check_if_valid_coordinate_point(points)

        # check desired network type, if provided, is valid
        if not isinstance(network_type, str):
            raise ValueError("network_type must be a string")
        # check whether desired network type can be used
        if network_type not in ["transit", "walk"]:
            raise ValueError(f"network_type must be one of the following: "
                             f"{' '.join(network_types.network_types_attributes)}")

        # check that use_elevation is either True or False
        if not isinstance(use_elevation, bool):
            raise ValueError("use_elevation must be a boolean")

        # check that cutoffs_minutes, if provided, is a list of integers
        if not isinstance(cutoffs_minutes, list) and cutoffs_minutes is not None:
            raise ValueError("cutoffs_minutes must be a list of integers or None")

        # now check whether network dataset of corresponding type already exists
        if use_elevation:
            using_elevation_tag = "z"
        else:
            using_elevation_tag = "no_z"
        feature_dataset_would_be_named = f"{network_type}_{using_elevation_tag}_{self.scenario_id}_fd"
        network_dataset_path = os.path.join(self.gdb_path, feature_dataset_would_be_named,
                                            f"{network_type}_{using_elevation_tag}_nd")

        # the next part involves preparing the input points and addresses for the 'facilities' sublayer for the analysis
        # points and addresses (which have been geocoded to points) will go into the input feature class, which will
        # then be used as the input for the facilities sublayer for the analysis
        logging.info(f"Processing inputted points/addresses: {points} {addresses}")
        # the list of input points that will go into the input feature class (includes points AND addresses as points)
        input_point_coordinates = []

        # add points to list of point coordinates
        if points is not None:
            if isinstance(points, list):
                for point in points:
                    input_point_coordinates.append(point)
            elif isinstance(points, tuple):
                input_point_coordinates.append(points)
            else:
                raise ValueError("points must be a list of tuples or a single tuple of (latitude, longitude)")

        # turn address into points (can pass either list of addresses or single address so no need to check type)
        if isinstance(addresses, list):
            address_coordinates = get_coordinates_from_address(addresses)
            for address_coordinate in address_coordinates:
                # add to the list of input
                input_point_coordinates.append(address_coordinate)
        elif isinstance(addresses, str):
            address_coordinates = get_coordinates_from_address(addresses)
            input_point_coordinates.append(address_coordinates)

        # now actually dealing with network dataset for the analysis
        logging.info("Checking whether or not a network dataset required for the analysis already exists")

        # check that 1) network dataset exists and 2) reset_network_dataset is False (no ND reset desired)
        if arcpy.Exists(network_dataset_path) and not reset_network_dataset:
            logging.info(f"Using existing network dataset {network_type}_{using_elevation_tag}")
            pass
        # in case no matching network dataset could be found or reset desired
        else:
            logging.info(f"No network dataset found. Creating {network_type}_{using_elevation_tag}")
            # passing the reset_network_dataset parameter to the create_network_dataset_from_place method just in case
            self.create_network_dataset_from_place(network_type=network_type, use_elevation=use_elevation,
                                                   full_reset=reset_network_dataset,
                                                   elevation_reset=reset_network_dataset)

        # create points feature classes
        input_fc_path = add_points_arcgis(feature_dataset_path=
                                          os.path.join(self.gdb_path, feature_dataset_would_be_named),
                                          fc_name=f"{self.snake_name}_test_points", point_coordinates=input_point_coordinates)

        # set travel mode
        _travel_mode = network_types.network_types_attributes[f"{network_type}_{using_elevation_tag}"][
                                                                                                "isochrone_travel_mode"]

        # set analysis_time to analyze
        if date is None:
            # use ArcGIS magic dates for day of week
            day_map = {
                "Today": "12/30/1899",
                "Sunday": "12/31/1899",
                "Monday": "1/1/1900",
                "Tuesday": "1/2/1900",
                "Wednesday": "1/3/1900",
                "Thursday": "1/4/1900",
                "Friday": "1/5/1900",
                "Saturday": "1/6/1900"
            }
            time_to_analyze = f"{day_map[day_of_week]} {analysis_time}"
        else:
            time_to_analyze = f"{date} {analysis_time}"

        # the path where the isochrone layer will go
        isochrone_path = os.path.join(self.gdb_path, isochrone_name)
        # now can create service area layer
        check_out_network_analyst_extension()
        logging.info("Creating service area analysis layer")
        result_object = arcpy.na.MakeServiceAreaAnalysisLayer(network_data_source=network_dataset_path,
                                                              layer_name=isochrone_name, travel_mode=_travel_mode,
                                                              travel_direction=travel_direction, # gives warning because expects literal but fine
                                                              time_of_day=time_to_analyze, cutoffs=cutoffs_minutes,
                                                              geometry_at_overlaps="DISSOLVE")
        # get layer object out
        layer_object = result_object.getOutput(0)

        # get facilities and polygons names
        sublayer_names = arcpy.na.GetNAClassNames(layer_object)
        facilities_layer_name = sublayer_names["Facilities"]
        polygons_layer_name = sublayer_names["SAPolygons"]

        # add facilities to layer
        logging.info("Adding inputted points/addresses to service area analysis layer")
        arcpy.na.AddLocations(layer_object, facilities_layer_name, in_table=input_fc_path)

        # solve the layer
        logging.info("Solving service area analysis layer")
        arcpy.na.Solve(layer_object)

        # save the service area as layer file ### NEED TO FIX LAYER NAME BECAUSE RIGHT NOW IT IS FULL PATH??
        layers_dir = os.path.join(self.arcgis_project.project_dir_path, "layers")
        if not os.path.exists(layers_dir):
            os.makedirs(layers_dir, exist_ok=True)
        output_layer_file = os.path.join(layers_dir, f"{isochrone_name}.lyrx")
        arcpy.SaveToLayerFile_management(in_layer=layer_object, out_layer=output_layer_file, is_relative_path="ABSOLUTE")

        # setting up the maps for the project
        maps = self.arcgis_project.arcObject.listMaps("Map")
        if not maps:
            self.arcgis_project.arcObject.createMap("Map")
            self.arcgis_project.arcObject.save()
            maps = self.arcgis_project.arcObject.listMaps("Map")

        # add the layer to the map
        aprxMap = maps[0]
        # now make LayerFile Object and add to map
        isochrone_layer_file = arcpy.mp.LayerFile(output_layer_file)
        aprxMap.addDataFromPath(isochrone_layer_file)

        # color to make all pretty like
        sa_polygons = aprxMap.listLayers(polygons_layer_name)[0]
        polygon_symbology = sa_polygons.symbology
        polygon_symbology.updateRenderer('GraduatedColorsRenderer')
        polygon_symbology.renderer.classificationField = "ToBreak"
        polygon_symbology.renderer.breakCount = len(cutoffs_minutes)
        polygon_symbology.renderer.classificationMethod = "NaturalBreaks"
        polygon_symbology.renderer.colorRamp = self.arcgis_project.arcObject.listColorRamps("Inferno")[0]

        # have to manually reverse colors because stupid
        breaks = polygon_symbology.renderer.classBreaks
        colors = [brk.symbol.color for brk in breaks]
        for i, brk in enumerate(breaks):
            brk.symbol.color = colors[i]

        # now can actually set the polygon's symbology
        sa_polygons.symbology = polygon_symbology

        # saving hopefully makes changes persist
        self.arcgis_project.arcObject.save()

        # housekeeping
        logging.info("Isochrones generated successfully")

        if open_on_complete:
            logging.info("Now opening ArcGIS Pro")
            # need to clear the arcObject to ensure it's not locked so can open automatically because lazy
            del self.arcgis_project.arcObject
            # sleepytime! (I'm so incredibly sick of debugging this, and I'm getting a bit loopy)
            from time import sleep as sleepytime
            sleepytime(1)
            # now can open project automatically?
            os.startfile(self.arcgis_project.path)

    # will add support for route as well
    def generate_route(self, route_name:str=None, addresses:list[str] | str=None,
                           points:list[tuple[float]] | tuple[float]=None, network_type:str="transit",
                           use_elevation:bool=False, day_of_week:str="Today", date:str=None, analysis_time:str="9:00 AM", 
                       open_on_complete=True):
        pass

    # important!! method that generates origin-destination cost matrix
    def generate_od_cost_matrix(self, matrix_name:str=None, origins_fc_name:str=None,
                                destinations_fc_name:str=None, network_type:str="transit",
                                use_elevation:bool=False, day_of_week:str="Today", analysis_date:str=None,
                                analysis_time:str="9:00 AM", open_on_complete:bool=False,
                                reset_network_dataset:bool=False):
        """
        :param matrix_name: (str) the name for the od-cost matrix that will be generated
        :param origins_fc_name: (str) the name of the feature class containing the points to be used as origins in
            generating the matrix. Note that name should be relative path and feature class must be points!
        :param destinations_fc_name: (str) the name of the feature class containing the points to be used as destinations
            in generating the matrix. Note that name should be relative path and feature class must be points!
        :param network_type: (str) the type of network to be used for the matrix
        :param use_elevation: (bool) if True, will try to use an elevation enabled network dataset
        :param day_of_week: (str)
        :param analysis_date:
        :param analysis_time:
        :param open_on_complete:
        :param reset_network_dataset:
        :return:
        """
        # set the environment/workspace
        arcpy.env.workspace = self.gdb_path
        arcpy.env.overwriteOutput = True

        # first, lots of type checking of course
        if not isinstance(matrix_name, str) and matrix_name is not None:
            raise Exception("Matrix name must be a string or left alone")

        if not isinstance(origins_fc_name, str) and origins_fc_name is not None:
            raise Exception("The name for the feature class to be used for the origins must be a string")

        # also check that necessary parameters passed
        if origins_fc_name is None or destinations_fc_name is None:
            raise Exception("Must provide the names of both the origins and destinations feature classes")

        # set matrix name if none provided
        if matrix_name is None:
            matrix_name = generate_random_base64_value(100000000)

        # now check whether network dataset of corresponding type already exists
        if use_elevation:
            using_elevation_tag = "z"
        else:
            using_elevation_tag = "no_z"
        feature_dataset_would_be_named = f"{network_type}_{using_elevation_tag}_{self.scenario_id}_fd"
        network_dataset_path = os.path.join(self.gdb_path, feature_dataset_would_be_named,
                                            f"{network_type}_{using_elevation_tag}_nd")

        # check that 1) network dataset exists and 2) reset_network_dataset is False (no ND reset desired)
        if arcpy.Exists(network_dataset_path) and not reset_network_dataset:
            logging.info(f"Using existing network dataset {network_type}_{using_elevation_tag}")
            pass
        # in case no matching network dataset could be found or reset desired
        else:
            logging.info(f"No network dataset found. Creating {network_type}_{using_elevation_tag}")
            # passing the reset_network_dataset parameter to the create_network_dataset_from_place method just in case
            self.create_network_dataset_from_place(network_type=network_type, use_elevation=use_elevation,
                                                   full_reset=reset_network_dataset,
                                                   elevation_reset=reset_network_dataset)


        # set travel mode for the analysis
        _analysis_travel_mode = network_types.network_types_attributes[f"{network_type}_{using_elevation_tag}"][
            "isochrone_travel_mode"]

        # set analysis_time to analyze
        if analysis_date is None:
            # use ArcGIS magic dates for day of week
            day_map = {
                "Today": "12/30/1899",
                "Sunday": "12/31/1899",
                "Monday": "1/1/1900",
                "Tuesday": "1/2/1900",
                "Wednesday": "1/3/1900",
                "Thursday": "1/4/1900",
                "Friday": "1/5/1900",
                "Saturday": "1/6/1900"
            }
            time_to_analyze = f"{day_map[day_of_week]} {analysis_time}"
        else:
            time_to_analyze = f"{analysis_date} {analysis_time}"

        # generate the actual cost matrix layer
        check_out_network_analyst_extension()
        logging.info("Creating OD Cost Matrix Analysis Layer")
        result_object = arcpy.na.MakeODCostMatrixAnalysisLayer(network_data_source=network_dataset_path,
                                                               layer_name=matrix_name, travel_mode=_analysis_travel_mode,
                                                               cutoff=None,
                                                               time_of_day=time_to_analyze,
                                                               time_zone="LOCAL_TIME_AT_LOCATIONS",
                                                               line_shape="NO_LINES")

        # get layer object out
        layer_object = result_object.getOutput(0)

        # get the names of the sublayers
        sublayer_names = arcpy.na.GetNAClassNames(layer_object)
        origins_layer_name = sublayer_names["Origins"]
        destination_layer_name = sublayer_names["Destinations"]

        # now need to add locations for origins and destinations
        logging.info("Adding origins to the matrix layer")
        arcpy.na.AddLocations(in_network_analysis_layer=layer_object, sub_layer=origins_layer_name,
                              in_table=origins_fc_name, search_tolerance="10000 Meters")
        logging.info("Adding destinations to the matrix layer")
        arcpy.na.AddLocations(in_network_analysis_layer=layer_object, sub_layer=destination_layer_name,
                              in_table=destinations_fc_name, search_tolerance="10000 Meters")

        # now can solve the layer
        logging.info("Solving the OD Cost Matrix")
        arcpy.na.Solve(layer_object)

        # the path where will save a copy of the layer file
        output_layer_file = os.path.join(self.cache_folder.path, "od_cost_matrices", f"{network_type}",
                                         f"{matrix_name}.lyrx")
        # make directories in case don't exist yet
        os.makedirs(os.path.dirname(output_layer_file), exist_ok=True)
        # now can save
        logging.info(f"Saving a copy of the OD cost matrix to {output_layer_file}")
        layer_object.saveACopy(output_layer_file)

        logging.info(f"Successfully solved matrix {matrix_name} for {self.main_reference_place.pretty_name} "
                     f"{self.geographic_scope} {analysis_time} {day_of_week}")

        # setting up the maps for the project
        logging.info("Adding OD Cost Matrix layer to map")
        maps = self.arcgis_project.arcObject.listMaps("Map")
        if not maps:
            self.arcgis_project.arcObject.createMap("Map")
            self.arcgis_project.arcObject.save()
            maps = self.arcgis_project.arcObject.listMaps("Map")

        # add the layer to the map
        aprxMap = maps[0]
        # now make LayerFile Object and add to map
        od_matrix_layer_file = arcpy.mp.LayerFile(output_layer_file)
        aprxMap.addDataFromPath(od_matrix_layer_file)






# # # # # # # # # # # # # # # # # # Testing Area :::: DO NOT REMOVE "if __name__ ..." # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    arc_package_project = ArcProject("upp_461_final")
    Berkeley = Place(arc_package_project, place_name="Chicago, Illinois, USA", geographic_scope="csa",
                       scenario_id="AM_Peak")
    Berkeley.generate_od_cost_matrix(matrix_name="AM_Peak", origins_fc_name="taz_centroids_",
                                     destinations_fc_name="taz_centroids_", network_type="transit", use_elevation=False)






    ### NOTE FOR DEBUGGING/FIXING CODE::::
    """
    create_network_dataset_from_place() fails if 
        1. The Place object (or an equivalent one with the same attributes) has already had a network dataset created
        and 
        2. The network dataset created prior's use_elevation attribute does not match the use_elevation attribute of the 
            current call. 
            
    This is because switched from caching the edges gdf to using geojson as cache and now precalculating walk times (as 
    opposed to calculating field IN Arc before). Easy solution is to calculate both "flat_walk_time" and graded walk times.
    Need to edit template xml tho
    
    also: at the moment, when using specified, MSA, or CSA as geographic scope, the graphs that osmnx returns only 
    include edges that are fully inside a given place. Thus, because in my workflow I am getting the graphs for each
    place in the list (whether that be each county for CSAs/MSAs or each specified place for specified), and then 
    composing them, at the moment the resulting network dataset has gaps at the borders of the places (in parentheses 
    above). I'm planning on fixing this by getting the polygon for each place from osmnx using the geocode_to_gdf 
    function and then combining the polygons using shapely, and then using graph_from_polygon instead of ...from_place,
    but that will be next. 
"""







