import math
import os
import shutil
import sys
from pathlib import Path

from arcgis import features



# Setting up python environment/making sure env points to extensions correctly
import arcpy_config
arcpy_config.set_up_arcpy_env()

# if you want to add modules, MUST (!!!!!!!) come after this block (ensures extensions work in cloned environment)
import arcpy
import arcpy_init
#

import pickle
import logging
import osmnx as ox  # used for getting streets data
import geopandas as gpd
import networkx as nx
import time  # used for checking runtimes of functions/methods
import requests  # used for USGS API querying
import multiprocessing as mp  # used for querying in bulk
import asyncio # used for querying USGS api asynchronously
import aiohttp # same as requests (basically) but for asyncio
from itertools import repeat
import re
import platform
import isodate
from datetime import datetime
from zipfile import ZipFile
import random
from shapely.geometry import Point
import numpy as np

# local module(s)
import transit_data_for_arcgis
import network_types
from general_tools import *
from gtfs_tools import *
from gtfs_tools import transit_land_api_key

# making sure that using windows because otherwise cannot use arcpy and ArcGIS
if platform.system() != "Windows":
    raise OSError("Cannot run this module because not using Windows. ArcGIS and ArcPy require Windows")

# logging setup
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# osmnx setup
ox.settings.timeout = 3000
ox.settings.max_query_area_size = 5000000000000
ox.settings.use_cache = True

# arcgis project template to allow for creating project
arcgis_project_template = arcpy.mp.ArcGISProject("project_template.aprx")

# simple thing to make sure network analyst is checked in/out
network_analyst_extension_checked_out = False


### need to fix arcproject setup/saving etc. ###

# ArcGIS project setup
class ArcProject:
    """ Class for ArcGIS Projects.
        Parameters:
            name: str,
            project_location: str | path to directory where project should exist, by default,
                the project will be within the same directory as this module.
            reset: bool (clears project if true).

        Methods:
            set_up_project(self) (see documentation below)
    """

    def __init__(self, name, project_location=Path(__file__).parent, reset=False):
        self.name = name
        self.project_location = project_location
        self.reset = reset
        self.project_dir_path = os.path.join(self.project_location, self.name)
        self.path = os.path.join(self.project_dir_path, rf"{self.name}.aprx")
        self.maps = []
        self.layouts = []
        self.project_file = self.path
        self.arcObject = None

        # set up on initialization
        self.set_up_project()

    # method to create project if doesn't already exist, and to activate project
    def set_up_project(self):
        """
        Checks if a project already exists with the same name at the selected path, if so,
        sets project as active workspace. Otherwise, creates a blank project with nothing in it with desired name in
        desire location.
        :return
            self.path | str
        """
        if arcpy.Exists(self.path):
            logging.info("Project already exists, opening it")
            self.arcObject = arcpy.mp.ArcGISProject(self.path)
        else:
            logging.info("Project does not exist, creating it")
            os.makedirs(self.project_dir_path)
            arcgis_project_template.saveACopy(self.path)
            self.arcObject = arcpy.mp.ArcGISProject(self.path)
        # set project as current workspace
        arcpy.env.workspace = self.project_dir_path

        return self.path
    # add more methods here. Add map, etc.

    def save_project(self):
        """Saves the arcproject"""
        try:
            # have to release the project file so can save????/
            del self.arcObject
            # reopen the project
            self.arcObject = arcpy.mp.ArcGISProject(self.path)
            # and now can save it
            self.arcObject.save()
            logging.info("Project saved and reopened successfully")

        except OSError as e:
            logging.warning(f"Could not save project (file may locked or project open: {e}")

# global functions needed at global level to do multiprocessing for API queries
def init_worker(counter, lock):
    # initializer function for multiprocess processes, need in order to have a global variable accessible to all workers
    global shared_counter, shared_lock
    shared_counter = counter
    shared_lock = lock

# functions for dealing with checking network analyst extension in and out
def check_out_network_analyst_extension():
    """
    Checks out network analyst extension
    :return: True - successful, exception if False
    """
    # need to check that network analyst extension is actually available to use, and then check it out
    if arcpy.CheckExtension("Network") == "Available":
        arcpy.CheckOutExtension("Network")
        logging.info("Network Analyst extension checked out")
        return True
    else:
        raise Exception("Network Analyst extension is not available")


def check_network_analyst_extension_back_in():
    """
    Checks network analyst extension back in
    :return: True - successful, exception if False
    """
    # need to check that network analyst extension is actually available to use, and then check it out
    if network_analyst_extension_checked_out:
        arcpy.CheckInExtension("Network")
        logging.info("Network Analyst extension checked back in")
        return True
    else:
        raise Exception("Network Analyst extension is not checked out")


def add_points_arcgis(feature_dataset_path: str, fc_name: str, point_coordinates: tuple | list[tuple]) -> str:
    """
    Adds points to a feature class in a feature dataset
    :param feature_dataset_path: str
    :param fc_name: str
    :param point_coordinates: tuple | list[tuple]
    :return: str | fc_path (the path to the feature class where the points were added)
    """
    # fc_path
    fc_path = os.path.join(feature_dataset_path, fc_name)

    # check to make sure valid (lat, lon) points provided
    if isinstance(point_coordinates, list):
        for point in point_coordinates:
            check_if_valid_coordinate_point(point)
    elif isinstance(point_coordinates, tuple):
        check_if_valid_coordinate_point(point_coordinates)
    else:
        raise Exception("Invalid point coordinates provided")

    # by default will overwrite existing feature class with same name in same loc (low risk of accidental collision)
    arcpy.env.overwriteOutput = True
    # create empty feature class
    arcpy.management.CreateFeatureclass(out_path=feature_dataset_path, out_name=fc_name, geometry_type="POINT",
                                        spatial_reference=arcpy.SpatialReference(4326))
    # fields for feature class
    fields = ["SHAPE@XY"]

    # in case where list of points provided, add each one
    if isinstance(point_coordinates, list):
        with arcpy.da.InsertCursor(fc_path, fields) as cursor:
            for point in point_coordinates:
                # flip to be (x, y) rather than lat, lon (y, x)
                lat, lon = point[0], point[1]
                point_xy = (lon, lat)

                # insert point_xy
                cursor.insertRow([point_xy])

    # in case where single point provided, add it
    elif isinstance(point_coordinates, tuple):
        with arcpy.da.InsertCursor(fc_path, fields) as cursor:
            # flip to be (x, y) rather than lat, lon (y, x)
            lat, lon = point[0], point[1]
            point_xy = (lon, lat)

            # insert point_xy
            cursor.insertRow([point_xy])

    return fc_path


class CacheFolder:
    """ Class for cache folder, takes param
        network_snake_name: str (MUST be in snake case) which will be the name of cache folder

        Attributes:
            network_snake_name | str: snake name of the network (passing snake name rather than StreetNetwork object
                because CacheFolder class must preceed StreetNetwork class in the code)
            env_dir_path | Path: path to the environment directory
            path | str: path to the cache folder

        Methods:
            check_if_cache_folder_exists(): Returns True if cache folder already exists for the city.
            set_up_cache_folder(): Return True if there is already a cache folder for city. If not, creates one.
            reset_cache_folder(): Completely reset the cache folder for the city (highly unadvisable because deletes
                osm data and elevation data
    """

    def __init__(self, snake_name_with_scope):
        self.snake_name_with_scope = snake_name_with_scope
        self.env_dir_path = Path(__file__).parent
        self.path = os.path.join(self.env_dir_path, "place_caches", f"{self.snake_name_with_scope}_cache")

    def check_if_cache_folder_exists(self):
        """ Returns True if cache folder already exists for the city."""
        if os.path.exists(self.path):
            return True
        else:
            return False

    def set_up_cache_folder(self):
        """Return True if there is already a cache folder for city. If not, creates one."""
        if os.path.exists(self.path):
            raise Exception(f"There is already a cache folder for {self.snake_name_with_scope}")
        else:
            os.makedirs(self.path)

    def reset_cache_folder(self):
        # completely reset the cache folder for the city
        if not os.path.exists(self.path):
            raise Exception(f"Cannot reset the cache folder for {self.snake_name_with_scope} "
                            f"because no such folder exists"
                            )
        else:
            os.makedirs(self.path, exist_ok=True)


# simple Cache class with obvious methods (read, write, check if exists)
class Cache:
    """
    Class for cache for use in saving street network data.

    Attributes:
        cache_folder | CacheFolder obj: cache folder for the street network
        cache_name | str: name of the cache
        cache_path | str: path to the cache

    Methods:
        check_if_cache_already_exists(): checks if cache already exists
        read_cache_data(): reads cached data from cache file
        write_cache_data(): writes desired data to cache file
    """

    def __init__(self, cache_folder: CacheFolder, cache_name):
        self.cache_folder = cache_folder
        self.cache_name = cache_name
        self.cache_path = os.path.join(self.cache_folder.path, self.cache_name)

    def check_if_cache_already_exists(self):
        if os.path.exists(self.cache_path):
            return True
        else:
            return False

    def read_cache_data(self):
        if self.check_if_cache_already_exists():
            with open(self.cache_path, "rb") as cache_file:
                cache_data = pickle.load(cache_file)
            return cache_data
        else:
            raise Exception("Cannot get cache data because there is no cache")

    def write_cache_data(self, data_to_cache):
        with open(self.cache_path, "wb") as cache_file:
            pickle.dump(data_to_cache, cache_file)


# StreetNetwork class very bare bones, just gets street network graph and makes associated GeoDataframse
class StreetNetwork:
    """
    Class representing street network for a location

    Attributes:
        geographic_scope (str): Geographic scope of the street network {"place_only", "msa", "csa"}
        reference_place_list: list of reference places to get network for (len will be one if using city limits, other-
        wise, will contain all the places in the MSA if MSA desired and in CSA if CSA desired)
        network_type (str): type of network being created {"walk_no_z", "walk_z", "bike_no_z", "bike_z",
        
        Local attributes:
        "transit_no_z", "transit_z", "drive", "transit_plus_biking_no_z", "transit_plus_biking_z"}
        bound_boxes (list): Bounding boxes passed (if bounding box used, len of list will be one)
        snake_name (str): Snake name of the city
        cache_folder (CacheFolder): Cache folder for the street network
        graph_cache (Cache): Cache for the street network graph
        nodes_cache (Cache): Cache for the street network nodes
        edges_cache (Cache): Cache for the street network edges
        network_graph (networkx.Graph): Street network graph
        network_nodes (geopandas.GeoDataFrame): Street network nodes
        network_edges (geopandas.GeoDataFrame): Street network edges
        elevation_enabled (bool): Whether elevation is enabled
        osmnx_type (str): Type of street network for osmnx ("walk", "bike", "drive", "drive_service", "all", "all_public")

        Methods:
            get_street_network_from_osm(timer_on=True, reset=False): Gets street network from OpenStreetMaps or
            from cache
    """

    def __init__(self, geographic_scope: str, reference_place_list: list[ReferencePlace], network_type="walk_no_z"):
        #### need to make __init__ method cleaner ####
        self.geographic_scope = geographic_scope
        self.reference_place_list = reference_place_list
        self.network_type = network_type

        # get place name and bbox out of reference place
        self.place_names = [reference_place.place_name for reference_place in reference_place_list]
        self.bound_boxes = [reference_place.bound_box for reference_place in reference_place_list]
        self.main_reference_place = reference_place_list[0]

        if self.main_reference_place.bound_box:
            self.geographic_scope = "bbox"

        # create snake name for StreetNetwork (the first item in the list is always the main place)
        self.snake_name = create_snake_name(self.main_reference_place)
        # create snake name with geographic scope encoded
        self.snake_name_with_scope = f"{self.snake_name}_{self.geographic_scope}"
        # link cache folder
        self.cache_folder = CacheFolder(self.snake_name_with_scope)
        # decode the network type into proper network type designation for osmnx query
        self.osmnx_type = network_types.network_types_attributes[self.network_type]["osmnx_network_type"]

        # check if there is a cache folder for desired street network
        if not self.cache_folder.check_if_cache_folder_exists():
            self.cache_folder.set_up_cache_folder()

        # setting up caches for this street network
        self.graph_cache = Cache(self.cache_folder, "graph_cache")
        self.nodes_cache = Cache(self.cache_folder, "nodes_cache")
        self.edges_cache = Cache(self.cache_folder, "edges_cache")
        self.edges_cache.cache_folder.check_if_cache_folder_exists()

        # placeholder for now, but will update in get_street_network_from_osm method
        # so can access gdfs and graph when passing instance as argument
        self.network_graph = None
        self.network_nodes = None
        self.network_edges = None
        self.elevation_enabled = False

    def get_street_network_from_osm(self, timer_on=True, reset=False):
        """
        Main function for class that gets street network edges and nodes, either from cache, if cache exists, or from
        OpenStreetMaps (through osmnx)
        :param timer_on: logs run time for method if, on by default
        :param reset: if True, will reset cache and get street network from OSM
        :return: street network graph, and the nodes and egdes that make up the graph in geodataframes
        """
        process_start_time = time.perf_counter()
        logging.info("Getting street network")

        # first if not using cache or no cache data
        if reset or (not self.graph_cache.check_if_cache_already_exists()):
            logging.info("Getting street network from OSM")

            # if using city, getting OSM data if not using cache or if no cache exists
            if self.main_reference_place.bound_box is None:

                # if there is more than one reference place (not using city limits) need to combine street networks
                if len(self.reference_place_list) > 1:
                    network_graphs = [] # list of the networkx graph objects representing city street networks
                    for reference_place in self.reference_place_list:
                        logging.info(f"Getting street network for {reference_place.pretty_name}")

                        # getting each individual graph one at a time first
                        single_network_graph = ox.graph_from_place(reference_place.place_name,
                                                                   network_type=self.osmnx_type, retain_all=True,
                                                                   truncate_by_edge=True)
                                                                   # need to use truncate_by_edge when using
                                                                   # multiple reference places to avoid gaps at borders

                        # add each individual graph to the list so can combine
                        network_graphs.append(single_network_graph)

                    logging.info("Combining street networks")
                    # using compose all to combine the various street grids
                    network_graph = nx.compose_all(network_graphs)

                # when using city limits, only need street grid for main place
                elif len(self.reference_place_list) == 1:
                    logging.info(f"Getting street network for {self.main_reference_place.pretty_name} city proper")
                    network_graph = ox.graph_from_place(self.main_reference_place.place_name,
                                                        network_type=self.osmnx_type, retain_all=True)

                else:
                    raise Exception("Cannot get street network because no place was specified")
                
                # turn graph into gdfs
                network_nodes, network_edges = ox.graph_to_gdfs(network_graph, nodes=True, edges=True)


            # if using bound box, getting OSM Data
            else:
                logging.info(f"Getting street network for {self.main_reference_place.pretty_name}")
                network_graph = ox.graph_from_bbox(bbox=self.main_reference_place.bound_box,
                                                   network_type=self.osmnx_type, retain_all=True)
                # again graph to gdfs
                network_nodes, network_edges = ox.graph_to_gdfs(network_graph, nodes=True, edges=True)

        # otherwise using cached data
        else:
            logging.info("Using cached street network")
            network_graph = self.graph_cache.read_cache_data()
            network_nodes, network_edges = self.nodes_cache.read_cache_data(), self.edges_cache.read_cache_data()

        # just because I want to keep track of everything
        if timer_on:
            logging.info(f"Got street network from OSM in {time.perf_counter() - process_start_time} seconds")
            logging.info(f"Street network for {self.main_reference_place.place_name} {self.geographic_scope} had "
                         f"{len(network_nodes)} nodes and "
                         f"{len(network_edges)} edges")

        self.graph_cache.write_cache_data(network_graph)
        self.nodes_cache.write_cache_data(network_nodes)
        self.edges_cache.write_cache_data(network_edges)
        self.network_graph, self.network_nodes, self.network_edges = network_graph, network_nodes, network_edges
        return network_graph, network_nodes, network_edges

    # methods to count nodes and edges in street network. No real reason to exist other than interest
    def count_nodes(self):
        if self.network_nodes is None:
            raise Exception("Cannot count nodes because there are no nodes")
        else:
            return len(self.network_nodes)

    def count_edges(self):
        if self.network_edges is None:
            raise Exception("Cannot count edges because there are no edges")
        else:
            return len(self.network_edges)


# ElevationMapper class adds elevation to nodes and grades to edges (coming soon?) using USGS EPQS API
class ElevationMapper:
    """ WARNINGS!:
            1. Adding elevation takes forever, ~130ms per node with 14 threads (1.8s per node per thread!)
                because the API is so slow
            2. Uses multiprocessing (scary) so use the "if __name__ == __main__" guard!!!!!
        Attributes:
            street_network | StreetNetwork obj, threads_available | int, reset | bool

        Methods:
            add_elevation_data_to_nodes(): gets elevation data from USGS EPQS API and adds to nodes geodataframe
            add_elevation_data_to_edges(): calculates grades for edges based on elevation data (must be called after
                add_elevation_data_to_nodes)
    """

    def __init__(self, street_network: StreetNetwork, concurrent_requests_desired:int=100, reset=False):
        self.street_network = street_network
        self.concurrent_requests_desired = concurrent_requests_desired
        self.reset = reset
        (self.street_network_graph, self.street_network_nodes, self.street_network_edges) = (
            street_network.network_graph, street_network.network_nodes, street_network.network_edges
        )
        self.node_counter = 0
        self.elevation_cache = Cache(street_network.cache_folder, "elevation")
        self.nodes_without_elevation = 0

    # decided to switch to using asyncio rather than multiprocessing because is I/O bound
    async def get_elevation_data_single_query(self, async_session:aiohttp.ClientSession,
                                              idx, lon, lat, semaphore: asyncio.Semaphore, node_counter,
                                              total_nodes, start_time):
        usgs_url = "https://epqs.nationalmap.gov/v1/json"
        params = {"x": float(lon), "y": float(lat), "units": "Meters", "wkid": 4326}

        # using semaphore to limit number of concurrent requests
        async with semaphore:
            # using try in case of exception in getting elevation for node
            try:
                # essentially same as requests.get but for async function. Timeout set as 30s should be high enough but
                # 45 or 60s might be better (using 100 because not happy with me)
                async with async_session.get(usgs_url, params=params, timeout=100) as response:

                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")

                    # wait on response then get the "value" output received from the API
                    elevation_data_response = await response.json()
                    elevation = elevation_data_response.get("value")

                    # check whethere received actual data or default no data, which for usgs is -1000000
                    if elevation == -1000000:
                        elevation = None

                    # update counter for progress tracking
                    node_counter["nodes_completed"] += 1
                    current_node = node_counter["nodes_completed"]
                    time_elapsed = time.perf_counter() - start_time
                    querying_rate = current_node / time_elapsed if time_elapsed > 0 else 0
                    estimated_total_time = total_nodes / querying_rate if querying_rate > 0 else 0

                    # convert the elapsed time and estimated time into minutes and seconds for prettier printing
                    time_elapsed_in_minutes_and_seconds = {"minutes": time_elapsed // 60, "seconds": time_elapsed % 60}
                    estimated_total_time_in_minutes_and_seconds = {"minutes": estimated_total_time // 60,
                                                                   "seconds": estimated_total_time % 60}
                    # printing progress bar/tracker thingy
                    print(f"\rElevation: {current_node}/{total_nodes} nodes processed. "
                          f"{time_elapsed_in_minutes_and_seconds['minutes']:.0f} minutes: "
                          f"{time_elapsed_in_minutes_and_seconds['seconds']:.0f} seconds/"
                          f"{estimated_total_time_in_minutes_and_seconds['minutes']:.0f} minutes: "
                          f"{estimated_total_time_in_minutes_and_seconds['seconds']:.0f} seconds"
                          f" (elapsed time/estimated total time). Rate = {querying_rate} nodes/second",
                          end="", flush=True)

                    # now that have elevation can return the index of the node (from itterrowing them) and its elevation
                    return idx, elevation

            # in case of exception, just returning None as the elevation for the node.
            except Exception as e:
                self.nodes_without_elevation += 1
                return idx, None

        logging.warning(f"Couldn't get elevation for {self.nodes_without_elevation}/{total_nodes} nodes")
        return idx, None

    # this function just tells the little async worker thingies how and when to query the API
    async def get_all_elevation_data(self):
        """
        Gets elevation data for all nodes in street network using the async function get_elevation_data_single_query.
        :return:
        """
        # nodes to go through
        nodes_list = [(idx, float(row["x"]), float(row["y"])) for idx, row in self.street_network_nodes.iterrows()]

        # the semaphore is the host of the restaurant in the Guido asyncio restaurant analogy
        semaphore = asyncio.Semaphore(self.concurrent_requests_desired)

        # have to manually define TCP connector to force close otherwise leave zombie connections
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=100, ttl_dns_cache=300, enable_cleanup_closed=True,
                                        force_close=True)

        node_counter = {"nodes_completed": 0}
        start_time = time.perf_counter()
        total_nodes = len(nodes_list)


        # open new aiohttp session (basically does same as response = response.get...)
        async with aiohttp.ClientSession(connector=connector) as session:

            # only asyncio task is getting elevation data for each node
            tasks = [self.get_elevation_data_single_query(async_session=session, idx=idx, lon=lon, lat=lat,
                                                          semaphore=semaphore, node_counter=node_counter,
                                                          total_nodes=total_nodes, start_time=start_time)
                    for idx, lon, lat in nodes_list]

            # results from querying elevation API for each node
            results = await asyncio.gather(*tasks)
            return dict(results)

    # this method is the main method that actually sends out for the elevation data, and then adds it to street network
    def add_elevation_data_to_nodes(self):

        logging.info(f"Adding elevation to the street network for "
                     f"{self.street_network.main_reference_place.place_name} {self.street_network.geographic_scope}")

        if not self.reset and self.elevation_cache.check_if_cache_already_exists():
            elevation_data = self.elevation_cache.read_cache_data()
        else:
            # run the async function to get the elevation data for all the nodes
            elevation_data = asyncio.run(self.get_all_elevation_data())

        # make list of elevations
        z_values:list = [elevation_data.get(idx) for idx in elevation_data]
        # update the local street network gdf within the ElevationMapper
        self.street_network_nodes["z"] = z_values
        self.street_network.network_nodes["z"] = z_values

        # cache elevation_data dict {<node_idx>: <elevation>}
        self.elevation_cache.write_cache_data(elevation_data)

        # now add to the original graph
        for node_idx, elevation in elevation_data.items():
            if node_idx in self.street_network_graph.nodes:
                elevation_value = float(elevation) if elevation is not None else None
                self.street_network_graph.nodes[node_idx]["elevation"] = elevation_value
                self.street_network_graph.nodes[node_idx]["z"] = elevation_value

        # update the elevation enabled flag
        self.street_network.elevation_enabled = True

        # update the geometry of the nodes to include the elevation
        self.street_network_nodes["geometry"] = [
            Point(row.geometry.x, row.geometry.y, z if z is not None else 0)
            for (idx, row), z in zip(self.street_network_nodes.iterrows(), z_values)
        ]

        # change original network nodes geometry to include elevation
        self.street_network.network_nodes["geometry"] = [
            Point(row.geometry.x, row.geometry.y, z if z is not None else 0)
            for (idx, row), z in zip(self.street_network_nodes.iterrows(), z_values)
        ]

        # print extra blank line after progress bar thingy
        print()
        logging.info(f"Got elevation data for {len(self.street_network_nodes)} nodes")
        logging.info("Elevation data added to the street network")

    def add_grades_to_edges(self):
        """
        Calculates and adds grade (slope) to edges based on node elevations.
        Grade is calculated as: (elevation_change / edge_length) * 100
        Positive grade = uphill, Negative grade = downhill, Grade = 0 means flat ground
        """
        number_of_edges_without_elevation_data = 0

        process_start_time = time.perf_counter()
        logging.info(
            f"Adding grades to edges for {self.street_network.main_reference_place.pretty_name} street network")

        # doubled check street network is elevation enabled
        if not self.street_network.elevation_enabled:
            raise Exception("Cannot add grades to edges, elevation data not available for nodes")

        # iterating throuuh all edges in the streetnetwork graph
        for start_node, end_node, key, data in self.street_network_graph.edges(keys=True, data=True):
            # elevation for start and end nodes (start_node and end_node respectively)
            start_elevation = self.street_network_graph.nodes[start_node].get("elevation")
            end_elevation = self.street_network_graph.nodes[end_node].get("elevation")
            edge_length = data.get("length", 0)

            # only calculate grade if elevation for start and end node available
            if start_elevation is not None and end_elevation is not None and edge_length > 0:
                elevation_change = end_elevation - start_elevation
                grade = (elevation_change / edge_length) * 100  # doing grade as percentage (100% = 45 degree slope)

                # updating graph data based on calculated grades
                data["grade"] = round(grade, 2)  # rounding because otherwise crazy number

                data["grade_magnitude"] = round(abs(grade), 2)  #
                data["direction"] = 1 if grade > 0 else 0
            else:
                number_of_edges_without_elevation_data += 1
                # handling missing data
                data["grade"] = None
                data["grade_magnitude"] = None
                data["direction"] = None

        logging.warning(f"Missing elevation or length data for {number_of_edges_without_elevation_data}/"
                    f"{len(self.street_network_graph.edges())} edges total")

        # update the edges geodataframe
        self.street_network.network_nodes, self.street_network.network_edges = ox.graph_to_gdfs(
            self.street_network_graph, nodes=True, edges=True
        )

        process_run_time = time.perf_counter() - process_start_time
        logging.info(f"Successfully added grades to edges in {turn_seconds_into_minutes(process_run_time)}")


### need to review GeoDatabase and ArcProject classes code to make sure no errors
class GeoDatabase:
    def __init__(self, arcgis_project: ArcProject, street_network: StreetNetwork):
        self.project = arcgis_project
        self.street_network = street_network
        self.project_file_path = arcgis_project.path
        self.gdb_name = self.street_network.snake_name_with_scope
        self.gdb_path = os.path.join(self.project.project_dir_path, f"{self.gdb_name}.gdb")

    def save_gdb(self):
        # using try to avoid errors in case of file lock
        try:
            # save the project
            self.project.save_project()

            # now can update the list of geodatabases and set created one as the default

            # get the list of dictionaries representing the databases in the project
            gdb_dictionary_for_adding = {"databasePath": self.gdb_path, "isDefaultDatabase": True}
            # get current list of gdbs for project
            current_gdbs = self.project.arcObject.databases
            # go through and make sure that none of the current gdbs will be default
            for gdb_dict in current_gdbs:
                gdb_dict["isDefaultDatabase"] = False

            # in case the desired gdb is not showing up in the current gdbs list
            if self.gdb_path not in [gdb_dict["databasePath"] for gdb_dict in current_gdbs]:
                current_gdbs.append(gdb_dictionary_for_adding)

            # now can safely set the geodatabase for this call as default
            for gdb_dict in current_gdbs:
                if gdb_dict["databasePath"] == self.gdb_path:
                    gdb_dict["isDefaultDatabase"] = True

            # now update project
            self.project.arcObject.updateDatabases(current_gdbs)

            # housekeeping
            logging.info("Project saved successfully and default geodatabase set")

        # will get OSError if not allowed to save because of lock
        except OSError as e:
            logging.warning(f"Could not save project (file may locked or project open: {e})")

    def set_up_gdb(self, reset=False):
        """ Creates geodatabase if one does not exist, and resets it if reset desired."""
        # making sure not trying to set up gdb before project
        if not arcpy.Exists(self.project.path):
            raise FileNotFoundError(f"Cannot set up geodatabase in the the provided project {self.project.name}"
                                    f"because it doesn't exist")

        # in case where gdb does not exist and reset is not desired
        if not reset and not arcpy.Exists(self.gdb_path):
            logging.info("No existing geodatabase found, creating new geodatabase")
            arcpy.management.CreateFileGDB(self.project.project_dir_path, self.gdb_name)

        # in case where gdb does not exist but reset is desired (erroneously)
        elif not arcpy.Exists(self.gdb_path) and reset:
            logging.warning(f"No geodatabase with the name {self.gdb_name} exists, but you indicated you would like to"
                            f"reset. If this was a mistake, stop the script (in case other stuff gets reset).")
            logging.info("No existing geodatabase found, creating new geodatabase")
            arcpy.management.CreateFileGDB(self.project.project_dir_path, self.gdb_name)

        # in case where gdb exists and reset is desired
        elif arcpy.Exists(self.gdb_path):
            logging.info("Existing geodatabase found, deleting and creating new geodatabase")

            # need to delete using Arc...
            arcpy.env.overwriteOutput = True
            arcpy.Delete_management(self.gdb_path)

            # then create
            arcpy.management.CreateFileGDB(self.project.project_dir_path, self.gdb_name)

        # in case where gdb exists and reset is not desired
        else:
            raise Exception(f"A GeoDatabase {self.gdb_name} already exists but reset desire was not indicated")

        # modifying current arcpy env
        arcpy.env.workspace = self.gdb_path
        # just setting current gdb as default. need to debug to figure out why have to do this way
        self.project.arcObject.defaultGeodatabase = self.gdb_path
        # save project to make sure that default gdb actually gets set
        self.save_gdb()
        current_gdbs = self.project.arcObject.databases
        debuggggg_me = True

class FeatureDataset:  # add scenario_id so can do multiple scenarios of same network type
    """
        The feature dataset is a container for various feature classes, and is where the network dataset will go.
        
        Attributes:
            gdb (GeoDatabase): The geodatabase where the feature dataset will be created.
            street_network (StreetNetwork): The street network that will be used to create the feature dataset.
            scenario_id (str): The ID of the scenario that the feature dataset will be created for.
            network_type (str): The type of network that the feature dataset will be created for.
            reset (bool): Whether to reset the feature dataset if it already exists.
            name (str): The name of the feature dataset.
            path (str): The path to the feature dataset.
            
        Methods:
            create_feature_dataset(): Creates the feature dataset.
            reset_feature_dataset(): Resets the feature dataset.
    """
    
    def __init__(self, gdb: GeoDatabase, street_network: StreetNetwork, scenario_id:str, network_type: str = "walk_no_z", reset=False):
        self.gdb = gdb
        self.street_network = street_network
        self.scenario_id = scenario_id
        self.network_type = network_type
        self.reset = reset
        self.name = f"{self.network_type}_{scenario_id}_fd"
        self.path = os.path.join(self.gdb.gdb_path, f"{self.name}")

        # if reset true, then overwrite existing features
        arcpy.env.overwriteOutput = True

    def create_feature_dataset(self):
        # check if FD exists, and reset if needed, else create
        logging.info("Creating feature dataset")

        # making sure don't accidentally try to create when desired feature dataset already exists
        if arcpy.Exists(self.path) and not self.reset:
            raise Exception(f"The feature dataset {self.path} already exists, either reset it or use it")

        elif arcpy.Exists(self.path):
            arcpy.Delete_management(self.path)
        # creating feature dataset
        arcpy.management.CreateFeatureDataset(self.gdb.gdb_path, self.name,
                                              spatial_reference=arcpy.SpatialReference(4326))
        logging.info("Feature dataset successfully created")

    def reset_feature_dataset(self):
        # just a method to reset the desired feature dataset
        if not arcpy.Exists(self.path):
            raise Exception(f"Cannot delete feature dataset at {self.path} because it doesn't exist")
        else:
            arcpy.Delete_management(self.path)
            arcpy.management.CreateFeatureDataset(self.gdb.gdb_path, self.name,
                                                  spatial_reference=arcpy.SpatialReference(4326))


class StreetFeatureClasses:
    """
    StreetFeatureClasses is nodes and edges for the street network

    Attributes:
        feature_dataset (FeatureDataset): Feature dataset containing street nodes and edges
        street_network (StreetNetwork): Street network containing necessary nodes and edges geodataframe
        use_elevation (bool): Whether to use elevation in calculations
        reset (bool): Whether to reset the feature classes

    Methods:
        create_empty_feature_classes(): Creates empty feature classes for street nodes and edges
        add_street_network_data_to_feature_classes(): Adds street network data to empty feature classes created
    """

    def __init__(self, feature_dataset: FeatureDataset, street_network: StreetNetwork, use_elevation=False,
                 reset=False):
        self.feature_dataset = feature_dataset
        self.street_network = street_network
        self.use_elevation = use_elevation  # determines whether to add z values to nodes in feature classes
        self.reset = reset

        # paths for the two feature classes
        self.nodes_fc_path = os.path.join(self.feature_dataset.path, "nodes_fc")
        self.edges_fc_path = os.path.join(self.feature_dataset.path, "edges_fc")

        # useful shorthand to have rather than writing self.street_network.network_nodes etc all the time
        self.nodes = street_network.network_nodes
        self.edges = street_network.network_edges

        # the cache folder
        self.cache_folder = self.street_network.cache_folder

        # error handling for using elevation when creating feature classes for street network
        if "z" not in self.street_network.network_nodes and self.use_elevation:
            raise Exception("Cannot use elevation because input StreetNetwork object has no z values")

    def calculate_walk_times(self) -> None:
        """
        Adds necessary walk time columns to geodataframes for edges. In case elevation not enabled on network
        dataset, then will calculate walk time as (length in meters/ 85 meters per minute). In case elevation
        enabled, then will calculate walk time using Tobler's hiking function and calculate against and along walk
        times.
         """
        logging.info("Calculating walk times for edges")

        def calculate_flat_ground_walk_time(edges_gdf) -> None:
            logging.info("Not using elevation; will calculate walk time as (length in meters/ 85 meters per minute)")
            # love pandas/geopandas because all I have to do to calculate a new field is this!
            edges_gdf["walk_time"] = (edges_gdf["length"] / 85)

        def calculate_walk_time_with_elevation(edges_gdf) -> None:
            logging.info("Using elevation; will use Tobler's to calculate walk time")

            # in case no grade then don't just fail in the background
            if "grade" not in edges_gdf.columns:
                logging.warning("Input gdf has no grade column, will calculate flat ground walk times")
                calculate_flat_ground_walk_time(edges_gdf)
            else:
                # Am using Tobler's hiking function here (see wikipedia). FT represents "from-to" or along for a given
                # edge, and TF represents "to-from" or against for a given edge.

                # first calculate speed for along
                speed_FT_km_per_hour = 6 * (np.exp(-3.5 * np.abs((edges_gdf["grade"]/100) + 0.05)))
                speed_FT_m_per_min = (speed_FT_km_per_hour * 1000) / 60

                # now calculate speed for against
                speed_TF_km_per_hour = 6 * (np.exp(-3.5 * np.abs((-(edges_gdf["grade"]/100)) + 0.05)))
                speed_TF_m_per_min = (speed_TF_km_per_hour * 1000) / 60

                # now  graded walk times for edges are just equal to the length divided by the respective speeds above
                edges_gdf["walk_time_graded_FT"] = edges_gdf["length"] / speed_FT_m_per_min
                edges_gdf["walk_time_graded_TF"] = edges_gdf["length"] / speed_TF_m_per_min

        # now can calculate walk times
        if self.use_elevation:
            calculate_walk_time_with_elevation(self.edges)
        else:
            calculate_flat_ground_walk_time(self.edges)

    # now going to convert gdfs to geojson (faster than creating feature class using insert cursor)
    def convert_geodataframes_to_geojson(self) -> tuple[str, str]:
        logging.info("Converting geodataframes to GeoJSONs")
        process_start_time = time.perf_counter()

        # create paths for the shapefiles
        nodes_geojson_path = os.path.join(self.cache_folder.path, "nodes_fc.geojson")
        edges_geojson_path = os.path.join(self.cache_folder.path, "edges_fc.geojson")

        # make sure that the required directories exist
        os.makedirs(os.path.dirname(nodes_geojson_path), exist_ok=True)
        os.makedirs(os.path.dirname(edges_geojson_path), exist_ok=True)

        # write the gdfs to geojsons
        self.nodes.to_file(nodes_geojson_path, driver="GeoJSON")
        self.edges.to_file(edges_geojson_path, driver="GeoJSON")

        # housekeeping
        process_run_time = time.perf_counter() - process_start_time
        logging.info(f"Finished converting geodataframes to GeoJSONs in {turn_seconds_into_minutes(process_run_time)}")

        return nodes_geojson_path, edges_geojson_path

    # now can convert the geojsons for nodes and edges into feature classes
    def convert_geojsons_to_feature_class(self, outputted_geojsons: tuple[str, str]):
        """
        Takes the street network geojson created by convert_geodataframes_to_geojson and turns them
        into geodatabase feature classes
        """
        logging.info("Converting GeoJSONs to GeoDatabase feature classes")
        process_start_time = time.perf_counter()

        # get the paths to the geojson out from the provided tuple
        nodes_geojson_path = outputted_geojsons[0]
        edges_geojson_path = outputted_geojsons[1]

        # just a reminder that the object has the desired fc paths as attributes
        # (self.nodes_fc_path and self.edges_fc_path)

        # convert nodes from geojson to feature class using arcpy
        logging.info("Converting nodes GeoJSON to feature class")
        arcpy.conversion.JSONToFeatures(in_json_file=nodes_geojson_path, out_features=self.nodes_fc_path,
                                        geometry_type="POINT")
        # do the same for edges
        logging.info("Converting edges GeoJSON to feature class")
        arcpy.conversion.JSONToFeatures(in_json_file=edges_geojson_path, out_features=self.edges_fc_path,
                                        geometry_type="POLYLINE")

        # housekeeping
        process_run_time = time.perf_counter() - process_start_time
        logging.info(f"Finished converting GeoJSONs to geodatabase feature classes in {turn_seconds_into_minutes(process_run_time)}")

        ### MIGHT WANT TO ADD BIT HERE TO DELETE THE geojsons? ###

    def map_street_network_to_feature_classes(self):
        """
        The only method needed to call for this feature class, maps the street network edges and
        nodes to feature classes.
        """
        # first check if cached GeoJSONs

        nodes_geojson_path = os.path.join(self.cache_folder.path, "nodes_fc.geojson")
        edges_geojson_path = os.path.join(self.cache_folder.path, "edges_fc.geojson")

        # in case where cached data exists
        if os.path.exists(nodes_geojson_path) and os.path.exists(edges_geojson_path) and not self.reset:

            logging.info("Cached GeoJSONs found, converting to feature classes")
            process_start_time = time.perf_counter()

            # just need to convert the geojsons to feature classes
            input_geojsons = (nodes_geojson_path, edges_geojson_path)
            self.convert_geojsons_to_feature_class(input_geojsons)

            # housekeeping
            process_run_time = time.perf_counter() - process_start_time
            logging.info(f"Mapped the street network to feature classes in {turn_seconds_into_minutes(process_run_time)}")

        else:
            logging.info("Cached data not available, mapping street network to feature classes")
            process_start_time = time.perf_counter()
            # first, calculate walk times while still a dataframe
            self.calculate_walk_times()

            # next, convert gdfs to geojsons and then geojsons to feature classes
            outputted_geojsons = self.convert_geodataframes_to_geojson()
            self.convert_geojsons_to_feature_class(outputted_geojsons)

            # housekeeping
            process_run_time = time.perf_counter() - process_start_time
            logging.info(f"Mapped the street network to feature classes in {turn_seconds_into_minutes(process_run_time)}")


class TransitNetwork:
    def __init__(self, geographic_scope, feature_dataset: FeatureDataset, reference_place_list: list[ReferencePlace],
                 modes: list = None, agencies_to_include:list[TransitAgency]=None, own_gtfs_data_paths:list[str]=None):
        """
        Transit network class for place
        
        Attributes:
            geographic_scope: GeographicScope | geographic scope of the transit network {"place_only", "msa", "csa"}
            feature_dataset: FeatureDataset | feature dataset where the transit network will be created
            reference_place_list: list[ReferencePlace] | list of reference places for the transit network
            modes: list | modes to be included in the transit network {"all", "bus", "heavy_rail", "light_rail",
            "regional_rail", "ferry", "gondola", "funicular", "trolleybus", "monorail"}
            (for more, see gtfs_tools.route_types documentation)

        **Methods:**
            get_transit_agencies_for_place: creates a list of transit agencies that serve place (see method doc)


        """
        self.geographic_scope = geographic_scope
        self.feature_dataset = feature_dataset
        self.reference_place_list = reference_place_list
    
        self.place_names = [reference_place.place_name for reference_place in reference_place_list]
        self.bound_box = [reference_place.bound_box for reference_place in reference_place_list]
        self.main_reference_place = reference_place_list[0]

        if self.main_reference_place.bound_box:
            self.geographic_scope = "bbox"

        self.snake_name = create_snake_name(self.main_reference_place)
        self.snake_name_with_scope = f"{self.snake_name}_{self.geographic_scope}"
        self.modes = modes
        # eventually this is what will be used to pass gtfs data to create network dataset
        self.gtfs_folders = None

        # link to cache folder
        self.cache_folder = CacheFolder(self.snake_name_with_scope)

        # if no agencies are specified, will default to all agencies that serve the place
        self.agencies_to_include = agencies_to_include
        if self.agencies_to_include is None:
            logging.info("a list of transit agencies to include was not specified so all agencies that serve the place"
                         " will be used")
            self.agencies_to_include = self.get_agencies_for_place()

        # adding the ability to bring your own gtfs data
        self.own_gtfs_data_paths = own_gtfs_data_paths
        if self.own_gtfs_data_paths is None:
            logging.info(f"Own GTFS data not provided, will query TransitLand API to get transit agencies that serve"
                         f"{self.main_reference_place.pretty_name}")

        # once the first three methods have been run (get_agencies_for_place, get_gtfs_for_transit_agencies,
        # unzip_gtfs_data), this dictionary will contain the paths of unzipped gtfs data available for place
        # for desired Agencies
        self.gtfs_folders = None
        self.agency_feed_valid_dates = {}

    # using requests instead of aihttp for this because only single request
    def get_agencies_for_place(self):
        """
        Takes a place name of format 'city, state, country'
        and returns a list of transit agencies that serve the place

        :return: list[TransitAgency] | list of transit agencies (TransitAgencyObjects) that serve the place
        """
        logging.info(f"Getting agencies that serve {self.main_reference_place.pretty_name}")
        # list of TransitAgency objects to be returned
        agencies_for_place = []

        # in case where using place (with geographic scope) rather than bounding box
        if self.main_reference_place.bound_box is None:

            # iterate through reference places to get the agencies that serve them
            for reference_place in self.reference_place_list:
                # transit land's API only requires the city name (although this seems stupid)
                place_short_name = reference_place.place_name.split(",")[0]

                transit_land_response = requests.get(f"https://transit.land/api/v2/rest/agencies?api_key="
                                                     f"{transit_land_api_key}"
                                                     f"&city_name={place_short_name}")
                transit_land_response.raise_for_status()

                # json containing the agencies
                transit_land_data = transit_land_response.json()

                # going through the agency dicts provided by the api and using them as kwargs for TransitAgency object
                for agency_data in transit_land_data["agencies"]:
                    # fill out TransitAgency objects using the data from the API
                    temp_agency = TransitAgency(**agency_data)
                    agencies_for_place.append(temp_agency)
                logging.info(f"Found {len(agencies_for_place)} agencies that serve {reference_place.pretty_name}"
                             f" {self.geographic_scope}")

        # in case where using bounding box
        else:
            # the bbox is concatenated into a string to be used in the query to transitland's API
            bbox_query_string = ",".join(self.main_reference_place.bound_box)
            transit_land_response = requests.get(f"https://transit.land/api/v2/rest/agencies?api_key="
                                                 f"{transit_land_api_key}"
                                                 f"&bbox={bbox_query_string}")
            transit_land_response.raise_for_status()
            transit_land_data = transit_land_response.json()

            for agency_data in transit_land_data["agencies"]:
                # fill out TransitAgency objects using the data from the API
                temp_agency = TransitAgency(**agency_data)
                agencies_for_place.append(temp_agency)

            logging.info(f"Found {len(agencies_for_place)} agencies that serve {self.main_reference_place.pretty_name}")

        # now can set self.agencies_that_serve_place
        return agencies_for_place

    def get_gtfs_for_transit_agencies(self):
        """
            Gets the latest static GTFS data for agencies desired.

            :param agencies_to_include: list[TransitAgency] | list of transit agencies to get GTFS data for
                (by default will be all agencies that serve the reference place)
            :return: gtfs_zip_folders | dict with TransitAgencies as keys and
             the (zipped) file names where the gtfs feeds are written as values.
            :return: agency_feed_valid_dates: dict{TransitAgency: {"last_updated": MMDDYYYY, "valid_until": MMDDYYYY}}
                | dictionary of the valid dates for each agency's feed and when it was last updated (because will have
                to deal with feeds that are not current or currently valid.
        """
        # if a list of agencies to include was not provided then by default will use every transit agency serving place
        logging.info(f"Getting GTFS data for {len(self.agencies_to_include)} transit agencies")
        # outputs for the method
        gtfs_zip_folders = {}


        # next, iterating through the list of desired agencies and getting their data
        for agency in self.agencies_to_include:
            # onestop_id (used to query for feed) is in feed_version["feed"]["onestop_id"]
            feed_version = agency.feed_version
            onestop_id = feed_version["feed"]["onestop_id"]
            transit_land_api_url = (f"https://transit.land/api/v2/rest/feeds/{onestop_id}/download_latest_feed_version"
                                    f"?api_key={transit_land_api_key}")

            # standard API query
            response = requests.get(transit_land_api_url)
            response.raise_for_status()

            # the file path where the zipped gtfs data will be saved to (yes complicated, but best for organization
            file_path = os.path.join(self.cache_folder.path, "gtfs_caches", f"{onestop_id}", "zipped_gtfs",
                                     f"{onestop_id}.zip")

            # set up folders in case they don't already exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # writing the content that was returned from transit land to a zip file
            with open(file_path, "wb") as gtfs_zipped_file:
                gtfs_zipped_file.write(response.content)

            # now need to see 1. when the feed was fetched (and that it's not too out of date) and 2. whether it's valid
            feed_version_query_response = requests.get(f"https://transit.land/api/v2/rest/feeds/{onestop_id}"
                                                       f"?api_key={transit_land_api_key}")
            feed_version_query_response.raise_for_status()
            feed_version_query_response_json = feed_version_query_response.json()

            # when the latest static feed was fetched by transit land
            latest_feed_fetch_date = (feed_version_query_response_json["feeds"][0]["feed_versions"]
                                      [0]["fetched_at"])
            # the latest date in the calendar that the data is valid for (can either extend or not use)
            latest_feed_valid_until = (feed_version_query_response_json["feeds"][0]["feed_versions"]
                                      [0]["latest_calendar_date"])

            # adding the zipped gtfs folder to the path dict
            gtfs_zip_folders[agency] = file_path
            # creating a dictionary with these dates needed for each agency
            self.agency_feed_valid_dates[agency] = {"last_updated": latest_feed_fetch_date,
                                               "valid_until": latest_feed_valid_until}


        logging.info("Successfully downloaded GTFS data for desired agencies")
        return gtfs_zip_folders

    def unzip_gtfs_data(self):
        """
        Unzips the GTFS data that was downloaded from Transit Land.
        :return: unzipped_gtfs_filepaths: dict{TransitAgency: path of unzipped gtfs folder}
        """
        agency_zip_folders = self.get_gtfs_for_transit_agencies()
        logging.info("Unzipping the downloaded GTFS data")
        unzipped_gtfs_filepaths = {}

        # go through each agency in the provided agency_zip_folders
        for agency in agency_zip_folders:

            # the path where each unzipped gtfs folder will be saved to
            onestop_id_directory = os.path.dirname(os.path.dirname(agency_zip_folders[agency]))
            unzipped_gtfs_filepath = os.path.join(onestop_id_directory,
                                                  "unzipped_gtfs")

            # set up folders in case they don't already exist
            os.makedirs(unzipped_gtfs_filepath, exist_ok=True)

            # using zipfile module to extract all .txt files provided
            with ZipFile(agency_zip_folders[agency], "r") as zipped_file:
                zipped_file.extractall(path=unzipped_gtfs_filepath)

            # adding the unzipped gtfs folder to the path dict
            unzipped_gtfs_filepaths[agency] = unzipped_gtfs_filepath

        # set the gtfs_folders attribute and return the unzipped folder paths
        self.gtfs_folders = unzipped_gtfs_filepaths
        logging.info("Successfully unzipped the downloaded GTFS data")

        return unzipped_gtfs_filepaths

    def check_whether_data_valid(self):
        """
        Checks whether the data is valid for the desired agencies.
        :return:
        """
        logging.info("Checking whether the downloaded data is still valid (and can therefore be used to create a "
                     "network dataset in ArcGIS)")

        # a dictionary that says for each agency if the downloaded
        still_valid_gtfs_data = {}

        for agency in self.agency_feed_valid_dates:
            # check if current date is past the "valid_until" date
            valid_until_date_iso = isodate.parse_date(self.agency_feed_valid_dates[agency]["valid_until"])
            current_date_iso = isodate.parse_date(datetime.now().strftime("%Y-%m-%d-%f"))

            if valid_until_date_iso <= current_date_iso:
                logging.warning(f"The data for {agency.agency_name} is no longer valid (valid until {valid_until_date_iso})")
            else:
                still_valid_gtfs_data[agency] = True

        logging.info(f"Data was valid for {len(still_valid_gtfs_data)}/{len(self.agency_feed_valid_dates)} agencies")
        return still_valid_gtfs_data

    def create_public_transit_data_model(self) -> None:
        """
        Creates a public transit data model in the feature dataset for this scenario using the valid GTFS data.
        :return:  None
        """


        # only using valid gtfs data to create network dataset because otherwise gets screwy
        valid_gtfs_data = self.check_whether_data_valid() # output of method is dict of transit agencies and validities
        logging.info(f"Creating public transit data model for {self.main_reference_place.pretty_name} using "
                     f"{len(valid_gtfs_data)} agencies")

        gtfs_folders_to_use = []
        for agency in valid_gtfs_data:
            if valid_gtfs_data[agency]:
                gtfs_folders_to_use.append(self.gtfs_folders[agency])

        # create a Public Transit Data Model using arcpy (using interpolate because some minor agencies have low quality
        # GTFS data, and while this is annoying, you  can just calculate arrival times using interpolate
        arcpy.transit.GTFSToPublicTransitDataModel(in_gtfs_folders=gtfs_folders_to_use,
                                                   target_feature_dataset=self.feature_dataset.path,
                                                   interpolate="INTERPOLATE", make_lve_shapes="MAKE_LVESHAPES")


    def connect_network_to_streets(self):
        # flesh out method
        edges_fc_path = os.path.join(self.feature_dataset.path, "edges_fc")
        arcpy.transit.ConnectPublicTransitDataModelToStreets(target_feature_dataset=self.feature_dataset.path,
                                                             in_streets_features=edges_fc_path)

class NetworkDataset:
    """
    Network dataset class for use in ArcGIS Pro.

    Attributes:
        feature_dataset (FeatureDataset): The feature dataset to create the network dataset in.
        network_type (str): The type of network dataset to create. Default is "walk_no_z".
        use_elevation (bool): Whether to use elevation in calculating bike or walk times. Default is False.
        reset (bool): Whether to reset the network dataset. Default is False.
        street_network (str): The path to the street network feature class.
        name (str): The name of the network dataset. By default will be the network type with "_nd" appended.
        path (str): The path to the network dataset (always C:\\...\\<feature_dataset_name>\\<network_dataset_name>.nd)

    """

    def __init__(self, feature_dataset: FeatureDataset, network_type: str = "walk_no_z", use_elevation=False,
                 reset=False):
        self.feature_dataset = feature_dataset
        self.network_type = network_type
        self.use_elevation = use_elevation
        self.reset = reset
        self.street_network = self.feature_dataset.street_network
        self.name = f"{self.network_type}_nd"
        self.path = os.path.join(self.feature_dataset.path, self.name)
        self.nodes_fc_path = os.path.join(self.feature_dataset.path, "nodes_fc")
        self.edges_fc_path = os.path.join(self.feature_dataset.path, "edges_fc")
        self.has_been_created = False

        # network_types module contains a dictionary containing templates and names for diff network types
        self.template_name = network_types.network_types_attributes[self.network_type]["network_dataset_template_name"]
        self.template_path = os.path.join(Path(__file__).parent, self.template_name)

        # check that network type exists and that a template for that network type exists
        if self.network_type not in network_types.network_types_attributes:
            raise Exception(f"Network type {self.network_type} does not exist "
                            f"(see network_types.py documentation for valid network types)")

        elif not os.path.exists(self.template_path):
            raise Exception(f"Template for network type {self.network_type} does not exist yet, sorry!")

        # check not trying to use elevation without elevation enabled street network
        if use_elevation and not self.street_network.elevation_enabled:
            raise Exception("Cannot use elevation")

    def create_network_dataset(self):
        """
        Creates network dataset of type (self.network type)
        :return
            Path of network dataset
        """
        process_start_time = time.perf_counter()
        logging.info(
            f"Creating {self.network_type} network dataset for {self.street_network.main_reference_place.pretty_name}")  

        # making sure network analyst checked out
        if not network_analyst_extension_checked_out:
            check_out_network_analyst_extension()

        # making sure not trying to obliviously create a network dataset if one with same name already exists
        if arcpy.Exists(self.path) and not self.reset:
            raise Exception(f"Cannot create new network dataset {self.name} because one already exists"
                            f"with that name at the desired path {self.path}")

        # main code block for the actual creating of the network dataset   
        try:
            if self.reset:
                arcpy.Delete_management(self.path)
                logging.info(
                    f"Existing network dataset {self.name} for {self.street_network.main_reference_place.pretty_name}"
                    f" {self.street_network.geographic_scope} already exists, deleting and creating new")  

            # check whether to use elevation in network dataset, raise error if no elevation data
            if self.use_elevation:
                # first need to check that the street network is actually elevation enabled
                if not self.street_network.elevation_enabled:
                    raise Exception("The street network provided does not have elevation data")

            # check that both nodes and edges feature classes exist in dataset
            if not arcpy.Exists(self.nodes_fc_path):
                raise Exception("Nodes feature class for street network does not exist in feature dataset")
            if not arcpy.Exists(self.edges_fc_path):
                raise Exception("Edges feature class for street network does not exist in feature dataset")

            arcpy.na.CreateNetworkDatasetFromTemplate(network_dataset_template=self.template_path,
                                                      output_feature_dataset=self.feature_dataset.path)
            logging.info("Successfully created walking network dataset from template")

            # mark that has been created
            self.has_been_created = True
            logging.info(f"Successfully created network dataset in {time.perf_counter() - process_start_time} seconds")
            return self.path

        finally:
            # have to check extension back in when done running
            arcpy.CheckInExtension("Network")

    def build_network_dataset(self, rebuild=False):
        """
        Builds the network dataset that has been created
        :param rebuild: bool (True if rebuilding network_dataset desired)
        :return: Network dataset path: str
        """
        process_start_time = time.perf_counter()
        logging.info("Building network dataset")
        # checking out network analyst extension if not already
        if not network_analyst_extension_checked_out:
            check_out_network_analyst_extension()

        # making sure that the network dataset actually exists!
        if not self.has_been_created:
            raise Exception("The network dataset has not been created yet!")
        arcpy.na.BuildNetwork(self.path)

        # checking extensions out has weird behavior so always need these checks to see that it isn't oddly out/in
        if network_analyst_extension_checked_out:
            check_network_analyst_extension_back_in()
        process_run_time = time.perf_counter() - process_start_time
        logging.info(f"Network dataset successfully built in {turn_seconds_into_minutes(process_run_time)}")

        # save the gdb?
        self.feature_dataset.gdb.save_gdb()
        return self.path

# makig sure that network analyst extension is checked back in after done running
if network_analyst_extension_checked_out:
    check_network_analyst_extension_back_in()


