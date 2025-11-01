import os
import sys
from pathlib import Path

from arcgis import features

# Setting up python environment/making sure env points to extensions correctly
arcgis_bin = r"C:\Program Files\ArcGIS\Pro\bin"
arcgis_extensions = r"C:\Program Files\ArcGIS\Pro\bin\Extensions"
os.environ["PATH"] = arcgis_bin + os.pathsep + arcgis_extensions + os.pathsep + os.environ.get("PATH", "")

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

# local module(s)
import transit_data_for_arcgis
import network_types
from general_tools import create_snake_name
from general_tools import ReferencePlace

# making sure that using windows because otherwise cannot use arcpy and ArcGIS
if platform.system() != "Windows":
    raise OSError("Cannot run this module because not using Windows. ArcGIS and ArcPy require Windows")

# logging setup
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# osmnx setup
ox.settings.timeout = 1000
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
        Attributes:
            name: str, project_location: str
            (path to directory where project should exist), and reset: bool (clears project if true).
            By default, the project will be within the same directory as this module.
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
        self.arcObject = arcpy.mp.ArcGISProject(self.path)

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
            self.arcObject = arcpy.mp.ArcGISProject(self.path)
        else:
            os.makedirs(self.project_dir_path)
            arcgis_project_template.saveACopy(self.path)
            self.arcObject = arcpy.mp.ArcGISProject(self.path)
        # set project as current workspace
        arcpy.env.workspace = self.project_dir_path

        return self.path
    # add more methods here. Add map, etc.


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

    def __init__(self, network_snake_name):
        assert not re.match("[A-Z]", network_snake_name) and not re.match(" ", network_snake_name), \
            "Argument network_snake_name must be a properly formed snake name (no spaces, lowercase only)"
        self.network_snake_name = network_snake_name
        self.env_dir_path = Path(__file__).parent
        self.path = os.path.join(self.env_dir_path, "place_caches", f"{self.network_snake_name}_cache")

    def check_if_cache_folder_exists(self):
        """ Returns True if cache folder already exists for the city."""
        if os.path.exists(self.path):
            return True
        else:
            return False

    def set_up_cache_folder(self):
        """Return True if there is already a cache folder for city. If not, creates one."""
        if os.path.exists(self.path):
            raise Exception(f"There is already a cache folder for {self.network_snake_name}")
        else:
            os.makedirs(self.path)

    def reset_cache_folder(self):
        # completely reset the cache folder for the city
        if not os.path.exists(self.path):
            raise Exception(f"Cannot reset the cache folder for {self.network_snake_name} "
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
        city_name (str): Name of the city

        network_type (str): type of network being created {"walk_no_z", "walk_z", "bike_no_z", "bike_z",
        "transit_no_z", "transit_z", "drive", "transit_plus_biking_no_z", "transit_plus_biking_z"}
        bound_box (list): Bounding box of the city
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

    def __init__(self, city_name, network_type="walk_no_z", bound_box=None):
        #### need to make __init__ method cleaner ####
        self.city_name = city_name
        self.network_type = network_type
        self.bound_box = bound_box  # need to figure out how I want to deal with bounding boxes

        # create snake name for StreetNetwork
        self.reference_place = ReferencePlace(place_name=city_name, bound_box=bound_box)
        self.snake_name = create_snake_name(self.reference_place)

        self.cache_folder = CacheFolder(self.snake_name)  # cache folder for this street network!
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
            if self.bound_box is None:
                network_graph = ox.graph_from_place(self.city_name, network_type=self.osmnx_type, retain_all=True)
                network_nodes, network_edges = ox.graph_to_gdfs(network_graph, nodes=True, edges=True)

            # if using bound box, getting OSM Data
            else:
                network_graph = ox.graph_from_bbox(self.bound_box, network_type=self.osmnx_type, retain_all=True)
                network_nodes, network_edges = ox.graph_to_gdfs(network_graph, nodes=True, edges=True)

        # otherwise using cached data
        else:
            logging.info("Using cached street network")
            network_graph = self.graph_cache.read_cache_data()
            network_nodes, network_edges = self.nodes_cache.read_cache_data(), self.edges_cache.read_cache_data()

        # just because I want to keep track of everything
        if timer_on:
            logging.info(f"Got street network from OSM in {time.perf_counter() - process_start_time} seconds")
            logging.info(f"Street network for {self.city_name} had {len(network_nodes)} nodes and "
                         f"{len(network_edges)} edges")

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

    def __init__(self, street_network: StreetNetwork, concurrent_requests_desired:int=300, reset=False):
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
                # 45 or 60s might be better
                async with async_session.get(usgs_url, params=params, timeout=60) as response:

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
                          f"{time_elapsed_in_minutes_and_seconds['minutes']:.1f} minutes:"
                          f"{time_elapsed_in_minutes_and_seconds['seconds']:.1f} seconds / "
                          f"{estimated_total_time_in_minutes_and_seconds['minutes']:.1f} minutes:"
                          f"{estimated_total_time_in_minutes_and_seconds['seconds']:.1f} seconds"
                          f" (elapsed time/estimated total time). Rate = {querying_rate} nodes/second",
                          end="", flush=True)

                    logging.warning(f"Couldn't get elevation for {self.nodes_without_elevation}/{total_nodes} nodes.")

                    # now that have elevation can return the index of the node (from itterrowing them) and its elevation
                    return idx, elevation

            # in case of exception, just returning None as the elevation for the node.
            except Exception as e:
                self.nodes_without_elevation += 1
                return idx, None

        logging.warning(f"Couldn't get eleavation for {self.nodes_without_elevation}/{total_nodes} nodes")
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
        connector = aiohttp.TCPConnector(limit=300, limit_per_host=300, ttl_dns_cache=300, enable_cleanup_closed=True,
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

        logging.info(f"Adding elevation to the street network for {self.street_network.city_name}")

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
        logging.info(f"Adding grades to edges for {self.street_network.snake_name} street network")

        # doubled check street network is elevation enabled
        if not self.street_network.elevation_enabled:
            raise Exception("Cannot add grades to edges, elevation data not available for nodes")

        # iterating throuuh all edges in the streetnetwork graph
        for start_node, end_node, key, data in self.street_network_graph.edges(keys=True, data=True):
            # elevation for start and end nodes (start_node and end_node respectively)
            start_elevation = self.street_network_graph.nodes[start_node].get("elevation")
            end_elevation = self.street_network_graph.nodes[end_node].get("elevation")
            edge_length = data.get("length", 0)

            # only caclulate grade if elevation for start and end node available
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
        logging.info(f"Successfully added grades to edges in {process_run_time:.2f} seconds")


### need to review GeoDatabase and ArcProject classes code to make sure no errors
class GeoDatabase:
    def __init__(self, arcgis_project: ArcProject, street_network: StreetNetwork):
        self.project = arcgis_project
        self.street_network = street_network
        self.project_file_path = arcgis_project.path
        self.gdb_path = os.path.join(self.project.project_dir_path, f"{self.street_network.snake_name}.gdb")

    def set_up_gdb(self, reset=False):
        """ Creates geodatabase if one does not exist, and resets it if reset desired."""
        # making sure not trying to set up gdb before project
        if not arcpy.Exists(self.project.path):
            raise FileNotFoundError(f"Cannot set up geodatabase in the the provided project {self.project.name}"
                                    f"because it doesn't exist")

        if not reset and not arcpy.Exists(self.gdb_path):
            logging.info("No existing geodatabase found, creating new geodatabase")
            arcpy.management.CreateFileGDB(self.project.project_dir_path, self.street_network.snake_name)

        elif arcpy.Exists(self.gdb_path):
            arcpy.Delete_management(self.gdb_path)
            arcpy.overwriteOutput = True
            arcpy.management.CreateFileGDB(self.project.project_dir_path, self.street_network.snake_name)

        else:
            arcpy.management.CreateFileGDB(self.project.project_dir_path, self.street_network.snake_name)

        # modifying current arcpy env
        arcpy.env.workspace = self.gdb_path
        # just setting current gdb as default. need to debug to figure out why have to do this way
        self.project.arcObject.defaultGeodatabase = self.gdb_path

    def save_gdb(self):
        # using try to avoid errors in case of file lock
        try:
            self.project.arcObject.save()
            self.project.arcObject.defaultGeodatabase = self.gdb_path
            logging.info("Project saved successfully")
        except OSError as e:
            logging.warning(f"Could not save project (file may locked or project open: {e}")


class FeatureDataset:  # add scenario_id so can do multiple scenarios of same network type
    def __init__(self, gdb: GeoDatabase, street_network: StreetNetwork, network_type: str = "walk_no_z", reset=False):
        self.gdb = gdb
        self.street_network = street_network
        self.network_type = network_type
        self.reset = reset
        self.name = f"{self.network_type}_fd"
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
                                              spatial_reference=arcpy.SpatialReference(4326)
                                              )
        logging.info("Feature dataset successfully created")

    def reset_feature_dataset(self):
        # just a method to reset the desired feature dataset
        if not arcpy.Exists(self.path):
            raise Exception(f"Cannot delete feature dataset at {self.path} because it doesn't exist")
        else:
            arcpy.Delete_management(self.path)
            arcpy.management.CreateFeatureDataset(self.gdb.gdb_path, self.name,
                                                  spatial_reference=arcpy.SpatialReference(4326)
                                                  )


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

        # error handling for using elevation when creating feature classes for street network
        if "z" not in self.street_network.network_nodes and self.use_elevation:
            raise Exception("Cannot use elevation because input StreetNetwork object has no z values")

    # method to set up empty feature classes for street network which data can be copied to
    def create_empty_feature_classes(self):
        logging.info("Creating empty feature classes for street network")

        # in case at least one of the two feature classes for street network (nodes or edges) does not exist or reset
        if (not (arcpy.Exists(self.nodes_fc_path) and arcpy.Exists(self.edges_fc_path))) or self.reset:
            # possible case here is that one exists but not the other so need to overwrite to be on
            arcpy.env.overwriteOutput = True

            # if want to use elevation for network dataset (already checked to make sure data has necessary z values)
            if self.use_elevation:
                arcpy.management.CreateFeatureclass(self.feature_dataset.path, "nodes_fc", geometry_type="POINT",
                                                    spatial_reference=arcpy.SpatialReference(4326),
                                                    has_z="ENABLED")
                arcpy.management.CreateFeatureclass(self.feature_dataset.path, "edges_fc", geometry_type="POLYLINE",
                                                    spatial_reference=arcpy.SpatialReference(4326),
                                                    )

            # not using elevation
            else:
                arcpy.management.CreateFeatureclass(self.feature_dataset.path, "nodes_fc", geometry_type="POINT",
                                                    spatial_reference=arcpy.SpatialReference(4326))
                arcpy.management.CreateFeatureclass(self.feature_dataset.path, "edges_fc", geometry_type="POLYLINE",
                                                    spatial_reference=arcpy.SpatialReference(4326))
        # both street FCs exist and reset not desired, so do nothing :)
        else:
            logging.info("Existing street network feature classes found and will be used")
            pass

        # method outputs
        logging.info("Successfully created empty feature classes")
        return self.nodes_fc_path, self.edges_fc_path

    # method uses arcpy cursor to add OSM gdfs data to feature classes, need to fix (attr fields empty except geometry?)
    def add_street_network_data_to_feature_classes(self):
        """
        Method takes street network geodata frames for nodes and edges and adds them to feature classes
        :return: None
        """
        process_start_time = time.perf_counter()
        logging.info("Mapping street network data to feature classes")

        # first checking that empty feature classes exist
        if not (arcpy.Exists(self.nodes_fc_path) and arcpy.Exists(self.edges_fc_path)):
            raise Exception(f"Cannot add street network data because either nodes "
                            f"{self.nodes_fc_path} or {self.edges_fc_path} doesn't exist (create empty first)")

        # now mapping data to feature classes for network for non-geometry fields (have to handle geometry separately)
        # this section just makes sure each field mapped to feature classes gets the write type
        for col in self.nodes.columns:
            # for nodes
            if col != "geometry":
                if col == "osmid":  # need to make osmid a str because for large networks, gets too big
                    field_type = "TEXT"
                    arcpy.management.AddField(self.nodes_fc_path, col, field_type, field_length=2048)
                elif self.nodes[col].dtype in ["int64", "int32"]:
                    field_type = "LONG"
                    arcpy.management.AddField(self.nodes_fc_path, col, field_type)
                elif self.nodes[col].dtype in ["float64", "float32"]:
                    field_type = "DOUBLE"
                    arcpy.management.AddField(self.nodes_fc_path, col, field_type)
                else:
                    field_type = "TEXT"
                    arcpy.management.AddField(self.nodes_fc_path, col, field_type, field_length=255)

        # now mapping non geometry fields for edges
        for col in self.edges.columns:
            if col != "geometry":
                if col == "osmid":  # need to make osmid a str because for large networks, gets too big
                    field_type = "TEXT"
                    arcpy.management.AddField(self.edges_fc_path, col, field_type, field_length=1024)
                    field_type = "TEXT"
                elif self.edges[col].dtype in ["int64", "int32"]:
                    field_type = "LONG"
                    arcpy.management.AddField(self.edges_fc_path, col, field_type)
                elif self.edges[col].dtype in ["float64", "float32"]:
                    field_type = "DOUBLE"
                    arcpy.management.AddField(self.edges_fc_path, col, field_type)
                else:
                    field_type = "TEXT"
                    arcpy.management.AddField(self.edges_fc_path, col, field_type, field_length=255)

        # if using elevation then need to make xyz points
        if self.use_elevation:
            if "grade" not in self.edges.columns:
                raise Exception("Cannot use elevation because edges do not have grades")

            # creating node attribute fields (
            node_fields = ["SHAPE@XYZ"] + [col for col in self.nodes.columns if col not in ["geometry", "x", "y", "z"]]
            with arcpy.da.InsertCursor(self.nodes_fc_path, node_fields) as cursor:
                for idx, row in self.nodes.iterrows():
                    geom = (row.geometry.x, row.geometry.y, row["z"] if "z" in row and row["z"] is not None else 0)
                    values = [geom] + [row[col] for col in self.nodes.columns if col not in ["geometry", "x", "y", "z"]]
                    cursor.insertRow(values)

            # creating edges feature classes
            edge_fields = ["SHAPE@"] + [col for col in self.edges.columns if col != "geometry"]
            with arcpy.da.InsertCursor(self.edges_fc_path, edge_fields) as cursor:
                for idx, row in self.edges.iterrows():
                    coords = list(row.geometry.coords)
                    array = arcpy.Array([arcpy.Point(x, y) for x, y in coords])
                    polyline = arcpy.Polyline(array, arcpy.SpatialReference(4326))
                    values = [polyline] + [
                        ', '.join(map(str, row[col])) if isinstance(row[col], list) else row[col]
                        for col in self.edges.columns if col != "geometry"
                    ]
                    cursor.insertRow(values)

            # adding fields for network dataset creations
            logging.info("Adding necessary fields for edges")

            # check if grade data exists, if so calculate graded walk time for against and along directions
            if "grade" in self.edges.columns:

                # walk time from to (along) field
                arcpy.management.AddField(self.edges_fc_path, "walk_time_graded_FT", "DOUBLE")
                arcpy.management.CalculateField(
                    self.edges_fc_path,
                    "walk_time_graded_FT",
                    # tobbler's hiking function for walking speed as function of grade
                    "!Shape.length@meters! / (100 * math.exp(-3.5 * abs(!grade! / 100 + 0.05)))",
                    "PYTHON3")

                # walk time to-from (against) field
                arcpy.management.AddField(self.edges_fc_path, "walk_time_graded_TF", "DOUBLE")
                arcpy.management.CalculateField(
                    self.edges_fc_path,
                    "walk_time_graded_TF",
                    "!Shape.length@meters! / (100 * math.exp(-3.5 * abs((-!grade!) / 100 + 0.05)))",
                    "PYTHON3")

            # if no grade data, calculate walk time without grade (same as if on flat ground/not using grade
            else:
                arcpy.management.AddField(self.edges_fc_path, "walk_time_graded_FT", "DOUBLE")
                arcpy.management.CalculateField(
                    self.edges_fc_path,
                    "walk_time_graded_FT",
                    # tobbler's hiking function for walking speed as function of grade
                    "!Shape.length@meters! / 85",
                    "PYTHON3"
                )

                # walk time to-from (against)
                arcpy.management.AddField(self.edges_fc_path, "walk_time_graded_TF", "DOUBLE")
                arcpy.management.CalculateField(
                    self.edges_fc_path,
                    "walk_time_graded_TF",
                    "!Shape.length@meters! / 85",
                    "PYTHON3")

            # making sure no multipart edges (had to change to updateCursor because was shadowing insert cursor before)
            with arcpy.da.UpdateCursor(self.edges_fc_path, ["SHAPE@"]) as update_cursor:
                broken_rows = []
                for row_idx, row in enumerate(update_cursor):
                    if row[0] is not None and row[0].isMultipart:
                        logging.warning("Multipart geometry detected!")
                        print(f"row_idx: {row_idx}, row:{row}, row[0]:{row[0]}")
                        break

                    # is true iff the row is empty!
                    if row[0] is None:
                        # delete empty row so doesn't cause problems with integrate function
                        print(f"row_idx: {row_idx}, row:{row}")
                        broken_rows.append(row)
                        update_cursor.deleteRow()
                        print(row in update_cursor)

            # integrate to snap nearby vertices
            arcpy.management.Integrate(self.edges_fc_path, "0.1 Meters")
            process_runtime = time.perf_counter() - process_start_time
            logging.info(f"Created Feature Classes from osmnx nodes and edges in {process_runtime} seconds")
            return self.nodes_fc_path, self.edges_fc_path

        else:  # in case where elevation not enabled
            # attribute table fields for nodes
            node_fields = ["SHAPE@XY"] + [col for col in self.nodes.columns if col not in ["geometry", "x", "y"]]
            with arcpy.da.InsertCursor(self.nodes_fc_path, node_fields) as cursor:
                for idx, row in self.nodes.iterrows():
                    geom = (row.geometry.x, row.geometry.y)
                    values = [geom] + [row[col] for col in self.nodes.columns if col not in ["geometry", "x", "y"]]
                    cursor.insertRow(values)

            # attribute table fields for edges                                                                           # left here because once grade added will need to change fields
            edge_fields = ["SHAPE@"] + [col for col in self.edges.columns if col != "geometry"]
            with arcpy.da.InsertCursor(self.edges_fc_path, edge_fields) as cursor:
                for idx, row in self.edges.iterrows():
                    coords = list(row.geometry.coords)
                    array = arcpy.Array([arcpy.Point(x, y) for x, y in coords])
                    polyline = arcpy.Polyline(array, arcpy.SpatialReference(4326))
                    values = [polyline] + [
                        ', '.join(map(str, row[col])) if isinstance(row[col], list) else row[col]
                        for col in self.edges.columns if col != "geometry"
                    ]
                    cursor.insertRow(values)

            # adding fields for network dataset creations
            logging.info("Adding necessary fields for edges")
            arcpy.management.AddField(self.edges_fc_path, "walk_time", "DOUBLE")
            arcpy.management.CalculateField(self.edges_fc_path, "walk_time",
                                            "!Shape.length@meters! / 85",
                                            "PYTHON3")

            # walk time with grade fields

            # making sure no multipart edges
            with arcpy.da.SearchCursor(self.edges_fc_path, ["SHAPE@"]) as cursor:
                for row in cursor:
                    if row[0] is not None and row[0].isMultipart:
                        logging.warning("Multipart geometry detected!")
                        break

            # integrate to snap nearby vertices
            arcpy.management.Integrate(self.edges_fc_path, "0.1 Meters")
            process_runtime = time.perf_counter() - process_start_time
            logging.info(f"Created Feature Classes from osmnx nodes and edges in {process_runtime} seconds")
            return self.nodes_fc_path, self.edges_fc_path

    def save_street_feature_classes_to_shapefile(self):
        """
        Saves the street feature classes created as is (whether fields created or not) to shapefile in cache folder for
        street network.
        Returns:
             (path of nodes shapefile, path of edges shapefile).
        """
        # make sure not trying to erroneously save non-existent feature classes
        assert arcpy.Exists(self.nodes_fc_path) and arcpy.Exists(self.edges_fc_path), \
            "The street feature classes cannot be saved because they have not yet been created"

        arcpy.env.overwriteOutput = True
        arcpy.conversion.FeatureClassToShapefile("nodes_fc", self.street_network.cache_folder.path)
        arcpy.conversion.FeatureClassToShapefile("edges_fc", self.street_network.cache_folder.path)
        logging.info(f"Nodes and edges feature classes succesfully saved as shapefiles in cache folder "
                     f"{self.street_network.cache_folder}")
        arcpy.env.overwriteOutput = False  # setting overwrite output back to false so no accidental overwriting

class TransitNetwork:
    def __init__(self, feature_dataset: FeatureDataset, place_name: str = None, bound_box: list = None,
                 modes: list = None):
        """
        Transit network class for place
        :param place_name: str | name of the place
        :param modes: list | modes to be included in the transit network {"all", "bus", "heavy_rail", "light_rail",
        "regional_rail", "ferry", "gondola", "funicular", "trolleybus", "monorail"}
        (for more, see gtfs_tools.route_types documentation)

        **Methods:**
            get_transit_agencies_for_place: creates a list of transit agencies that serve place (see method doc)


        """
        self.feature_dataset = feature_dataset
        self.place_name = place_name
        self.bound_box = bound_box

        self.transit_agencies = []
        self.modes = modes
        self.gtfs_folders = None

    def get_transit_agencies_for_place(self, geographic_scope: str, mode):
        """

        :param geographic_scope: what geographic scope to get transit agencies for. If using place name rather than
        bounding box: {"place", "adjacent" (place plus all immediately adjacent places),
        :param "network_extent" (specified
        place plus all places served by any transit agency that also serves specified place) "msa", "csa"}
        :return:
        """
        pass

    def get_gtfs_data(self, list_of_agencies):
        pass

    def create_transit_feature_classes(self):
        # flesh out method
        arcpy.transit.GTFSToPublicTransitDataModel(in_gtfs_folders=self.gtfs_folders,
                                                   target_feature_dataset=self.feature_dataset.path,
                                                   make_lve_shapes="MAKE_LVESHAPES")

    def connect_network_to_streets(self):
        # flesh out method
        arcpy.transit.ConnectPublicTransitDataModelToStreets(target_feature_dataset=self.feature_dataset.path,
                                                             in_streets_features=self.feature_dataset.street_network)


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
        path (str): The path to the network dataset (always C:\...\<feature_dataset_name>\<network_dataset_name>.nd)

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
            f"Creating {self.network_type} network dataset for {self.street_network.snake_name}")  # change so looks better (use short name rather than snake name)

        # making sure network analyst checked out
        if not network_analyst_extension_checked_out:
            check_out_network_analyst_extension()

        # making sure not trying to obliviously create a network dataset if one with same name already exists
        if arcpy.Exists(self.path) and not self.reset:
            raise Exception(f"Cannot create new network dataset {self.name} because one already exists"
                            f"with that name at the desired path {self.path}")

        # main code block for the actual creating of the network dataset                                                   # currently starting with just walking network dataset but will eventually add support for transit (+ biking?)
        try:
            if self.reset:
                arcpy.Delete_management(self.path)
                logging.info(
                    f"Existing network dataset {self.name} for {self.street_network.snake_name}")  # again replace snake name with short name

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
            logging.info(
                f"Successfully created network dataset in {time.perf_counter() - process_start_time} seconds")
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
        logging.info(f"Network dataset successfully built in {process_run_time} second")
        return self.path


# makig sure that network analyst extension is checked back in after done running
if network_analyst_extension_checked_out:
    check_network_analyst_extension_back_in()


