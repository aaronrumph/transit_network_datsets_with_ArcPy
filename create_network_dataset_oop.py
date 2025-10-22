# Currently in process of rewriting my code to use classes to make my code more readable
# Going to add docstrings for every class, method, and function so that can actually publish as package
# and have be understandable. Also adding better exception handling.

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
import osmnx as ox                   # used for getting streets data
import geopandas as gpd
import networkx as nx
import config
import time                          # used for checking runtimes of functions/methods
import requests                      # used for USGS API querying
import multiprocessing as mp         # used for querying in bulk
from itertools import repeat
import re
import platform

# making sure that using windows because otherwise cannot use arcpy and ArcGIS
if platform.system() != "Windows":
    raise OSError("Cannot run this module because not using Windows. ArcGIS and ArcPy require Windows")

# logging setup
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
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
        self.arcObject = None

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

def generic_get_elevation_function(process_start_time, total_nodes, idx, x_y: tuple):
    # this is the generic function for getting elevation from API for a point x_Y
    usgs_url = r"https://epqs.nationalmap.gov/v1/json"
    lon = x_y[0]
    lat = x_y[1]
    parameters_for_usgs_api = {"x": lon, "y": lat, "units": "Meters", "wkid": 4326}

    try:
        response = requests.get(usgs_url, params=parameters_for_usgs_api, timeout=10)
        response.raise_for_status()
        data = response.json()
        elevation = data.get('value')

        if elevation == -1000000:  # no data available (-1000000 is default for no value)
            elevation = None

        # Fixed node counter
        with shared_lock:
            shared_counter.value += 1
            current = shared_counter.value
        time_since_start = time.perf_counter() - process_start_time
        print(f"\rElevation: {current}/{total_nodes} nodes | Time: {time_since_start:.1f}s / "
              f"{(total_nodes / 7.8):.1f}s", end="", flush=True)

        return elevation

    except Exception as e:
        logging.warning(f"Error getting elevation for node {idx}: {e}")
        return None

# functions for dealing with checking network analyst extension in and out
def check_out_network_analyst_extension():
    """
    Checks out network analyst extension
    :return: True - successful, exception if False
    """
    # need to check that network analyst extension is actually available to use, and then check it out
    if arcpy.CheckExtension("network") == "Available":
        arcpy.CheckOutExtension("network")
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

# make snake name from place
def create_snake_name(reference_place:dict):
    """
    :param reference_place: dict {"place_name": str, ("bound_box": [str,str,str,str])}
    :return: Snake name for Place
    """
    if "bound_box" in reference_place:  ### FIX THIS SO THAT FILE NAMES NOT TOO LONG ###
        snake_name = ("_".join(part.strip().lower() for part in reference_place["place_name"].split(",")) +
                           "_".join(bound_coord for bound_coord in reference_place["bound_box"])
        )
    else:
        snake_name = "_".join(part.strip().lower() for part in reference_place["place_name"].split(","))

    return snake_name


class CacheFolder:
    """ Class for cache folder, takes param
        network_snake_name: str (MUST be in snake case) which will be the name of cache folder
    """
    def __init__(self, network_snake_name):
        assert not re.match("[A-Z]", network_snake_name) and not re.match(" ", network_snake_name), \
            "Argument network_snake_name must be a properly formed snake name (no spaces, lowercase only)"
        self.network_snake_name = network_snake_name
        self.env_dir_path = Path(__file__).parent
        self.path = os.path.join(self.env_dir_path, f"{self.network_snake_name}_cache")
    
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
                            f"because nos uch folder exists"
            )
        else:
            os.makedirs(self.path, exist_ok=True)

# simple Cache class with obvious methods (read, write, check if exists)
class Cache:
    def __init__(self, cache_folder:CacheFolder, cache_name):
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

    def __init__(self, city_name, network_type="walk", use_cache=True, bound_box=None):
        #### need to make __init__ method cleaner ####
        self.city_name = city_name
        self.use_cache = use_cache   # in case don't want to use cache for street network or want to reset cache
        self.type = network_type   # {“all”, “all_public”, “bike”, “drive”, “drive_service”, “walk”}
        self.bound_box = bound_box  # need to figure out how I want to deal with bounding boxes

        if self.bound_box is not None:  ### FIX THIS SO THAT FILE NAMES NOT TOO LONG ###
            self.snake_name =  (self.type + "_".join(part.strip().lower() for part in self.city_name.split(",")) +
                                "_".join(bound_coord for bound_coord in self.bound_box)
            )
        else:
            self.snake_name =  self.type + "_".join(part.strip().lower() for part in self.city_name.split(","))

        self.cache_folder = CacheFolder(self.snake_name) # cache folder for this street network!

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


    def get_street_network_from_osm(self, timer_on=True):
        # this method does the brunt of the work for the class
        process_start_time = time.perf_counter()
        logging.info("Getting street network")

        # first if not using cache or no cache data
        if (not self.use_cache) or (not self.graph_cache.check_if_cache_already_exists()):
            logging.info("Getting street network from OSM")

            # if using city, getting OSM data if not using cache or if no cache exists
            if self.bound_box is None:
                network_graph = ox.graph_from_place(self.city_name, network_type=self.type, retain_all=True)
                network_nodes, network_edges = ox.graph_to_gdfs(network_graph, nodes=True, edges=True)

            # if using bound box, getting OSM Data
            else:
                network_graph = ox.graph_from_bbox(self.bound_box, network_type=self.type, retain_all=True)
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

# ElevationMapper class adds elevation to nodes and grades to edges (coming soon?) using USGS EPQS API
class ElevationMapper:
    """ WARNINGS!:
            1. Adding elevation takes forever, ~130ms per node with 14 threads (1.8s per node per thread!)
                because the API is so slow
            2. Uses multiprocessing (scary) so use the "if __name__ == __main__" guard!!!!!
        Attrbitutes:
            street_network | StreetNetwork obj, threads_available | int, reset | bool
    """
    def __init__(self, street_network:StreetNetwork, threads_available=mp.cpu_count(), reset=False):
        self.street_network = street_network
        self.threads_available = threads_available
        self.reset = reset
        (self.street_network_graph, self.street_network_nodes, self.street_network_edges) = (
            street_network.network_graph, street_network.network_nodes, street_network.network_edges
        )
        self.node_counter = 0
        self.elevation_cache = Cache(street_network.cache_folder, "elevation")

    # using multiprocessing to get elevation data if no cache exists
    def add_elevation_data_to_nodes(self):
        logging.info("Adding elevation to nodes")

        # in case no elevation data exists
        if not self.elevation_cache.check_if_cache_already_exists() or self.reset:
            logging.info("Getting elevation data from USGS EPQS API")
    
            number_of_nodes_total = len(self.street_network_nodes)
            idx_args_for_elevation = [] # only useful for tracking progress
            x_y_args_for_elevation = [] # list of tuples of x,y values

            # making lists of desire arguments so can use .starmap()
            for idx, (node_idx, row) in enumerate(self.street_network_nodes.iterrows()):
                idx_args_for_elevation.append(idx)
                x_y_pair = (row["x"], row["y"])
                x_y_args_for_elevation.append(x_y_pair)

            shared_counter = mp.Value("i", 0)  # 'i' for integer
            lock = mp.Lock()
            process_start_time = time.perf_counter()

            # using multiprocessing to send as many API requests at once as possible
            with mp.Pool(processes=self.threads_available,
            initializer=init_worker,
            initargs=(shared_counter, lock)) as pooled_threads:
                z_values = pooled_threads.starmap(generic_get_elevation_function,
                    zip(repeat(process_start_time), repeat(number_of_nodes_total),
                        idx_args_for_elevation, x_y_args_for_elevation
                    )
                )
            pooled_threads.close()

            self.street_network_nodes["z"] = z_values
            self.elevation_cache.write_cache_data(self.street_network_nodes["z"].to_dict())

            # updating street network attributes
            self.street_network.network_nodes["z"] = self.street_network_nodes # updating input StreetNetwork object
            self.street_network.elevation_enabled = True
            
            # outputs for method
            run_time = time.perf_counter() - process_start_time
            logging.info(f"Got elevation data for {len(self.street_network_nodes)} nodes in {run_time} seconds")
            return self.street_network_nodes

        else: # if elevation data already cached
            logging.info("Using cached elevation data")
            z_values = self.elevation_cache.read_cache_data()   # loading elevation data from cache

            # check if the nodes gdf already has elevation
            if "z" not in self.street_network_nodes.columns:
                self.street_network_nodes["z"] = self.street_network_nodes.index.map(z_values)
                # updating input StreetNetwork objects
                self.street_network.network_nodes["z"] = self.street_network_nodes.index.map(z_values)
                logging.info("Successfully added elevation data to nodes from cache")
                self.street_network.elevation_enabled = True
            else:
                logging.info("Nodes gdf already has z values")
            # method output
            return self.street_network

    ### need to calculate grades for edges so can make walking time vary with slope ###
    def add_grades_to_edges(self):
        pass

### need to review GeoDatabase and ArcProject classes code to make sure no errors
class GeoDatabase:
    def __init__(self, arcgis_project:ArcProject, street_network:StreetNetwork, reset=True):
        self.project = arcgis_project
        self.street_network = street_network
        self.project_file_path = arcgis_project.path
        self.reset = reset
        self.gdb_path = os.path.join(self.project.project_dir_path, f"{self.street_network.snake_name}.gdb")


    def set_up_gdb(self):
        """ Creates geodatabase if one does not exist, and resets it if reset desired."""
        if not self.reset:
            pass

        elif arcpy.Exists(self.gdb_path):
            arcpy.Delete_management(self.gdb_path)
            arcpy.overwriteOutput = True
            arcpy.management.CreateFileGDB(self.project.project_dir_path, self.street_network.snake_name)
        else:
            arcpy.management.CreateFileGDB(self.project.project_dir_path, self.street_network.snake_name)
        
        arcpy.env.workspace = self.gdb_path
        self.project_file_path.defaultGeodatabase = self.gdb_path
    
    def save_gdb(self):
        # using try to avoid errors in case of file lock
        try:
            self.project.arcObject.save()
            logging.info("Project saved successfully")
        except OSError as e:
            logging.warning(f"Could not save project (file may locked or project open: {e}")


class FeatureDataset:
    def __init__(self, gdb:GeoDatabase, street_network:StreetNetwork, network_type:str = "walking", reset=False):
        self.gdb = gdb
        self.street_network = street_network
        self.network_type = network_type
        self.reset = reset
        self.path = os.path.join(self.gdb.gdb_path, f"{self.street_network.snake_name}_{network_type}_fd")
        self.name = f"{self.network_type}_{self.street_network.snake_name}"

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
            arcpy.management.CreateFeatureDataset(self.gdb.gdb_path, self.street_network.snake_name,
                                              spatial_reference=arcpy.SpatialReference(4326)
            )


class StreetFeatureClasses:
    # StreetFeatureClasses is nodes and edges for the street network
    def __init__(self, feature_dataset:FeatureDataset, street_network:StreetNetwork, use_elevation=False, reset=False):
        self.feature_dataset = feature_dataset
        self.street_network = street_network
        self.use_elevation = use_elevation # determines whether to add z values to nodes in feature classes
        self.reset = reset

        # paths for the two feature classes
        self.nodes_fc_path = os.path.join(self.feature_dataset.path, "nodes_walking_fc")
        self.edges_fc_path = os.path.join(self.feature_dataset.path, "edges_walking_fc")

        # useful shorthand to have rather than writing self.street_network.network_nodes etc all the time
        self.nodes = street_network.network_nodes
        self.edges = street_network.network_edges
        
        # erorr handling for using elevation when creating feature classes for street network
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
                arcpy.management.CreateFeatureclass(self.feature_dataset.path, "nodes_walking_fc",
                                    geometry_type="POINT", spatial_reference=arcpy.SpatialReference(4326), 
                                        has_z="ENABLED"
                )
                arcpy.management.CreateFeatureclass(self.feature_dataset.path, "edges_walking_fc",
                                    geometry_type="POLYLINE", spatial_reference=arcpy.SpatialReference(4326)
                )

            # not using elevation
            else:
                arcpy.management.CreateFeatureclass(self.feature_dataset.path, "nodes_walking_fc",
                                    geometry_type="POINT", spatial_reference=arcpy.SpatialReference(4326)
                )
                arcpy.management.CreateFeatureclass(self.feature_dataset.path, "edges_walking_fc",
                                    geometry_type="POLYLINE", spatial_reference=arcpy.SpatialReference(4326)
                )

        # both street FCs exist and reset not desired, so do nothing :)
        else:
            logging.info("Existing street network feature classes found and will be used")
            pass

        # method outputs
        logging.info("Successfully created empty feature classes")
        return self.nodes_fc_path, self.edges_fc_path



### FINISH ADAPTING FOR OOP ###

    # method uses arcpy cursor to add OSM gdfs data to feature classes, need to fix (attr fields empty except geometry?)
    def add_street_network_data_to_feature_classes(self):
        process_start_time = time.perf_counter()
        logging.info("Mapping street network data to feature classes")

        # first checking that empty feature classes exist
        if not (arcpy.Exists(self.nodes_fc_path) and arcpy.Exists(self.edges_fc_path)):
            raise Exception(f"Cannot add street network data because either nodes "
                            f"{self.nodes_fc_path} or {self.edges_fc_path} doesn't exist (create empty first)")

        # now mapping data to feature classes for network for non-geometry fields (have to handle geometry seperately)
        # this section just makes sure each field mapped to feature classes gets the write type
        for col in self.nodes.columns:
            # for nodes
            if col != "geometry":
                if self.nodes[col].dtype in ["int64", "int32"]:
                    field_type = "LONG"
                    arcpy.management.AddField(self.nodes_fc_path, col, field_type)
                elif self.nodes[col].dtype in ["float64", "float32"]:
                    field_type = "DOUBLE"
                    arcpy.management.AddField(self.nodes_fc_path, col, field_type)
                else:
                    field_type = "TEXT"
                    arcpy.management.AddField(self.nodes_fc_path, col, field_type, field_length=255)

        # now mapping non goeometry fields for edges
        for col in self.edges.columns:
            if col != "geometry":
                field_type = "TEXT"
                if self.edges[col].dtype in ["int64", "int32"]:
                    field_type = "LONG"
                    arcpy.management.AddField(self.edges_fc_path, col, field_type)
                elif self.edges[col].dtype in ["float64", "float32"]:
                    field_type = "DOUBLE"
                    arcpy.management.AddField(self.edges_fc_path, col, field_type)
                else:
                    field_type = "TEXT"
                    arcpy.management.AddField(self.edges_fc_path, col, field_type, field_length=255)

        # if using elevation then need to make xyz points                                                                # FIGURE OUT HOW TO HANDLE EDGES
        if self.use_elevation:
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
            arcpy.management.AddField(self.edges_fc_path, "walk_time", "DOUBLE")
            arcpy.management.CalculateField(
                self.edges_fc_path,
                "walk_time",
                "!Shape.length@meters! / 85",
                "PYTHON3")

            # making sure no multipart edges
            with arcpy.da.SearchCursor(self.edges_fc_path, ["SHAPE@"]) as cursor:
                for row in cursor:
                    if row[0].isMultipart:
                        logging.warning("Multipart geometry detected!")
                        break

            # integrate to snap nearby vertices
            arcpy.management.Integrate(self.edges_fc_path, "0.1 Meters")
            process_runtime = time.perf_counter() - process_start_time
            logging.info(f"Created Feature Classes from osmnx nodes and edges in {process_runtime} seconds")
            return self.nodes_fc_path, self.edges_fc_path

        else: # in case where elevation not enabled
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
            logging.info("Adding necessary fields for edges")                                                            # will need to change this when adding grade to edges for elevation enabled network dataset
            arcpy.management.AddField(self.edges_fc_path, "walk_time", "DOUBLE")
            arcpy.management.CalculateField(self.edges_fc_path,"walk_time","!Shape.length@meters! / 85",
                "PYTHON3")

            # making sure no multipart edges
            with arcpy.da.SearchCursor(self.edges_fc_path, ["SHAPE@"]) as cursor:
                for row in cursor:
                    if row[0].isMultipart:
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
        arcpy.conversion.FeatureClassToShapefile("nodes_walking_fc", self.street_network.cache_folder.path)
        arcpy.conversion.FeatureClassToShapefile("edges_walking_fc", self.street_network.cache_folder.path)
        logging.info(f"Nodes and edges feature classes succesfully saved as shapefiles in cache folder "
                     f"{self.street_network.cache_folder}")
        arcpy.env.overwriteOutput = False # setting overwrite output back to false so no accidental overwriting

class NetworkDataset:
    """
    Network dataset class
    """
    def __init__(self, feature_dataset:FeatureDataset, network_type:str ="walking", use_elevation=False, reset=False):
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

        # check not trying to use elevation without elevation enabled street network
        if use_elevation and not self.street_network.elevation_enabled:
            raise Exception("Cannot use elevat")
    def create_network_dataset(self):
        """
        Creates network dataset of type (self.network type)
        :return
            Path of network dataset
        """
        process_start_time = time.perf_counter()
        logging.info(f"Creating {self.network_type} network dataset for {self.street_network.snake_name}")               # change so looks better (use short name rather than snake name)

        # making sure network analyst checked out
        if not network_analyst_extension_checked_out:
            check_out_network_analyst_extension()

        # making sure not trying to obliviously create a network dataset if one with same name already exists
        if arcpy.Exists(self.path) and not self.reset:
            raise Exception(f"Cannot create new network dataset {self.name} because one already exists"
                            f"with that name at the desired path {self.path}")

        # main code block for the actual creating of the network dataset                                                                                                                 # currently starting with just walking network dataset but will eventually add support for transit (+ biking?)
        try:
            if self.reset:
                arcpy.Delete_management(self.path)
                logging.info(f"Existing network dataset {self.name} for {self.street_network.snake_name}")               # again replace snake name with short name

            # check whether to use elevation in network dataset
            if self.use_elevation:
                # first need to check that the street network is actually elevation enabled
                if not self.street_network.elevation_enabled:
                    raise Exception("The street network provided does not have elevation data")

                if self.network_type == "walking":
                    raise Exception("Oops, elevation network dataset not supported yet")                                 # write code for creating network datasets using elevation here
                
            elif self.network_type == "walking":
                # check that both nodes and edges feature classes exist in dataset
                if not arcpy.Exists(self.nodes_fc_path):
                    raise Exception("Nodes feature class does not exist in feature dataset")
                if not arcpy.Exists(self.edges_fc_path):
                    raise Exception("Edges feature class does not exist in feature dataset")

                arcpy.na.CreateNetworkDatasetFromTemplate(network_dataset_template="walking_nd_template.xml")
                logging.info("Successfully created walking network dataset from template")
            
            # error handling
            else:
                raise ValueError("Selected network_type not supported")
            logging.info(f"Successfully created network dataset in {time.perf_counter() - process_start_time} seconds")

        finally: # have to check extension back in when done running
            arcpy.CheckInExtension("Network")
            self.has_been_created = True
            return self.path

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
        check_network_analyst_extension_back_in()
        process_run_time = time.perf_counter() - process_start_time
        logging.info(f"Network dataset successfully built in {process_run_time} second")
        return self.path

class Place:
    def __init__(self, arcgis_project:ArcProject, name, bound_box=None):
        self.arcgis_project = arcgis_project
        self.name = name
        self.bound_box = bound_box
        self.street_network_data_exists = False
        self.elevation_data_exists = False
        self.gdb_exists = False
        self.feature_dataset_exists = False
        self.streets_feature_classes_exists = False
        self.network_dataset_types = []
        self.network_datasets_built = {}
        self.reference_place = ({"place_name": self.name})
        if bound_box is not None:
            self.reference_place["bound_box"] = self.bound_box
        self.snake_name = create_snake_name(self.reference_place)
        # cache folder for place
        self.cache_folder = CacheFolder(self.snake_name)


        if not self.cache_folder.check_if_cache_folder_exists():
            self.cache_folder.set_up_cache_folder()

    def get_street_network_data_from_place(self, network_type, use_elevation=False):
        pass

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
    
    def create_network_dataset_from_place(self, network_type, use_elevation):                                             # still need to figure out what to do with bounding box rather than place
        # create StreetNetwork object for this place
        street_network_for_place = StreetNetwork(self.name, network_type=network_type)
        street_network_for_place.get_street_network_from_osm()

        if use_elevation:
            elevation_mapper_for_place = ElevationMapper(street_network_for_place)
            elevation_mapper_for_place.add_elevation_data_to_nodes()
            elevation_mapper_for_place.add_grades_to_edges()
            logging.warning("Elevation data added for nodes, but cannot create network dataset with elevation yet")

        geodatabase_for_place = GeoDatabase(self.arcgis_project, street_network=street_network_for_place)
        feature_dataset_for_place = FeatureDataset(geodatabase_for_place, street_network_for_place)


        

if network_analyst_extension_checked_out:
    check_network_analyst_extension_back_in()
# # # # # # # # # # # # # # # # # # Testing Area :::: DO NOT REMOVE "if __name__ ..." # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    pinole_street_network = StreetNetwork("El Sobrante, California, USA")
    pinole_street_network_with_elevation = ElevationMapper(pinole_street_network, reset=True)
    print(pinole_street_network_with_elevation.add_elevation_data_to_nodes().head())


