# Currently in process of rewriting my code to use classes to make my code more readable

import os
import sys
from pathlib import Path

# Setting up python environment/making sure env points to extensions correctly
arcgis_bin = r"C:\Program Files\ArcGIS\Pro\bin"
arcgis_extensions = r"C:\Program Files\ArcGIS\Pro\bin\Extensions"
os.environ["PATH"] = arcgis_bin + os.pathsep + arcgis_extensions + os.pathsep + os.environ.get("PATH", "")

# if you want to add modules, MUST (!!!!!!!) come after this block (ensures extensions work in cloned)
import arcpy
import arcpy_init
#

import subprocess
import pickle
import logging
import osmnx as ox
import geopandas as gpd
import networkx as nx
import config
import time
import requests
import multiprocessing as mp
from itertools import repeat


logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

ox.settings.timeout = 1000
ox.settings.max_query_area_size = 5000000000000
ox.settings.use_cache = True

# ArcGIS project setup
arc_project_path = config.arc_project_path
arc_project_file = arcpy.mp.ArcGISProject(os.path.join(arc_project_path, r"network_dataset_package_project.aprx"))
arcgis_py_env = config.arcgis_py_env

def init_worker(counter, lock):
    # initializer function for multiprocess processes, need in order to have a global variable accessible to all workers
    global shared_counter, shared_lock
    shared_counter = counter
    shared_lock = lock

def generic_get_elevation_function(start_time, total_nodes, idx, x_y: tuple):
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
        time_since_start = time.perf_counter() - start_time
        print(f"\rElevation: {current}/{total_nodes} nodes | Time: {time_since_start:.1f}s / "
              f"{(total_nodes / 7.8):.1f}s", end="", flush=True)

        return elevation

    except Exception as e:
        logging.warning(f"Error getting elevation for node {idx}: {e}")
        return None

class CacheFolder:
    def __init__(self, network_snake_name):
        self.network_snake_name = network_snake_name
        self.env_dir_path = Path(__file__).parent
        self.path = os.path.join(self.env_dir_path, f"{self.network_snake_name}_cache")
    
    def check_if_cache_folder_exists(self):
        # returns true if cache folder already exists for the city
        if os.path.exists(self.path):
            return True
        else:
            return False

    def set_up_cache_folder(self):
        # check if there is already a cache folder for city, if not, make one
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


    def get_street_network_from_osm(self, timer_on=True):
        # this method does the brunt of the work for the class
        start_time = time.perf_counter()
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
            logging.info(f"Got street network from OSM in {time.perf_counter() - start_time} seconds")
            logging.info(f"Street network for {self.city_name} had {len(network_nodes)} nodes and "
                         f"{len(network_edges)} edges")

        return network_graph, network_nodes, network_edges

# ElevationMapper class adds elevation to nodes and grades to edges (coming soon?) using USGS EPQS API
class ElevationMapper:
    # WARNINGS!: 1. Adding elevation takes forever, ~130ms per node with 14 threads (1.8s per node per thread!) 
    #                   because the API is so slow
    #            2. Uses multiprocessing (scary) so DO NOT take out the "if __name__ == ...." guard!!!!!

    def __init__(self, street_network:StreetNetwork, threads_available=mp.cpu_count(), reset=False):
        self.street_network = street_network
        self.threads_available = threads_available
        self.reset = reset
        (self.street_network_graph, self.street_network_nodes, self.street_network_edges) = (
            self.street_network.get_street_network_from_osm()
        )
        self.node_counter = 0
        self.elevation_cache = Cache(street_network.cache_folder, "elevation")

    # using multiprocessing to get elevation data if no cache exists
    def add_elevation_data_to_nodes(self):
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
            start_time = time.perf_counter()
            # using multiprocessing to send as many API requests at once as possible
            with mp.Pool(processes=self.threads_available,
            initializer=init_worker,
            initargs=(shared_counter, lock)) as pooled_threads:
                z_values = pooled_threads.starmap(generic_get_elevation_function,
                    zip(repeat(start_time), repeat(number_of_nodes_total),
                        idx_args_for_elevation, x_y_args_for_elevation
                    )
                )
            pooled_threads.close()

            self.street_network_nodes["z"] = z_values
            self.elevation_cache.write_cache_data(self.street_network_nodes["z"].to_dict())

            # outputs for method
            run_time = time.perf_counter() - start_time
            logging.info(f"Got elevation data for {len(self.street_network_nodes)} nodes in {run_time} seconds")
            return self.street_network_nodes
        else:
            z_values = self.elevation_cache.read_cache_data()   # loading elevation data from cache

            # check if the nodes gdf already has elevation
            if "z" not in self.street_network_nodes.columns:
                self.street_network_nodes["z"] = self.street_network_nodes.index.map(z_values)
                logging.info("Successfully added elevation data to nodes from cache")
            else:
                logging.info("Nodes gdf already has z values")

            return self.street_network_nodes

    # need to calculate grades for edges so can make walking time vary with slope
    def add_grades_to_edges(self):
        pass


class GeoDatabase:
    pass

class FeatureDataset:
    pass

class FeatureClass:
    pass

class NetworkDataset:
    pass


# # # # # # # # # # # # # # # # # # Testing Area :::: DO NOT REMOVE "if __name__ ..." # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    pinole_street_network = StreetNetwork("El Sobrante, California, USA")
    pinole_street_network_with_elevation = ElevationMapper(pinole_street_network, reset=True)
    print(pinole_street_network_with_elevation.add_elevation_data_to_nodes().head())


