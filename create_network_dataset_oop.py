import os
import sys
from pathlib import Path

from numba.cuda import runtime

arcgis_bin = r"C:\Program Files\ArcGIS\Pro\bin"
arcgis_extensions = r"C:\Program Files\ArcGIS\Pro\bin\Extensions"
os.environ["PATH"] = arcgis_bin + os.pathsep + arcgis_extensions + os.pathsep + os.environ.get("PATH", "")

# if you want to add modules, MUST (!!!!!!!) come after this block (ensures extensions work in cloned)
import arcpy
import arcpy_init
###################################################################

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

class Cache:
    def __init__(self, cache_path, file_type):
        self.cache_path = cache_path
        self.file_type = file_type

class StreetNetwork:
    def __init__(self, city_name, network_type="walk", bounding_box=None):
        self.city_name = city_name
        self.use_cache = True
        self.type = network_type
        self.bounding_box = bounding_box

    def do_not_use_cache(self):
        self.use_cache = False

    def load_from_cache(self):
        pass

    def generate_street_network_graph(self):
        if self.bounding_box is None:
            pass




class GeoDatabase:
    pass

class FeatureDataset:
    pass

class FeatureClass:
    pass

class NetworkDataset:
    pass



