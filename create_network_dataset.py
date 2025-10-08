import os
import osmnx as ox
import arcpy
import geopandas as gpd
import logging
import config # using a config file to keep paths and whatnot private
from pathlib import Path
import pickle
import networkx as nx

# setup before can run anything else
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
ox.settings.timeout = 300
ox.settings.max_query_area_size = 50000000
ox.settings.use_cache = True

# ArcGIS project setup stuff
arc_project_path = config.arc_project_path
arc_project_file = arcpy.mp.ArcGISProject(r""
                        r"C:\Users\aaron\localArcGISProjects\network_dataset_package\network_dataset_package.aprx")
walk_nd_template = config.walk_nd_path
arc_maps_list = arc_project_file.listMaps()
# check if any maps, if not, make one
if arc_maps_list:
    arc_project_map = arc_maps_list[0]
    logging.info(f"Map {arc_project_map.name} found")
else:
    arc_project_map = arc_project_file.createMap("Map")
    logging.info("Created new map")
    arc_project_file.save()


# setting up cache folder so I don't have to keep running the same things over and over again

class City:

    def __init__(self, name, hard_reset=False, streets_reset=False, use_cache=False, gdb_reset=False, fd_reset=False, fc_reset=False, nd_reset=False):
        self.fd_reset = fd_reset
        self.fc_reset = fc_reset
        self.nd_reset = nd_reset
        self.network_dataset_path = None
        self.feature_dataset_path = None
        self.edges_walking_fc_path = None
        self.gdb_path = None
        self.nodes_walking_fc_path = None
        self.use_cache = use_cache
        self.walking_streets = None
        self.all_streets = None
        self.gdb_reset = gdb_reset # gdb_reset also resets FD and FC
        self.streets_reset = streets_reset
        self.hard_reset = hard_reset
        self.edges_walking_fc = None
        self.nodes_walking_fc = None
        self.nodes_walking = None
        self.edges_walking = None
        self.name = name
        self.cache_dir = None
        self.snake_name = "_".join(part.strip().lower() for part in self.name.split(","))

        # hard reset override
        if self.hard_reset:
            self.use_cache = False
            self.streets_reset = True
            self.gdb_reset = True
            self.fd_reset = True
            self.fc_reset = True
            self.nd_reset = True

        # do associated resets


        # setting up cache folder so don't have to keep running the same things over and over again
        env_dir_path = Path(__file__).parent
        self.cache_dir = os.path.join(env_dir_path, f"{self.snake_name}_cache")
        logging.info("checking for cached data")
        if os.path.exists(self.cache_dir) and not self.hard_reset:
            logging.info("Cache found")
            if self.use_cache:
                logging.info("Using existing cache")
            else:
                raise Exception(f"You already have a cache folder (suspicously) with the name{self.cache_dir}")

        else:
            os.makedirs(self.cache_dir, exist_ok=True)

    # check if streets data already cached, if not, get streets graphs from osmnx

### SPLIT into seperate cache setup and data fetching methods?


    def update_streets_data(self):
        logging.info("Fetching streets data from osmnx or cache")

        # paths for data to cache, with graphs saved as .graphml and gdfs as GeoJsons
        all_streets_cache_path = os.path.join(self.cache_dir, "all_streets.graphml")
        walking_streets_cache_path = os.path.join(self.cache_dir, "walking_streets.graphml")
        nodes_walking_cache_path = os.path.join(self.cache_dir, "nodes_walking.GeoJSON")
        edges_walking_cache_path = os.path.join(self.cache_dir, "edges_walking.GeoJSON")

        # check whether data cached
        streets_cache_exists = all([os.path.exists(all_streets_cache_path),
                            os.path.exists(walking_streets_cache_path),
                            os.path.exists(nodes_walking_cache_path),
                            os.path.exists(edges_walking_cache_path)])
        # if caches
        if streets_cache_exists and self.use_cache and not self.streets_reset:
            logging.info("you already have streets data in your cache_directory, which will be returned to you \n"
                         "if you would like to rewrite this data, please pass reset=True when constructing City object")

### IMPORTANT: need to fix caching, because something weird happens when try to load in either the graphs or gdfs

            with open(all_streets_cache_path, 'rb') as cache_file:
                all_streets = pickle.load(cache_file)
            with open(walking_streets_cache_path, 'rb') as cache_file:
                walking_streets = pickle.load(cache_file)
            with open(nodes_walking_cache_path, 'rb') as cache_file:
                nodes_walking = pickle.load(cache_file)
            with open(edges_walking_cache_path, 'rb') as cache_file:
                edges_walking = pickle.load(cache_file)

        # use cache to set attributes
            (self.all_streets, self.walking_streets, self.nodes_walking,
             self.edges_walking) = all_streets, walking_streets, nodes_walking, edges_walking
            return all_streets, walking_streets, nodes_walking, edges_walking


        else:
            logging.info(f"Creating streets data for {self.name}")
        # create graphs from osmnx for selected place
            all_streets = ox.graph_from_place(self.name, network_type="all")
            walking_streets = ox.graph_from_place(self.name, network_type="walk")
        # turn graphs into 2 gdfs, one for nodes, one for edges
            # nodes_all, edges_all = ox.graph_to_gdfs(all_streets, nodes=True, edges=True)
            nodes_walking, edges_walking = ox.graph_to_gdfs(walking_streets, nodes=True,  edges=True)

            cache_dict = {all_streets_cache_path: all_streets, walking_streets_cache_path: walking_streets,
                          nodes_walking_cache_path: nodes_walking, edges_walking_cache_path: edges_walking}
            (self.all_streets, self.walking_streets, self.nodes_walking,
             self.edges_walking) = all_streets, walking_streets, nodes_walking, edges_walking

        # save to cache
            for file_path in cache_dict:
                with open(file_path, 'wb') as cache_file:
                    pickle.dump(cache_dict[file_path], cache_file)

        return all_streets, walking_streets, nodes_walking, edges_walking

### DON'T FORGET: write code to make caches for everything generated in this method
    def setup_gdb(self):
        logging.info("Setting up geodatabse where network dataset will go")

        # set up arcgis workspace
        arcpy.env.workspace = arc_project_path
        arcpy.env.overwriteOutput = True
        # path
        self.gdb_path = os.path.join(arc_project_path, f"{self.snake_name}_network_dataset.gdb")
        logging.debug(f"GDB path: {self.gdb_path}")
        logging.debug(f"GDB exists: {arcpy.Exists(self.gdb_path)}")
        logging.debug(f"gdb_reset: {self.gdb_reset}")

### IMPORTANT TO DELETE ONCE CACHING FOR GDB SETUP
        if arcpy.Exists(self.gdb_path) and self.gdb_reset:
            arcpy.Delete_management(self.gdb_path)
        arcpy.management.CreateFileGDB(arc_project_path, f"{self.snake_name}_network_dataset.gdb")
        arcpy.env.workspace = self.gdb_path
        arc_project_file.defaultGeodatabase = self.gdb_path
        arc_project_file.save()

# create feature dataset for nodes and edges to go into (as well as GTFS in future potentially)
    def create_feature_dataset(self):
        logging.info("Creating Feature Dataset")
        # path
        self.feature_dataset_path = os.path.join(self.gdb_path, f"{self.snake_name}_feature_dataset")
        # checking in FD already exists and deleting if reset selected
        if arcpy.Exists(self.feature_dataset_path) and not self.fd_reset:
            raise Exception(f"There is already a feature dataset in your "
                            f"geodatabase with name {self.snake_name}_feature_dataset")
        else:
            arcpy.Delete_management(self.feature_dataset_path)
        feature_dataset_name = f"{self.snake_name}_feature_dataset"
        arcpy.management.CreateFeatureDataset(self.gdb_path, feature_dataset_name,
                                              spatial_reference=arcpy.SpatialReference(4326))


# split here for Feature Dataset and encapsulated fc method
    def create_feature_classes(self):
        logging.info("Creating feature classes for edges and nodes")
        # paths for feature classes
        self.nodes_walking_fc_path = os.path.join(self.feature_dataset_path, "nodes_walking_fc")
        self.edges_walking_fc_path = os.path.join(self.feature_dataset_path, "edges_walking_fc")

        if not self.fc_reset:
            logging.info("Using cached data")
            return self.nodes_walking_fc_path, self.edges_walking_fc_path

        else:
            logging.info("Creating feature classes")
            # creating feature class for edges and nodes
            arcpy.management.CreateFeatureclass(self.feature_dataset_path,"nodes_walking_fc",
                                                geometry_type="POINT",
                                                spatial_reference=arcpy.SpatialReference(4326))
            arcpy.management.CreateFeatureclass(self.feature_dataset_path, "edges_walking_fc",
                                                geometry_type="POLYLINE",
                                                spatial_reference=arcpy.SpatialReference(4326))

            # creating nodes feature class
            for col in self.nodes_walking.columns:
                if col != "geometry":
                    field_type = "TEXT"
                    if self.nodes_walking[col].dtype in ["int64", "int32"]:
                        field_type = "LONG"
                        arcpy.management.AddField(self.nodes_walking_fc_path, col, field_type)
                    elif self.nodes_walking[col].dtype in ["float64", "float32"]:
                        field_type = "DOUBLE"
                        arcpy.management.AddField(self.nodes_walking_fc_path, col, field_type)
                    else:
                        field_type = "TEXT"
                        arcpy.management.AddField(self.nodes_walking_fc_path, col, field_type, field_length=255)

            node_fields = ["SHAPE@XY"] + [col for col in self.nodes_walking.columns if col not in ["geometry", "x", "y"]]
            with arcpy.da.InsertCursor(self.nodes_walking_fc_path, node_fields) as cursor:
                for idx, row in self.nodes_walking.iterrows():
                    geom = (row.geometry.x, row.geometry.y)
                    values = [geom] + [row[col] for col in self.nodes_walking.columns if col not in ["geometry", "x", "y"]]
                    cursor.insertRow(values)

            # creating edges feature class
            for col in self.edges_walking.columns:
                if col != "geometry":
                    field_type = "TEXT"
                    if self.edges_walking[col].dtype in ["int64", "int32"]:
                        field_type = "LONG"
                        arcpy.management.AddField(self.edges_walking_fc_path, col, field_type)
                    elif self.edges_walking[col].dtype in ["float64", "float32"]:
                        field_type = "DOUBLE"
                        arcpy.management.AddField(self.edges_walking_fc_path, col, field_type)
                    else:
                        field_type = "TEXT"
                        arcpy.management.AddField(self.edges_walking_fc_path, col, field_type, field_length=255)

            edge_fields = ["SHAPE@"] + [col for col in self.edges_walking.columns if col != "geometry"]
            with arcpy.da.InsertCursor(self.edges_walking_fc_path, edge_fields) as cursor:
                for idx, row in self.edges_walking.iterrows():
                    coords = list(row.geometry.coords)
                    array = arcpy.Array([arcpy.Point(x, y) for x, y in coords])
                    polyline = arcpy.Polyline(array, arcpy.SpatialReference(4326))
                    values = [polyline] + [
                        ', '.join(map(str, row[col])) if isinstance(row[col], list) else row[col]
                        for col in self.edges_walking.columns if col != "geometry"
                    ]
                    cursor.insertRow(values)
            # adding fields for network dataset creations
            logging.info("Adding necessary fields for edges")
            arcpy.management.AddField(self.edges_walking_fc_path, "walk_time", "DOUBLE")
            arcpy.management.CalculateField(
                self.edges_walking_fc_path,
                "walk_time",
                "!Shape.length@meters! / 85",
                "PYTHON3")


            # add feature classes to current map
            current_map = arc_project_file.listMaps()[0]
            current_map.addDataFromPath(self.nodes_walking_fc_path)
            current_map.addDataFromPath(self.edges_walking_fc_path)

            return self.nodes_walking_fc_path, self.edges_walking_fc_path

    def create_network_dataset(self):
        # using the node and edge feature classes to create a new network dataset
        logging.info("Creating network dataset")

        # activate network analyst extension
        if arcpy.CheckExtension("network") == "Available":
            arcpy.CheckOutExtension("network")
            logging.info("Network Analyst extension checked out")
        else:
            raise Exception("Network Analyst extension is not available")

        # path
        network_dataset_name = "ND"
        self.network_dataset_path = os.path.join(self.feature_dataset_path, network_dataset_name)

        # make sure there's no network dataset in the current FD
        if arcpy.Exists(self.network_dataset_path):
            arcpy.Delete_management(self.network_dataset_path)
        fd_name = f"{self.snake_name}_feature_dataset"
        arcpy.na.CreateFeatureDataset(
            self.feature_dataset_path,
            "ND",
            ["nodes_walking_fc", "edges_walking_fc"], elevation_model="NO_ELEVATION"
        )

    def compile_overall(self):
        logging.info(f"Now Creating Network Dataset for {self.name}")
        self.update_streets_data()
        self.setup_gdb()
        self.create_feature_dataset()
        self.create_feature_classes()
        self.create_network_dataset()
        logging.info("Done!")

Berkeley = City("Berkeley, California, USA", hard_reset=True, use_cache=True, gdb_reset=True, fc_reset=True, fd_reset=True, nd_reset=True)
Berkeley.compile_overall()


 