import os
import osmnx as ox
import arcpy
import geopandas as gpd
import logging
import config # using a config file to keep paths and whatnot private
from pathlib import Path

# setup before can run anything else
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
ox.settings.timeout = 300
ox.settings.max_query_area_size = 50000000
ox.settings.use_cache = True
arc_project_path = config.arc_project_path


# setting up cache folder so I don't have to keep running the same things over and over again

class City:

    def __init__(self, name, hard_reset=False, streets_reset=False, use_cache=False, gdb_reset=False):
        self.use_cache = use_cache
        self.walking_streets = None
        self.all_streets = None
        self.gdb_reset = gdb_reset
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

    # check if streets data already cached, if not, get strets graphs from osmnx
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

# IMPORTANT: need to fix caching, because something weird happens when try to load in either the graphs or gdfs
            all_streets = ox.load_graphml(all_streets_cache_path)
            walking_streets = ox.load_graphml(walking_streets_cache_path)
            nodes_walking = gpd.read_file(nodes_walking_cache_path)
            edges_walking = gpd.read_file(edges_walking_cache_path)

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

            (self.all_streets, self.walking_streets, self.nodes_walking,
             self.edges_walking) = all_streets, walking_streets, nodes_walking, edges_walking

        # save to cache
            ox.save_graphml(all_streets, all_streets_cache_path)
            ox.save_graphml(walking_streets, walking_streets_cache_path)
            nodes_walking.to_file(nodes_walking_cache_path, driver="GeoJSON")
            edges_walking.to_file(edges_walking_cache_path, driver="GeoJSON")



        return all_streets, walking_streets, nodes_walking, edges_walking

### DON'T FORGET: write code to make caches for everything generated in this method
    def setup_gdb(self):
        logging.info("Setting up geodatabse where network dataset will go")

        # set up arcgis workspace
        arcpy.env.workspace = arc_project_path
        arcpy.env.overwriteOutput = True

        gdb_path = os.path.join(arc_project_path, f"{self.snake_name}_network_dataset.gdb")

        logging.debug(f"GDB path: {gdb_path}")
        logging.debug(f"GDB exists: {arcpy.Exists(gdb_path)}")
        logging.debug(f"gdb_reset: {self.gdb_reset}")

### IMPORTANT TO DELETE ONCE CACHING FOR GDB SETUP
        if arcpy.Exists(gdb_path) and self.gdb_reset:
            arcpy.Delete_management(gdb_path)
        arcpy.management.CreateFileGDB(arc_project_path, f"{self.snake_name}_network_dataset.gdb")

        # paths for feature classes
        nodes_walking_fc_path = os.path.join(gdb_path, "nodes_walking_fc")
        edges_walking_fc_path = os.path.join(gdb_path, "edges_walking_fc")

        # creating feature class for edges and nodes
        arcpy.management.CreateFeatureclass(gdb_path,"nodes_walking_fc",
                                            geometry_type="POINT",
                                            spatial_reference=arcpy.SpatialReference(3857))
        arcpy.management.CreateFeatureclass(gdb_path, "edges_walking_fc",
                                            geometry_type="POLYLINE",
                                            spatial_reference=arcpy.SpatialReference(3857))
        for col in self.nodes_walking.columns:
            if col != "geometry":
                field_type = "TEXT"
                if self.nodes_walking[col].dtype in ["int64", "int32"]:
                    field_type = "LONG"
                    arcpy.management.AddField(nodes_walking_fc_path, col, field_type)
                elif self.nodes_walking[col].dtype in ["float64", "float32"]:
                    field_type = "DOUBLE"
                    arcpy.management.AddField(nodes_walking_fc_path, col, field_type)
                else:
                    field_type = "TEXT"
                    arcpy.management.AddField(nodes_walking_fc_path, col, field_type, field_length=255)

        for col in self.edges_walking.columns:
            if col != "geometry":
                field_type = "TEXT"
                if self.edges_walking[col].dtype in ["int64", "int32"]:
                    field_type = "LONG"
                    arcpy.management.AddField(edges_walking_fc_path, col, field_type)
                elif self.edges_walking[col].dtype in ["float64", "float32"]:
                    field_type = "DOUBLE"
                    arcpy.management.AddField(edges_walking_fc_path, col, field_type)
                else:
                    field_type = "TEXT"
                    arcpy.management.AddField(edges_walking_fc_path, col, field_type, field_length=255)

        node_fields = ["SHAPE@XY"] + [col for col in self.nodes_walking.columns if col not in ["geometry", "x", "y"]]
        with arcpy.da.InsertCursor(nodes_walking_fc_path, node_fields) as cursor:
            for idx, row in self.nodes_walking.iterrows():
                geom = (row.geometry.x, row.geometry.y)
                values = [geom] + [row[col] for col in self.nodes_walking.columns if col not in ["geometry", "x", "y"]]
                cursor.insertRow(values)

        edge_fields = ["SHAPE@"] + [col for col in self.edges_walking.columns if col != "geometry"]
        with arcpy.da.InsertCursor(edges_walking_fc_path, edge_fields) as cursor:
            for idx, row in self.edges_walking.iterrows():
                coords = list(row.geometry.coords)
                array = arcpy.Array([arcpy.Point(x, y) for x, y in coords])
                polyline = arcpy.Polyline(array, arcpy.SpatialReference(3857))
                values = [polyline] + [
                    ', '.join(map(str, row[col])) if isinstance(row[col], list) else row[col]
                    for col in self.edges_walking.columns if col != "geometry"
                ]
                cursor.insertRow(values)

        self.nodes_walking_fc, self.edges_walking_fc = nodes_walking_fc_path, edges_walking_fc_path
        return nodes_walking_fc_path, edges_walking_fc_path

    def create_network_dataset(self):
        logging.info(f"Now Creating Network Dataset for {self.name}")
        self.update_streets_data()
        self.setup_gdb()

Berkeley = City("Berkeley, California, USA", hard_reset=True, use_cache=True, gdb_reset=True)
Berkeley.create_network_dataset()


 