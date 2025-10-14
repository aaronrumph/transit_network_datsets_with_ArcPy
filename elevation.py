# this module is what I'm going to use to add elevation data for the nodes from the osmnx graph
import osmnx
from pathlib import Path
import os
import logging



class Elevation_mapper:
    # Elevation_mapper object takes a graph and maps the nodes to elevation using the USGS DEM/tnm data


    def __init__(self, name):
        self.name = name
        self.elevation_data_folder_path = os.path.join(Path(__file__).parent, f"{self.name}_elevation_data")
        self.tnm_elevation_data_api_base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

        if not os.path.exists(self.elevation_data_folder_path):
            logging.info(f"Creating new elevation data folder for {name}")
            os.mkdir(self.elevation_data_folder_path, exist_ok=True)
        else:
            logging.info(f"Elevation data for {name} already exists")

        

