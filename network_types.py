"""
This module is used to define the network types and their attributes used for creation of network dataset
"""
import os
from pathlib import Path

network_dataset_template_dir = os.path.join(Path(__file__).parent,"nd_templates")

def get_template_path(template_name):
    """ Takes network name and gives path to corresponding template """
    return os.path.join(network_dataset_template_dir, f"{template_name}_nd_template.xml")


network_types_attributes = {"walk_no_z": {
                                "osmnx_network_type":"walk",
                                "network_dataset_template_name": get_template_path("walk_no_z"),
                                "isochrone_travel_mode":"Walking"},
                            "walk_z": {
                                "osmnx_network_type":"walk",
                                "network_dataset_template_name": get_template_path("walk_z"),
                                "isochrone_travel_mode": "Walking"},
                            "bike_no_z": {
                                "osmnx_network_type":"bike",
                                "network_dataset_template_name": get_template_path("bike_no_z")},
                            "bike_z": {
                                "osmnx_network_type":"bike",
                                "network_dataset_template_name": get_template_path("bike_z")},
                            "transit_no_z": {
                                "osmnx_network_type":"walk",
                                "network_dataset_template_name": get_template_path("transit_no_z"),
                                "isochrone_travel_mode": "Public transit time"},
                            "transit_z": {
                                "osmnx_network_type":"walk",
                                "network_dataset_template_name": get_template_path("transit_z"),
                                "isochrone_travel_mode": "Public transit time"},
                            "drive": {
                                "osmnx_network_type":"drive",
                                "network_dataset_template_name": get_template_path("drive")},
                            "transit_plus_biking_no_z" : {
                                "osmnx_network_type":"bike",
                                "network_dataset_template_name": get_template_path("transit_plus_biking_no_z")},
                            "transit_plus_biking_z" : {
                                "osmnx_network_type":"bike",
                                "network_dataset_template_name": get_template_path("transit_plus_biking_z")}
                            }