import os
import sys
import subprocess
import logging
import arcpy


# Using this module to create network dataset in Arcgis' python env
# because arcpy.na doesn't work in cloned env

# info/variables passed from create_network_dataset in City.create_network_dataset(self) subprocess
script_name = sys.argv[0] # should be the name of this moduele??
city_name = sys.argv[1]
feature_dataset_path = sys.argv[2]
nodes_walking_fc_pat = sys.argv[3]
edges_walking_fc_path = sys.argv[4]

logging.info(f"Creating network dataset for {city_name} with run_in_arcgispro.py")

# check that network analyst ext available otherwise won't work
if arcpy.CheckExtension("network") == "Available":
    arcpy.CheckOutExtension("network")
else:
    raise Exception("Network Analyst extension not available")

try:
    # name/path for ND
    network_dataset_name = "walking_nd"
    network_dataset_path = os.path.join(feature_dataset_path, network_dataset_name)

    # check to manage sure not already ND with that name in desired location
    if arcpy.Exists(network_dataset_path):
        logging.info("there was already a network dataset in the feature dataset "
                     "with that name, it will be deleterd")
        arcpy.Delete_management(network_dataset_path)

    # using Result arcobject to be able to get info out of process
    result = arcpy.na.CreateNetworkDataset(feature_dataset=feature_dataset_path, out_name=network_dataset_name,
                                    source_feature_class_names=["nodes_walking_fc",
                                                                   "edges_walking_fc"],elevation_model="NO_ELEVATION")
    logging.info(f"Successfully created network dataset {result}")

# check network analyst back in
finally:
    arcpy.CheckInExtension("network")