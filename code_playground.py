""" This is the main user-interface module for those who want to experiment with my code """

from tools_for_place import *

if __name__ == "__main__":
    your_arcgis_project = ArcProject(name="code_demonstration") # input the name of the ArcGIS Pro project you want to
                                                           # use (this will be created if it doesn't exist)

    your_place = Place(arcgis_project=your_arcgis_project, place_name = "Berkeley, California, USA",
                       geographic_scope="place_only")   # input the place you want to create a network dataset (see
                                                        # documentation for Place for more information)

    your_place.create_network_dataset_from_place()  # input any parameters you would like to change, or run the default
                                                    # and see what happens! (see documentation
                                                    # for create_network_dataset_from_place for more information)

    your_place.generate_isochrone(addresses=["2534 Durant Ave, Berkeley, CA, USA","1561 Solano Ave, Berkeley, CA, USA"])
                                                    # input any parameters you would like to change, or run the default
                                                    # and see what happens!
                                                    # (see documentation for generate_isochrone for more information)

