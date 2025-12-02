""" This module uses the tools in tools_for_place to generate cost matrices for Chicago's public transit system """

from tools_for_place import *

if __name__ == "__main__":
    arc_gis_project = ArcProject("upp_461_final")
    # messed up when running this before so need to keep using scenaruio_id AM_Peak to avoid making new network dataset
    chicago_place = Place(arcgis_project=arc_gis_project, place_name="Chicago, Illinois, USA",
                          geographic_scope="csa", scenario_id="AM_Peak")

    # the times of day want to generate cost matrices for
    times_of_day = [{"matrix_name": "Overnight", "analysis_time" : "1:00 AM"},
                    {"matrix_name": "AM_Shoulder", "analysis_time" : "6:00 AM"},
                    {"matrix_name": "AM_Mid_Peak", "analysis_time" : "8:00 AM"},
                    {"matrix_name": "AM_Peak", "analysis_time" : "9:00 AM"},
                    {"matrix_name": "Midday", "analysis_time" : "12:00 PM"},
                    {"matrix_name": "PM_Shoulder", "analysis_time" : "3:00 PM"},
                    {"matrix_name": "PM_Peak", "analysis_time" : "5:00 PM"},
                    {"matrix_name": "PM_After_Peak", "analysis_time" : "7:00 PM"}]

    taz_centroids_path = r"C:\Users\Aaron\PycharmProjects\transit_network_datsets_with_ArcPy\upp_461_final\chicago_illinois_usa_csa.gdb\taz_centroids_"

    cycle_start_time = time.perf_counter()
    # now going to go through each time of day and generate cost matrix
    for time_of_day in times_of_day:
        # first check that not AM_Peak or PM_Peak since those were already done
        this_matrix_name = time_of_day["matrix_name"]
        this_analysis_time = time_of_day["analysis_time"]
        if this_matrix_name in ["AM_Peak", "PM_Peak"]:
            print(f"Skipping {this_matrix_name} since already done")
            continue
        chicago_place.generate_large_od_cost_matrix(matrix_name=this_matrix_name, origins_fc_path=taz_centroids_path,
                                                    destinations_fc_path=taz_centroids_path, network_type="transit",
                                                    use_elevation=False, day_of_week="Monday",
                                                    analysis_time=this_analysis_time, open_on_complete=False)
        print(f"Completed {this_matrix_name} cost matrix")
    cycle_end_time = time.perf_counter()
    total_cycle_time = cycle_end_time - cycle_start_time
    print(f"Total time to generate all cost matrices: {turn_seconds_into_minutes(total_cycle_time)}")
    print("All cost matrices for existing network generated!!!!")

