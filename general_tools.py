"""
This module contains helpful helper functions that can be used in any given module with no circular import
problems
"""
import base64
import logging
import random
import re
import pluscodes
import requests

# getting API key from .env file
import os
from dotenv import load_dotenv
load_dotenv("api_keys.env")
census_bureau_api_key = os.getenv("CENSUS_BUREAU_API_KEY")
from geopy.geocoders import Nominatim

# nominatim user agent
nominatim_user_agent = os.getenv("NOMINATIM_USER_AGENT")



# regex matching patterns for use in checking other stuff
regex_matching_patterns = {
    # regex pattern to match valid longitudes built using regex generator because regex hard:
    "longitude":
        r"^[-+]?(180(\.0+)?|1[0-7]\d(\.\d+)?|\d{1,2}(\.\d+)?)$",

    # regex pattern to match valid latitudes
    "latitude":
        r"^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)$"}

def time_function(func):
    """
    Decorator to time functions
    :param func:
    :return:
    """
    import time
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.info(f"Function {func.__name__} took {turn_seconds_into_minutes(elapsed_time)} to complete.")
        return result
    return wrapper


class ReferencePlace:
    def __init__(self, place_name:str=None, bound_box:tuple[str|float, str|float, str|float, str|float]=None):
        """
        ReferencePlace class acts like a sort of dictionary and contains the place name and or bounding box for a place
        Attributes:
            place_name | str: the place name in the form "place, (division), country"
            bound_box | tuple[str,str,str,str]: (longitude_min, latitude_min, longitude_max, latitude_max)
            plus_codes_for_bbox_corners | tuple[str]: plus codes for each corner of the bounding box

        """
        self.place_name = place_name
        self.bound_box = bound_box

        if self.bound_box:
            self.pretty_name = self.bound_box
        else:
            self.pretty_name = self.place_name


        """ 
        Using plus codes as a way to represent the bounding box provided. This is mostly done to shorten file names. 
        The downside is that 1. Plus codes are not as legible as coordinates (I know that SF, for instance, is around 37
        something degrees north), and 2. Plus codes can be longer than the bounding box coordinates (e.g. SF is around
        12 characters long, while the bounding box coordinates are around 11 characters long). But better to have a 
        standardized way of doing it than to have a mix of formats and plus codes are shorter in cases where precise 
        coordinates are used.
        """
        self.plus_codes_for_bbox_corners:list = []
        self.out_name:str = ""

        # check that either a bound box or a place name has been provided:
        if self.place_name is None and self.bound_box is None:
            raise ValueError("Reference place arguments (place_name and bound_box) cannot both be undefined")

        # this part is just exception handling because everything else follows from this, so need correct ReferencePlace
        # first checking that bound_box was provided
        if self.bound_box is not None:
            # det out name to reflect bounding box
            self.out_name = f"Bounding box: {self.bound_box}"
            # check if ALL the provided values for the bbox were floats
            if all(isinstance(coord, float) for coord in self.bound_box):
            # in case the bounding box is not in the correct format (aka, if floats passed instead), rewrite it
                _rewrite_bound_box:list = [str(float_coord) for float_coord in self.bound_box]
                self.bound_box = tuple(_rewrite_bound_box)

            # next checking that each element in bounding box is 1. a str, 2. only float-ables, and 3. between -180 and 180
            for coord in self.bound_box:
                # check first that coord is string
                if not isinstance(coord, str):
                    # since already converted bbox from floats to strs (if was floats), just raise error for other types
                        raise ValueError("Bounding box coordinates must be given as strings")

                # easy check first to see that coordinate in bounding box does not contain bad characters
                if not re.match(r"-?\d+(?:\.\d+)?", coord):
                    raise ValueError("Bounding box coordinates must be strings containing "
                                     "only digits and decimal points")

            # now need to check that the provided bounding box is in fact (left, bottom, right, top)
            if not re.match(regex_matching_patterns["longitude"], self.bound_box[0]): #
                raise ValueError("First value passed in bound_box must be a valid longitude string")
            if not re.match(regex_matching_patterns["longitude"], self.bound_box[2]):
                raise ValueError("Third value passed in bound_box must be a valid longitude string")
            if not re.match(regex_matching_patterns["latitude"], self.bound_box[1]):
                raise ValueError("Second value passed in bound_box must be a valid latitude string")
            if not re.match(regex_matching_patterns["latitude"], self.bound_box[3]):
                raise ValueError("Fourth value passed in bound_box must be a valid latitude string")

            # if have made it this far without an error then bound box is safe to use and can now create plus codes
            self.create_plus_codes_for_bbox()

        # place name was provided but no bound box
        else:
            self.out_name = f"Place:{self.place_name}"

    def create_plus_codes_for_bbox(self):
        """
        Creates plus codes for the bounding box of the reference place using the bottom left corner and top right corner

        :return self.plus_codes_for_bbox_corners: tuple[str,str] | where each string in tuple is a pluscode
        """

        # always first check that bounding box was actually provided
        if self.bound_box is None:
            raise ValueError("Cannot generate plus codes for the bounding box because none was provided!")

        else:
            # need to convert bbox coordinates to floats to encode as plus codes
            lat_long_bottom_left_corner = (float(self.bound_box[1]), float(self.bound_box[0]))
            lat_long_top_right_corner = (float(self.bound_box[3]), float(self.bound_box[2]))

            # using pluscodes module to encode (lat, lon) to plus codes
            plus_code_bottom_left = pluscodes.encode(lat_long_bottom_left_corner[1], lat_long_bottom_left_corner[0])
            plus_code_top_right = pluscodes.encode(lat_long_top_right_corner[1], lat_long_top_right_corner[0])

            # now update attribute
            self.plus_codes_for_bbox_corners.append(plus_code_bottom_left)
            self.plus_codes_for_bbox_corners.append(plus_code_top_right)

            # now want to make tuple so can't accidentally modify
            self.plus_codes_for_bbox_corners = tuple(self.plus_codes_for_bbox_corners)

            # method output
            return self.plus_codes_for_bbox_corners

    # automatically create plus codes

def create_snake_name(name:str | ReferencePlace):
    """
    Creates snake name for place. For use in a bunch of other stuff. Can either take a string (simple place name like
    'San Francisco, California, USA' or a reference place dictionary (see reference place class) with a place name and
    a bounding box
    :param name:
    :return:
    """
    # first, if passed simple place name, just replace commas and spaces.
    if isinstance(name, str):
        return name.replace(" ", "_").replace(",", "").replace(r"/", "").lower()

    # else if passed ReferencePlace
    elif isinstance(name, ReferencePlace):
        # in case where bounding box is not provided
        if name.bound_box is None:
            concatenated_place_name = name.place_name.replace(" ", "_").replace(
                                            ",", "").replace(r"/", "").lower()
            return concatenated_place_name

        # in case where bounding box is given but place name is not
        elif name.bound_box is None:
            concatenated_plus_codes = "_".join(name.plus_codes_for_bbox_corners)
            return concatenated_plus_codes

        # case where both bounding box and place name are provided, by default will use bounding box
        else:
            concatenated_plus_codes = "_".join(name.plus_codes_for_bbox_corners)
            return concatenated_plus_codes

    # in case where tried to pass something other than a string or a ReferencePlace
    else:
        raise ValueError("Name must be either a string or a ReferencePlace instance")

# dictionary for converting state abbreviation to fips and name
state_fips_and_abbreviations = {
    "AL": {"fips": "01", "name": "Alabama"},
    "AK": {"fips": "02", "name": "Alaska"},
    "AZ": {"fips": "04", "name": "Arizona"},
    "AR": {"fips": "05", "name": "Arkansas"},
    "CA": {"fips": "06", "name": "California"},
    "CO": {"fips": "08", "name": "Colorado"},
    "CT": {"fips": "09", "name": "Connecticut"},
    "DE": {"fips": "10", "name": "Delaware"},
    "DC": {"fips": "11", "name": "District of Columbia"},
    "FL": {"fips": "12", "name": "Florida"},
    "GA": {"fips": "13", "name": "Georgia"},
    "HI": {"fips": "15", "name": "Hawaii"},
    "ID": {"fips": "16", "name": "Idaho"},
    "IL": {"fips": "17", "name": "Illinois"},
    "IN": {"fips": "18", "name": "Indiana"},
    "IA": {"fips": "19", "name": "Iowa"},
    "KS": {"fips": "20", "name": "Kansas"},
    "KY": {"fips": "21", "name": "Kentucky"},
    "LA": {"fips": "22", "name": "Louisiana"},
    "ME": {"fips": "23", "name": "Maine"},
    "MD": {"fips": "24", "name": "Maryland"},
    "MA": {"fips": "25", "name": "Massachusetts"},
    "MI": {"fips": "26", "name": "Michigan"},
    "MN": {"fips": "27", "name": "Minnesota"},
    "MS": {"fips": "28", "name": "Mississippi"},
    "MO": {"fips": "29", "name": "Missouri"},
    "MT": {"fips": "30", "name": "Montana"},
    "NE": {"fips": "31", "name": "Nebraska"},
    "NV": {"fips": "32", "name": "Nevada"},
    "NH": {"fips": "33", "name": "New Hampshire"},
    "NJ": {"fips": "34", "name": "New Jersey"},
    "NM": {"fips": "35", "name": "New Mexico"},
    "NY": {"fips": "36", "name": "New York"},
    "NC": {"fips": "37", "name": "North Carolina"},
    "ND": {"fips": "38", "name": "North Dakota"},
    "OH": {"fips": "39", "name": "Ohio"},
    "OK": {"fips": "40", "name": "Oklahoma"},
    "OR": {"fips": "41", "name": "Oregon"},
    "PA": {"fips": "42", "name": "Pennsylvania"},
    "RI": {"fips": "44", "name": "Rhode Island"},
    "SC": {"fips": "45", "name": "South Carolina"},
    "SD": {"fips": "46", "name": "South Dakota"},
    "TN": {"fips": "47", "name": "Tennessee"},
    "TX": {"fips": "48", "name": "Texas"},
    "UT": {"fips": "49", "name": "Utah"},
    "VT": {"fips": "50", "name": "Vermont"},
    "VA": {"fips": "51", "name": "Virginia"},
    "WA": {"fips": "53", "name": "Washington"},
    "WV": {"fips": "54", "name": "West Virginia"},
    "WI": {"fips": "55", "name": "Wisconsin"},
    "WY": {"fips": "56", "name": "Wyoming"},
    "AS": {"fips": "60", "name": "American Samoa"},
    "GU": {"fips": "66", "name": "Guam"},
    "MP": {"fips": "69", "name": "Northern Mariana Islands"},
    "PR": {"fips": "72", "name": "Puerto Rico"},
    "UM": {"fips": "74", "name": "U.S. Minor Outlying Islands"},
    "VI": {"fips": "78", "name": "U.S. Virgin Islands"}
}

def get_reference_places_for_scope(place_name:str, geographic_scope:str):
    """
    using nested functions because I am tired and want to
    """

    logging.info(f"Getting reference places for {place_name} {geographic_scope}")
    # first can just return city proper if try passing "place_only" as scope
    if geographic_scope == "place_only":
        return [ReferencePlace(place_name)]

    def get_lat_lon(place_name: str):
        """
        This function takes a place name (like "San Francisco, California, USA") and returns a list of ReferencePlaces
        corresponding to places in the same MSA as the place given.
        :param place_name:
        :return:
        """

        # first get the lat and lon of the place using the nominatim api
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        # q is unspecified query
        nominatim_params = {
            "q": place_name,
            "limit": "10",
            "format": "json"}
        # need to give valid User-Agent header
        nominatim_headers = {"User-Agent": nominatim_user_agent}

        # now get response from API
        nominatim_response = requests.get(nominatim_url, params=nominatim_params, headers=nominatim_headers)
        nominatim_response.raise_for_status()
        nominatim_data = nominatim_response.json()

        # extract lat and lon from response
        place_lat = nominatim_data[0]["lat"]
        place_lon = nominatim_data[0]["lon"]
        place_lat_lon:dict = {"lat": place_lat, "lon": place_lon}

        return place_lat_lon

    # the url for the geocoding API
    census_bureau_geocoding_url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"

    # adding in support for just the county level for a given place
    def get_county_from_lat_lon(place_lat_lon:dict):

        census_bureau_geocoding_params = {
            "x": place_lat_lon["lon"],
            "y": place_lat_lon["lat"],
            "format": "json",
            "benchmark": "Public_AR_Current",   # just the weird current version shorthand
            "vintage": "4",                     # same as above
            "layers": "82"}  # 80 is the layer code for county

        census_bureau_response = requests.get(census_bureau_geocoding_url, params=census_bureau_geocoding_params)
        census_bureau_response.raise_for_status()

        # turn into JSON
        census_bureau_data = census_bureau_response.json()
        returned_geographies = census_bureau_data["result"]["geographies"]
        name_for_county = returned_geographies["Counties"][0]["NAME"]

        # going to return a set of counties for consistency with msa and csa
        set_of_counties = {name_for_county}

        return set_of_counties

    # now that have lat and lon in place, can get MSA (and state) geoids using Census geocoder
    def get_msa_from_lat_lon(place_lat_lon:dict):
        """

        :param place_lat_lon:
        :return: geoids | {"msa_geoid": msa_geoid, "state_geoid": state_geoid}
        """
        # using the Census Bureau's geocoding API because free and gives geoids back getting msa geoid and state geoid
        census_bureau_geocoding_params = {
            "x": place_lat_lon["lon"],
            "y": place_lat_lon["lat"],
            "format": "json",
            "benchmark": "Public_AR_Current",   # just the weird current version shorthand
            "vintage": "4",                     # same as above
            "layers": "93,80"}  # 93 is the layer code for MSA and 80 is the layer code for state

        census_bureau_response = requests.get(census_bureau_geocoding_url, params=census_bureau_geocoding_params)
        census_bureau_response.raise_for_status()

        # take response json data and use to po
        census_bureau_data = census_bureau_response.json()
        returned_geographies = census_bureau_data["result"]["geographies"]

        # get state geoid out of returned geographies
        state_geoid = returned_geographies["States"][0]["GEOID"]
        # can safely use first result because msas are mutually exclusive
        msa_geoid = returned_geographies["Metropolitan Statistical Areas"][0]["GEOID"]
        # get state name out of returned geographies
        state_name = returned_geographies["States"][0]["BASENAME"]

        # returning a dict: {"msa_geoid": geoid (str), "state_geoid": geoid (str)}
        geoids = {"msa_geoid": msa_geoid, "state_geoid": state_geoid, "state_name": state_name}
        return geoids

    # basically same function as above, just for csa
    def get_csa_from_lat_lon(place_lat_lon:dict):
        census_bureau_geocoding_params = {
            "x": place_lat_lon["lon"],
            "y": place_lat_lon["lat"],
            "format": "json",
            "benchmark": "Public_AR_Current",   # just the weird current version shorthand
            "vintage": "4",                     # same as above
            "layers": "97,80"}  # 97 is csa code and 80 is state

        census_bureau_response = requests.get(census_bureau_geocoding_url, params=census_bureau_geocoding_params)
        census_bureau_response.raise_for_status()

        # take response json data and use to po
        census_bureau_data = census_bureau_response.json()
        returned_geographies = census_bureau_data["result"]["geographies"]
        csa_data = returned_geographies["Combined Statistical Areas"][0]


        # can safely use first result for csa
        csa_geoid = csa_data["GEOID"]
        # get name of csa (for finding states)
        csa_name = csa_data["NAME"]

        # now the hard part, turning the name into a list of states in the csa. e.g., Chicago-Naperville, IL-IN-WI CSA
        csa_comma_separated_name = csa_name.split(", ") # note the space, so first char of last item is letter not space

        # since every csa is named "..., STATE ABBREVIATION(s) CSA", can just take last element of comma separated
        csa_states_with_csa = csa_comma_separated_name[-1]
        # now slice string so that we have just the state abbreviation(s) (i.e., take all but last four characters)
        concatenated_state_abbreviations = csa_states_with_csa[:-4]
        # the list of state names to return
        state_abbreviations = []
        # now check if multiple states in csa
        if "-" in concatenated_state_abbreviations:
            for state_abbreviation in concatenated_state_abbreviations.split("-"):
                state_abbreviations.append(state_abbreviation)
        else:
            state_abbreviations.append(concatenated_state_abbreviations)

        # using the state_fips_and_abbreviations dict, can now get state geoids and names
        state_geoids = []
        state_names = []
        for state_abbreviation in state_abbreviations:
            state_geoid = state_fips_and_abbreviations[state_abbreviation]["fips"]
            state_geoids.append(state_geoid)
            state_name = state_fips_and_abbreviations[state_abbreviation]["name"]
            state_names.append(state_name)


        # retunr dictionary with info on the csa (geoid, the states in the csa's names and geoids)
        csa_info = {"csa_geoid": csa_geoid, "state_geoids": state_geoids, "state_names": state_names}
        return csa_info

    def get_msas_from_csa(csa_info):
        # get csd info out of provided dict
        csa_geoid = csa_info["csa_geoid"]
        state_geoids = csa_info["state_geoids"]
        state_names = csa_info["state_names"]
        # the list of msa_geoid_dict dicts to return
        msa_geoid_list = []

        # the geoinfo census bureau api url needed for query
        geoinfo_url = (f"https://api.census.gov/data/2023/geoinfo?get=NAME&for=metropolitan%20statistical%20area/"
                       f"micropolitan%20statistical%20area:*&in=combined%20statistical%20area:{csa_geoid}"
                       f"&key={census_bureau_api_key}")

        # query the API
        geoinfo_response = requests.get(geoinfo_url)
        geoinfo_response.raise_for_status()
        geoinfo_data = geoinfo_response.json()

        # go through each msa returned (the first is always just the format so can skip)
        for msa in geoinfo_data[1:]:
            # just in case test to same msa[0] not "NAME"
            if msa[0] != "NAME":
                # geoid is 3rd element of list returned by api
                msa_geoid = msa[2]
                # need to create a dictionary for each msa with state name and geoid
                msa_geoid_dict = {"msa_geoid": msa_geoid, "state_geoids": state_geoids, "state_names": state_names}
                # now add to list
                msa_geoid_list.append(msa_geoid_dict)
        return msa_geoid_list

    def get_counties_from_msa(msa_geoids:list[dict]):
        """ Returns a set of county names (e.g. "Alameda County, California") for the msa provided"""

        # the set of counties to return
        counties_in_msa = set()

        # iterating through msas provided
        for msa in msa_geoids:
            # get data out of the msa dict
            msa_geoid = msa["msa_geoid"]
            state_geoids = msa["state_geoids"]
            state_names = msa["state_names"]

            # because we do not know which state the msa is in so need to try each state with each msa
            for state_geoid, state_name in zip(state_geoids, state_names):
                logging.debug(f"Trying state geoid {state_geoid} for state {state_name} for MSA {msa_geoid}")
                # easier to just define new url for each msa rather than try to set parameters with base url
                geoinfo_counties_url = (f"https://api.census.gov/data/2023/geoinfo?get=NAME&for=county:*&in=metropolitan%20"
                                        f"statistical%20area/micropolitan%20statistical%20area:{msa_geoid}%20"
                                        f"state%20(or%20part):{state_geoid}&key={census_bureau_api_key}")

                # query the API
                geoinfo_counties_response = requests.get(geoinfo_counties_url)

                # check if the response code is 200, meaning actually has data (otherwise no data and will fail)
                if geoinfo_counties_response.status_code == 200:
                    logging.debug(f"State geoid {state_geoid} for state {state_name} for MSA {msa_geoid} was successful")
                    # since actually has data, can turn response in json
                    geoinfo_counties_data = geoinfo_counties_response.json()
                    # go through each county returned (the first is always just the format so can skip)
                    for county in geoinfo_counties_data[1:]:
                        # just in case test to same county[0] not "NAME"
                        if county[0] != "NAME":
                            county_no_state_name = county[0].split(";")[0]
                            counties_in_msa.add(f"{county_no_state_name}, {state_name}")

        return counties_in_msa

    # now main part of code
    if geographic_scope == "msa":
        this_lat_lon = get_lat_lon(place_name=place_name)
        this_msa = get_msa_from_lat_lon(this_lat_lon)
        these_county_names = get_counties_from_msa([this_msa]) # remember, am passing a list of msas

    elif geographic_scope == "csa":
        this_lat_lon = get_lat_lon(place_name=place_name)
        this_csa = get_csa_from_lat_lon(this_lat_lon)
        these_msas = get_msas_from_csa(this_csa)
        these_county_names = get_counties_from_msa(these_msas)

    elif geographic_scope == "county":
        this_lat_lon = get_lat_lon(place_name=place_name)
        these_county_names = get_county_from_lat_lon(this_lat_lon)

    else:
        raise Exception("Geographic scope must be either 'city', 'county', 'msa', or 'csa'")

    # list to return
    county_reference_places = [ReferencePlace(place_name=place_name)] # starting off with original place name RefPlace

    # go through county names and make ReferencePlaces
    for county_name in these_county_names:
        county_reference_places.append(ReferencePlace(place_name=county_name))

    logging.info(f"Found {len(county_reference_places) - 1} counties for {place_name} in {geographic_scope}")

    # return list of ReferencePlaces (counties)
    return county_reference_places

# function to check that a point is a valid coordinate
def check_if_valid_coordinate_point(point: tuple) -> None:
    """
    Used to check that a point is a valid coordinate (i.e. a tuple or list of two values that are
    both floats and match lat and lon requirements)

    :param point: tuple or list of two values (latitude, longitude)
    :return: None
    """

    # in case a point is not a list or tuple
    if not isinstance(point, tuple):
        raise ValueError("each point must be a tuple (or list) of (latitude, longitude)")

    # otherwise (point is either a list or tuple)
    else:
        # check whether the tuple or list has exactly two values
        if len(point) != 2:
            raise ValueError("each point must be a tuple (or list) of (latitude, longitude). You passed "
                             "a tuple or list with fewer or more than two values")
        # point has two values, so check that they are valid lat lon values
        else:
            # check that each coordinate is a float
            for coordinate in point:
                if not isinstance(coordinate, float):
                    raise ValueError("points must contain two float values")

                # check that lat and lon values are within valid ranges
            if point[0] < -90 or point[0] > 90:
                raise ValueError("first value for must be a valid latitude (between -90 and 90)")
            if point[1] < -180 or point[1] > 180:
                raise ValueError("second value for point must be a valid longitude (between -180 and 180)")

def get_coordinates_from_address(address: list[str] | str):
    """
    Takes a single address or list of addresses as input and then turns it into a tuple of (latitude, longitude)
    using nominatim API
    """

    # set up for query
    nominatim_url = "https://nominatim.openstreetmap.org/search"

    # first in case where only one address provided
    if isinstance(address, str):
        logging.info(f"Geocoding input address {address}")
        # again using q for open ended search which will return a bunch of characteristics of the address
        nominatim_params = {
            "q": address,
            "limit": "1",
            "format": "json"}

        # again, need to use user agent for nominatim API (but can be an string value????)
        nominatim_headers = {"User-Agent": nominatim_user_agent}

        # now can actually query the API
        nominatim_response = requests.get(nominatim_url, params=nominatim_params, headers=nominatim_headers)
        nominatim_response.raise_for_status()
        nominatim_data = nominatim_response.json()

        # if nominatim couldn't find anywhere, then address was improperly formed
        if len(nominatim_data) == 0:
            raise ValueError(f"Address {address} could not be geocoded, "
                             f"please double check that the address is correct")

        # get lat and lon out of response
        nominatim_content = nominatim_data[0]
        address_lat = float(nominatim_content["lat"])
        address_lon = float(nominatim_content["lon"])

        # now define address coordinate tuple to return
        address_coordinates = (address_lat, address_lon)
        logging.info(f"Successfully geocoded address: {address}, lat, lon = {address_coordinates}")
        return address_coordinates

    # in case where multiple addresses provided
    elif isinstance(address, list):
        # list of coordinate tuples to return
        all_address_coordinates = []
        # again using q for open ended search which will return a bunch of characteristics of the address
        for individual_address in address:
            logging.info(f"Geocoding input addresses {individual_address}")
            # again using q for open ended search which will return a bunch of characteristics of the address
            nominatim_params = {
                "q": individual_address,
                "limit": "1",
                "format": "json"}

            # again, need to use user agent for nominatim API (but can be any string value????)
            nominatim_headers = {"User-Agent": nominatim_user_agent}

            # now can actually query the API
            nominatim_response = requests.get(nominatim_url, params=nominatim_params, headers=nominatim_headers)
            nominatim_response.raise_for_status()
            nominatim_data = nominatim_response.json()

            # if nominatim couldn't find anywhere, then address was improperly formed
            if len(nominatim_data) == 0:
                raise ValueError(
                    f"Address {address} could not be geocoded, please double check that the address is correct")

            # get lat and lon out of response
            nominatim_content = nominatim_data[0]
            ind_address_lat = float(nominatim_content["lat"])
            ind_address_lon = float(nominatim_content["lon"])

            individual_address_coordinates:tuple = (ind_address_lat, ind_address_lon)

            # now define address coordinate tuple to return
            all_address_coordinates.append(individual_address_coordinates)

            # need to sleep because rate limits of nominatim API
            from time import sleep as sleepytime
            sleepytime(1.1)
        logging.info(f"Successfully geocoded {len(address)} addresses")
        return all_address_coordinates

    # in weird third case
    else:
        raise ValueError("Must provide either a string or a list of strings")

def generate_random_base64_value(input_number) -> str:
# generate random scenario id
        random_float = random.random()
        random_integer_value = str(int(input_number * random_float)) # using random number between 0 and 1 billion
        random_base64_value = base64.b64encode(random_integer_value.encode()).decode()
        random_base64_value.replace("/", "xx") # replacing dash just to be safe

        return random_base64_value

def turn_seconds_into_minutes(seconds: float) -> str:
    minutes = int(seconds / 60)
    remaining_seconds = seconds % 60
    if minutes == 0:
        return f"{remaining_seconds:.2f} seconds"
    else:
        return f"{minutes} minutes, {remaining_seconds:.2f} seconds"



if __name__ == "__main__":
    san_francisco_ref_place = ReferencePlace(place_name="San Francisco, California, USA",
                                             bound_box=(-122.4194, 37.7749, -122.3731, 37.8091))
    print(san_francisco_ref_place.plus_codes_for_bbox_corners)


