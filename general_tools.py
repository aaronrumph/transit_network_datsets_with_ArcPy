"""
This module contains helpful helper functions that can be used in any given module with no circular import
problems
"""
import re
import pluscodes


# regex matching patterns for use in checking other stuff
regex_matching_patterns = {
    # regex pattern to match valid longitudes built using regex generator because regex hard:
    "longitude":
        r"^[-+]?(180(\.0+)?|1[0-7]\d(\.\d+)?|\d{1,2}(\.\d+)?)$",

    # regex pattern to match valid latitudes
    "latitude":
        r"^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)$"}


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
        self.plus_codes_for_bbox_corners:list = []

        # check that either a bound box or a place name has been provided:
        if self.place_name is None and self.bound_box is None:
            raise ValueError("Reference place arguments (place_name and bound_box) cannot both be undefined")

        # this part is just exception handling because everything else follows from this, so need correct ReferencePlace
        # first checking that bound_box was provided
        if self.bound_box is not None:

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
            return name.place_name.replace(" ", "_").replace(
                                            ",", "").replace(r"/", "").lower()

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


if __name__ == "__main__":
    san_francisco_ref_place = ReferencePlace(place_name="San Francisco, California, USA",
                                             bound_box=(-122.4194, 37.7749, -122.3731, 37.8091))
    print(san_francisco_ref_place.plus_codes_for_bbox_corners)
