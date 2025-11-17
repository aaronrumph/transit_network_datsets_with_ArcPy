
import os

def set_up_arcpy_env():
    arcgis_bin = r"C:\Program Files\ArcGIS\Pro\bin"
    arcgis_extensions = r"C:\Program Files\ArcGIS\Pro\bin\Extensions"
    os.environ["PATH"] = arcgis_bin + os.pathsep + arcgis_extensions + os.pathsep + os.environ.get("PATH", "")
