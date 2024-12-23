import os
import zipfile
from pathlib import Path
from typing import Union

import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from scipy.interpolate import griddata

ASF_CHUNK_ORDER = [
    "mission",
    "beam_mode",
    "product_type_resolution",
    "process_level_class_polarization",
    "start_date_time",
    "end_date_time",
    "absolute_orbit_number",
    "mission_data_take_id",
    "product_unique_identifier",
]


def parse_asf_data_from_filename(filename: Union[str, Path]) -> dict:
    if not isinstance(filename, Path):
        try:
            filepath = Path(filename)
        except TypeError as te:
            raise TypeError(f"Unable to convert input (type: {type(filename)}) to path. Supply input as str or Path.")

    else:
        filepath = filename

    stem = filepath.stem
    chunks = stem.split("-")

    data_dict = {}

    for chunk_key, chunk in zip(ASF_CHUNK_ORDER, chunks):
        data_dict[chunk_key] = chunk

    return data_dict


class ASFDataFile:
    """
    Represents a single .tiff file and associated annotations.

    Loads and maintains reference to annotation information without needing to load measurement data.

    Loads measurement data into memory when explicitly called.
    """
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.filestem = self._find_filestem(filename)
        self.asf_metadata = parse_asf_data_from_filename(filename)

        self.polarization = self.asf_metadata["process_level_class_polarization"]
        self.beam_mode = self.asf_metadata["beam_mode"]

        self.geolocation_array = np.array([])
        self.geo_bounds = np.zeros((4, 2))
        self.data_array = None

        self.annotation_filepath = None
        self.annotation_filename = None

        self._load_annotation_file()
        self._load_annotation_data()

    @staticmethod
    def _find_filestem(filename: str) -> str:
        filepath = Path(filename)
        return filepath.stem

    def _load_annotation_file(self) -> None:
        datafile_path = Path(self.filename)
        product_path = datafile_path.parent.parent
        annotation_path = product_path / "annotation" / f"{self.filestem}.xml"
        self.annotation_filepath = annotation_path
        self.annotation_filename = str(annotation_path)

    def _load_annotation_data(self) -> None:
        ann_tree = ET.parse(self.annotation_filepath)
        ann_root = ann_tree.getroot()
        geolocation_grid_element = ann_root.find("geolocationGrid")
        grid_point_list = geolocation_grid_element.find("geolocationGridPointList")
        geolocation_array = np.array([
            (
                int(p.find("line").text),
                int(p.find("pixel").text),
                float(p.find("latitude").text),
                float(p.find("longitude").text)
            ) for p in grid_point_list.findall("geolocationGridPoint")
        ])

        four_corner_indices = [
            np.argmin(geolocation_array[:, 2]),
            np.argmax(geolocation_array[:, 2]),
            np.argmin(geolocation_array[:, 3]),
            np.argmax(geolocation_array[:, 3]),
        ]

        four_corner_elements = np.take(geolocation_array, four_corner_indices, axis=0)
        four_corner_lines = four_corner_elements[:, 0]
        four_corner_pixels = four_corner_elements[:, 1]
        four_corner_coords = four_corner_elements[:, 2:]

        self.geolocation_array = geolocation_array
        self.geo_bounds = four_corner_coords
        self.corner_lines = four_corner_lines
        self.corner_pixels = four_corner_pixels

    def load_data_array(self) -> None:
        """
        Loads measurement data into self.data_array.
        """
        if self.data_array is None:
            with rasterio.open(self.filename, "r") as raster_data:
                self.data_array = raster_data.read(1)

    def get_geo_arrays(self, line_skip: int = 1, pixel_skip: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute geo-referenced data array for the given sampling.

        Parameters
        ----------
        line_skip : int
            Number of elements to skip along "lines" axis
            i.e. 2 -> every other element from data_array is skipped.
            Default: 1 (no skips, return full array)
        pixel_skip : int
            Number of elements to skip along "pixels" axis
            Default: 1 (no skips, return full array)
        Returns
        -------
        tuple (3) numpy arrays of shape data_array[::line_skip, ::pixel_skip]
            (np.ndarray): latitude of data points
            (np.ndarray): longitude of data points
            (np.ndarray): data points
        """
        d_shape = self.data_array.shape
        lines = self.geolocation_array[:, 0]
        pixels = self.geolocation_array[:, 1]
        lats = self.geolocation_array[:, 2]
        lons = self.geolocation_array[:, 3]

        # Interpolate input only to desired output resolution
        target_lines, target_pixels = np.meshgrid(
            np.arange(0, d_shape[0], line_skip),
            np.arange(0, d_shape[1], pixel_skip),
            indexing="ij"
        )
        lat_grid = griddata((lines, pixels), lats, (target_lines, target_pixels), method="linear")
        lon_grid = griddata((lines, pixels), lons, (target_lines, target_pixels), method="linear")

        reduced_data_array = self.data_array[::line_skip, ::pixel_skip]

        return lat_grid, lon_grid, reduced_data_array


class ASFDataScene:
    """
    Represents a full product directory, including the .zip archive file downloaded from the ASF database

    Contains Data:
    - data files (measurement): .tiff
    - annotation files (annotation): .xml
    - metadata from annotation files, filestem, etc.
    - preview imagery

    Contains methods for rendering preview image, finding geographic extents of sub-measurements, etc.
    """
    def __init__(self, zipfilename: str) -> None:
        self.zipfilename = zipfilename
        self.root_dir = None

        self.data_filenames = []
        self.annotation_filenames = []
        self._parse_zip_file()

    def _parse_zip_file(self) -> None:
        if not os.path.isfile(self.zipfilename):
            raise ValueError(f"Could not find zip archive file at: {self.zipfilename}")
        data_stems = []
        with zipfile.ZipFile(self.zipfilename, "r") as zip_file:
            for z_name in zip_file.namelist():
                z_path = Path(z_name)
                if z_path.suffix == ".tiff":
                    self.data_filenames.append(str(z_path))
                    data_stems.append(z_path.stem)
            for z_name in zip_file.namelist():
                z_path = Path(z_name)
                if z_path.stem in data_stems and z_path.suffix == ".xml":
                    self.annotation_filenames.append(str(z_path))

    # TODO: Add logic for detecting number of sub-swaths and polarization for sub-swaths?

    # TODO: Add method for rendering preview image (preview / quick-look.png) along with North arrow?

    # TODO: Add method for getting 4-point geocode lat/long from preview (very rough)

    # TODO: Add method for finding geographic bounds of all sub-measurements and polarizations


class ASFData:
    """
    Deprecating this in favor of more specific class names.

    Will move most logic into `ASFDataScene` class.
    """
    def __init__(self) -> None:
        self.filepath = None
        self.file_stem = None

        self.mission = None
        self.beam_mode = None
        self.product_type_resolution = None
        self.process_level_class_polarization = None
        self.start_date_time = None
        self.end_date_time = None
        self.absolute_orbit_number = None
        self.mission_data_take_id = None
        self.product_unique_identifier = None

    # TODO: Change to reference scene directory, i.e. S1A...SAFE
    #  This way, can load any of the tiffs or annotations.
    #  May need a Scene class?
    def load(self, filepath: str) -> None:
        self.filepath = filepath
        filepath_p = Path(filepath)
        self.file_stem = filepath_p.stem
        self.parse_info_from_filename(self.file_stem)

    def parse_info_from_filename(self, filename: Union[str, Path]) -> None:
        data_dict = parse_asf_data_from_filename(filename)

        for k, v in data_dict.items():
            self.__setattr__(k, v)
