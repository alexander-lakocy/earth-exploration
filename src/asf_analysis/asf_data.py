import glob
import logging
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import os
from pathlib import Path
from scipy.interpolate import griddata
from typing import Union
import xml.etree.ElementTree as ET
import zipfile


module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

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
        """Locate annotation file in ASF tree structure."""
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

    def _confirm_data_loaded(self) -> None:
        if self.data_array is None:
            raise ValueError(f"Data array not loaded yet. Call ASFDataFile.load_data_array() first.")

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
        self._confirm_data_loaded()

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

    def plot_magnitude_data(self, log_transform: bool = False, *args, **kwargs) -> Axes:
        self._confirm_data_loaded()
        lat_grid, lon_grid, reduced_data_array = self.get_geo_arrays(50, 50)
        lat_f, lon_f, data_f = lat_grid.flatten(), lon_grid.flatten(), reduced_data_array.flatten()

        title_prefix = ""
        if log_transform:
            data_f = np.log(data_f + 1)
            title_prefix = "Log "

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.scatter(lon_f, lat_f, c=np.abs(data_f), s=1, *args, **kwargs)
        ax.set_title(f"{title_prefix}Magnitude Data: {self.polarization.upper()}-{self.beam_mode.upper()}")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        fig.suptitle(f"{self.filestem}")
        return ax

    # TODO: Add phase plot method
    # def plot_phase_data(self, *args, **kwargs) -> Axes:
    #     self._confirm_data_loaded()


def make_dir_if_not_exists(extract_location: str) -> None:
    if not os.path.isdir(extract_location):
        os.mkdir(extract_location)


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

    # TODO: Write method to check that contents are all present
    def __init__(self, safe_path: Union[str, Path], zip_path: Union[str, Path] = None, extracted: bool = False) -> None:
        """
        Default constructor for ASFDataScene.

        Parameters
        ----------
        safe_path: Path or str to .SAFE directory for extracted files
        zip_path
        extracted
        """

        if not isinstance(safe_path, Path):
            safe_path = Path(safe_path)
        self.safe_path = safe_path
        self.extracted = extracted

        if zip_path is not None:
            if not isinstance(zip_path, Path):
                zip_path = Path(zip_path)
            self.zip_path = zip_path
        else:
            self.zip_path = None

        self._local_search_for_zipfile(safe_path)

        self.asf_data_files = []

        self.data_filenames = []
        self.annotation_filenames = []
        self.corner_coordinates = []

        if not self.extracted:
            self._parse_zip_file()
        else:
            self._parse_safe_file()

    @classmethod
    def from_zipfile(cls, zip_path: Union[str, Path], safe_path: Union[str, Path] = None):
        """
        Factory method for creating ASFDataScene directly from zip archive file.

        Assumes zip archive has not been extracted. If it has been, use default constructor: ASFDataScene() with path
         to SAFE directory as the argument.

        Parameters
        ----------
        zip_path
        safe_path

        Returns
        -------
        ASFDataScene instance created from zip archive file.

        """
        if safe_path is None:
            safe_path = cls._get_safe_path(zip_path)
            if safe_path is None:
                zip_path = Path(zip_path)
                safe_path = zip_path.parent / f"{zip_path.stem}.SAFE"
            extracted = False

        extracted = os.path.isdir(safe_path)

        asf_datascene = cls(safe_path, zip_path=zip_path, extracted=extracted)

        if asf_datascene.zip_path is None:
            asf_datascene.zip_path = Path(zip_path).absolute()

        return asf_datascene

    def _local_search_for_zipfile(self, safe_path: Union[str, Path]) -> None:
        print(f"_local_search_for_zipfile: {safe_path = }")
        if not isinstance(safe_path, Path):
            safe_path = Path(safe_path)

        search_str = str(safe_path.parent / safe_path.stem) + ".zip"
        print(f"_local_search_for_zipfile: {search_str = }")
        zips = glob.glob(search_str)
        print(f"_local_search_for_zipfile: {zips = }")
        if len(zips) > 0:
            zip_filename = zips[0]
            self.zip_path = Path(zip_filename).absolute()

    @staticmethod
    def _get_safe_path(zip_path: Union[str, Path]) -> Path | None:
        if not isinstance(zip_path, Path):
            zip_path = Path(zip_path)

        safe = glob.glob(str(zip_path.parent / zip_path.stem) + ".SAFE")
        if len(safe) > 0:
            return Path(safe[0]).absolute()
        else:
            return None

    def _parse_zip_file(self) -> None:
        if not os.path.isfile(self.zip_path):
            raise ValueError(f"Could not find zip archive file at: {self.zip_path}")
        data_stems = []

        with zipfile.ZipFile(self.zip_path, "r") as zip_file:
            # Find measurement files
            for z_name in zip_file.namelist():
                z_path = Path(z_name)
                if z_path.suffix == ".tiff":
                    # asf_data_file = ASFDataFile(str(z_path))
                    # self.asf_data_files.append(asf_data_file)
                    self.data_filenames.append(str(z_path))
                    data_stems.append(z_path.stem)
            # Find associated annotation files
            for z_name in zip_file.namelist():
                z_path = Path(z_name)
                if z_path.stem in data_stems and z_path.suffix == ".xml":
                    self.annotation_filenames.append(str(z_path))

    def _parse_safe_file(self) -> None:
        if not os.path.isdir(self.safe_path):
            raise ValueError(f"Could not find SAFE directory at: {self.safe_path}")
        data_stems = []

        measure_globs = glob.glob(str(self.safe_path / "measurement" / "*.tiff"))
        for m_g in measure_globs:
            asf_data_file = ASFDataFile(str(m_g))
            self.asf_data_files.append(asf_data_file)
            self.data_filenames.append(str(Path(m_g).absolute()))

        annotate_globs = glob.glob(str(self.safe_path / "annotation" / "*.xml"))
        for a_g in annotate_globs:
            self.annotation_filenames.append(str(Path(a_g).absolute()))

    def extract(self, extract_location: str = None, **kwargs) -> None:
        """Pulls files out of zip archive into extract_location.
        Defaults to pull all files, can include `members` kwarg for specific files/directories only.
        """
        if extract_location is None:
            extract_location = self.safe_path.parent
        else:
            make_dir_if_not_exists(extract_location)

        with zipfile.ZipFile(self.zip_path, "r") as zip_file:
            zip_file.extractall(path=extract_location, **kwargs)

        # root_stem = Path(self.zipfilename).stem
        # self.safe_path = Path(extract_location) / (str(root_stem) + ".SAFE")
        self.extracted = True

    def locate(self) -> None:
        map_overlay_file = self.safe_path / "preview" / "map-overlay.kml"
        if not self.extracted:
            module_logger.warning(f"Archive not extracted yet. Try calling ASFDataScene.extract() first.")
        if not os.path.exists(map_overlay_file):
            raise FileNotFoundError(f"File not found at: {map_overlay_file}")
        self.corner_coordinates = self._get_corner_coords_from_map_overlay(map_overlay_file)

    @staticmethod
    def _get_corner_coords_from_map_overlay(kml_file: str) -> list:
        tree = ET.parse(kml_file)
        root = tree.getroot()
        namespaces = {"gx": "http://www.google.com/kml/ext/2.2"}
        coordinates_tag = root.find(".//gx:LatLonQuad/coordinates", namespaces)
        coordinates_text = coordinates_tag.text.strip()
        coordinates = [
            tuple(map(float, coord.split(","))) for coord in coordinates_text.split()
        ]
        return coordinates

    # TODO: Add logic for detecting number of sub-swaths and polarization for sub-swaths?

    # TODO: Add North arrow? and center coordinates?
    def plot_preview(self, *args, **kwargs) -> Axes:
        img = plt.imread(self.safe_path / "preview" / "quick-look.png")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img, *args, **kwargs)
        return ax

    # TODO: Add method for finding geographic bounds of all sub-measurements and polarizations
