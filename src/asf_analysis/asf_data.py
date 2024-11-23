from pathlib import Path
from typing import Union

import xml.etree.ElementTree as ET

import numpy as np

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
        self.geo_bounds = np.zeros((4, 2))

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

        four_corner_coords = np.take(geolocation_array, four_corner_indices, axis=0)[:, 2:]

        self.geo_bounds = four_corner_coords


class ASFData:
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
