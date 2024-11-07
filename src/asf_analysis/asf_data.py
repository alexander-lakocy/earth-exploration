from pathlib import Path


class ASFData:
    def __init__(self):
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

    def load(self, filepath: str) -> None:
        self.filepath = filepath
        filepath_p = Path(filepath)
        self.file_stem = filepath_p.stem
        self.parse_info_from_stem(self.file_stem)

    def parse_info_from_stem(self, stem: str) -> None:
        chunks = stem.split("-")
        self.mission = chunks[0]
        self.beam_mode = chunks[1]
        self.product_type_resolution = chunks[2]
        self.process_level_class_polarization = chunks[3]
        self.start_date_time = chunks[4]
        self.end_date_time = chunks[5]
        self.absolute_orbit_number = chunks[6]
        self.mission_data_take_id = chunks[7]
        self.product_unique_identifier = chunks[8]