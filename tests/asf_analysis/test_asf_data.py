import os
from pathlib import Path

import numpy as np
import pytest

from asf_analysis.asf_data import ASFData, ASFDataFile, parse_asf_data_from_filename, ASF_CHUNK_ORDER


@pytest.fixture
def s1a_filename():
    yield (
        "./data/sentinel-1/S1A_IW_SLC__1SDV_20180606T141441_20180606T141508_022237_0267EB_EC03.SAFE"
        "/measurement/s1a-iw1-slc-vh-20180606t141443-20180606t141508-022237-0267eb-001.tiff"
    )


@pytest.fixture
def s1a_asf_datafile(s1a_filename):
    yield ASFDataFile(s1a_filename)


@pytest.mark.unit
def test_parse_asf_data_from_filename_str(s1a_filename):
    meta_dict = parse_asf_data_from_filename(s1a_filename)

    attrs = ASF_CHUNK_ORDER
    for attr in attrs:
        assert attr in meta_dict


@pytest.mark.unit
def test_parse_asf_data_from_filename_path(s1a_filename):
    s1a_file_path = Path(s1a_filename)
    meta_dict = parse_asf_data_from_filename(s1a_file_path)

    attrs = ASF_CHUNK_ORDER
    for attr in attrs:
        assert attr in meta_dict


class TestASFDataFile:
    @pytest.mark.unit
    def test_create_asf_data_file(self, s1a_filename):
        asf_datafile = ASFDataFile(s1a_filename)

    @pytest.mark.unit
    def test_asf_data_file_has_attrs(self, s1a_asf_datafile):
        for attr in ASF_CHUNK_ORDER:
            assert attr in s1a_asf_datafile.asf_metadata

        assert s1a_asf_datafile.polarization == "vh"
        assert s1a_asf_datafile.beam_mode == "iw1"

    @pytest.mark.unit
    def test_asf_data_find_file_stem(self, s1a_filename):
        true_stem = s1a_filename.split("/")[-1].split(".")[0]
        assert ASFDataFile._find_filestem(s1a_filename) == true_stem

    @pytest.mark.unit
    def test_asf_data_file_locates_annotation_file(self, s1a_asf_datafile):
        assert hasattr(s1a_asf_datafile, "annotation_filename")
        assert isinstance(s1a_asf_datafile.annotation_filename, str)
        assert os.path.isfile(s1a_asf_datafile.annotation_filename)

    @pytest.mark.unit
    def test_has_geo_bounds(self, s1a_asf_datafile):
        assert hasattr(s1a_asf_datafile, "geo_bounds")
        geo_bounds = s1a_asf_datafile.geo_bounds

        assert isinstance(geo_bounds, np.ndarray)
        assert geo_bounds.shape[0] == 4
        assert geo_bounds.shape[1] == 2

        actual_geo_bounds = np.array([
            [41.08364918, -120.83964114],
            [42.74025269, -121.51561604],
            [41.23010369, -121.88101754],
            [42.593359434, -120.4492047],
        ])

        assert np.allclose(geo_bounds, actual_geo_bounds)


class TestASFData:
    @pytest.mark.unit
    def test_create_asf_data(self, tmpdir):
        asf_data = ASFData()
        assert asf_data.filepath is None

    @pytest.mark.unit
    def test_load_asf_file(self, tmpdir, s1a_filename):
        path_to_file = tmpdir + s1a_filename
        asf_data = ASFData()
        asf_data.load(path_to_file)

        assert asf_data.filepath == path_to_file

    @pytest.mark.unit
    def test_parse_info_from_filename(self, s1a_filename):
        asf_data = ASFData()
        asf_data.parse_info_from_filename(s1a_filename)

        assert asf_data.absolute_orbit_number == "022237"

    @pytest.mark.unit
    @pytest.mark.skip(reason="may change implementation details...")
    def test_asf_data_has_annotation_xml(self):
        asf_data = ASFData()
        asf_data.load("./data/sentinel-1/")


# my_data = ASFData()
#
# my_data.load(filepath)