import glob
import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from asf_analysis.asf_data import ASFData, ASFDataFile, parse_asf_data_from_filename, ASF_CHUNK_ORDER


@pytest.fixture
def s1a_all_filenames():
    yield glob.glob("./data/sentinel-1/S1A_IW_SLC__1SDV_20180606T141441_*.SAFE/measurement/*.tiff")


@pytest.fixture
def single_s1a_filename():
    yield (
        "./data/sentinel-1/S1A_IW_SLC__1SDV_20180606T141441_20180606T141508_022237_0267EB_EC03.SAFE"
        "/measurement/s1a-iw1-slc-vh-20180606t141443-20180606t141508-022237-0267eb-001.tiff"
    )


@pytest.fixture
def s1a_asf_datafile(single_s1a_filename):
    yield ASFDataFile(single_s1a_filename)


@pytest.mark.unit
def test_parse_asf_data_from_filename_str(single_s1a_filename):
    meta_dict = parse_asf_data_from_filename(single_s1a_filename)

    attrs = ASF_CHUNK_ORDER
    for attr in attrs:
        assert attr in meta_dict


@pytest.mark.unit
def test_parse_asf_data_from_filename_path(single_s1a_filename):
    s1a_file_path = Path(single_s1a_filename)
    meta_dict = parse_asf_data_from_filename(s1a_file_path)

    attrs = ASF_CHUNK_ORDER
    for attr in attrs:
        assert attr in meta_dict


class TestASFDataFile:
    @pytest.mark.unit
    def test_create_asf_data_file(self, single_s1a_filename):
        asf_datafile = ASFDataFile(single_s1a_filename)

    @pytest.mark.unit
    def test_asf_data_file_has_attrs(self, s1a_asf_datafile):
        for attr in ASF_CHUNK_ORDER:
            assert attr in s1a_asf_datafile.asf_metadata

        assert s1a_asf_datafile.polarization == "vh"
        assert s1a_asf_datafile.beam_mode == "iw1"

    @pytest.mark.unit
    def test_asf_data_find_file_stem(self, single_s1a_filename):
        true_stem = single_s1a_filename.split("/")[-1].split(".")[0]
        assert ASFDataFile._find_filestem(single_s1a_filename) == true_stem

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

    @pytest.mark.plot
    def test_see_multiple_data_file_extents(self, s1a_all_filenames):
        fig, ax = plt.subplots(1, 1)  #, sharey="row")
        for filename in s1a_all_filenames:
            asf_datafile = ASFDataFile(filename)
            ax.scatter(
                asf_datafile.geo_bounds[:, 1],
                asf_datafile.geo_bounds[:, 0],
                label=asf_datafile.beam_mode + asf_datafile.polarization
            )
        ax.legend()
        plt.show()

    @pytest.mark.unit
    def test_load_data_array(self, single_s1a_filename):
        asf_datafile = ASFDataFile(single_s1a_filename)
        asf_datafile.load_data_array()
        data_array = asf_datafile.data_array

        assert isinstance(data_array, np.ndarray)
        assert data_array.dtype == np.complex64
        assert data_array.shape == (13473, 21198)

        assert len(asf_datafile.corner_lines) == 4
        assert len(asf_datafile.corner_pixels) == 4

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "line_skip_values, pixel_skip_values",
        [
            (4, 4),
            (100, 100),
            (50, 200),
            (217, 505),
        ]
    )
    def test_get_geo_arrays_downsampled(self, single_s1a_filename, line_skip_values, pixel_skip_values):
        asf_datafile = ASFDataFile(single_s1a_filename)
        asf_datafile.load_data_array()

        lat_grid, lon_grid, data_array = asf_datafile.get_geo_arrays(line_skip=line_skip_values,
                                                                     pixel_skip=pixel_skip_values)
        right_shape = (np.ceil(13473 / line_skip_values), np.ceil(21198 / pixel_skip_values))

        assert lat_grid.shape == right_shape
        assert lon_grid.shape == right_shape
        assert data_array.shape == right_shape

    @pytest.mark.plot
    def test_plot_downsampled_geo_arrays(self, single_s1a_filename):
        line_skip_values = 50
        pixel_skip_values = 50
        asf_datafile = ASFDataFile(single_s1a_filename)
        asf_datafile.load_data_array()

        lat_grid, lon_grid, data_array = asf_datafile.get_geo_arrays(line_skip=line_skip_values,
                                                                     pixel_skip=pixel_skip_values)

        lat_f = lat_grid.flatten()
        lon_f = lon_grid.flatten()
        data_f = data_array.flatten()

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].scatter(lon_f, lat_f, c=np.abs(data_f), s=1)
        axs[0].set_title(
            f"Magnitude data: {asf_datafile.polarization.upper()}-{asf_datafile.beam_mode.upper()}"
        )
        axs[1].scatter(lon_f, lat_f, c=np.angle(data_f), s=1)
        axs[1].set_title(
            f"Phase data: {asf_datafile.polarization.upper()}-{asf_datafile.beam_mode.upper()}"
        )
        fig.suptitle(asf_datafile.filestem)
        plt.show()


# TODO: Update this to hold entire .SAFE directory data
#  Add methods for "mosaicing" multiple georeferenced coords into a plottable structure?
#  Pre-prepared plotting methods (Magnitude top, phase bottom; columns for VH versus VV)
#  False color composite ability
class TestASFData:
    @pytest.mark.unit
    def test_create_asf_data(self, tmpdir):
        asf_data = ASFData()
        assert asf_data.filepath is None

    @pytest.mark.unit
    def test_load_asf_file(self, tmpdir, single_s1a_filename):
        path_to_file = tmpdir + single_s1a_filename
        asf_data = ASFData()
        asf_data.load(path_to_file)

        assert asf_data.filepath == path_to_file

    @pytest.mark.unit
    def test_parse_info_from_filename(self, single_s1a_filename):
        asf_data = ASFData()
        asf_data.parse_info_from_filename(single_s1a_filename)

        assert asf_data.absolute_orbit_number == "022237"

    @pytest.mark.unit
    @pytest.mark.skip(reason="may change implementation details...")
    def test_asf_data_has_annotation_xml(self):
        asf_data = ASFData()
        asf_data.load("./data/sentinel-1/")


# my_data = ASFData()
#
# my_data.load(filepath)