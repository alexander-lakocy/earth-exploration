import glob
import os
import shutil
import time
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt

from asf_analysis.asf_data import ASFDataFile, parse_asf_data_from_filename, ASF_CHUNK_ORDER, ASFDataScene


@pytest.fixture(scope="session")
def s1a_all_filenames():
    yield glob.glob("./data/sentinel-1/S1A_IW_SLC__1SDV_20180606T141441_*.SAFE/measurement/*.tiff")


@pytest.fixture(scope="session")
def single_s1a_filename():
    yield (
        "./data/sentinel-1/S1A_IW_SLC__1SDV_20180606T141441_20180606T141508_022237_0267EB_EC03.SAFE"
        "/measurement/s1a-iw1-slc-vh-20180606t141443-20180606t141508-022237-0267eb-001.tiff"
    )


@pytest.fixture(scope="session")
def s1a_zipfile():
    yield "./tests/test_data/zip/S1A_IW_SLC__1SDV_20180606T141441_20180606T141508_022237_0267EB_EC03.zip"


@pytest.fixture(scope="session")
def s1a_safedir():
    yield "./data/sentinel-1/S1A_IW_SLC__1SDV_20180606T141441_20180606T141508_022237_0267EB_EC03.SAFE"


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


@pytest.mark.unit
def test_parse_info_from_filename(single_s1a_filename):
    info_dict = parse_asf_data_from_filename(single_s1a_filename)
    assert info_dict["mission"] == "s1a"
    assert info_dict["process_level_class_polarization"] == "vh"
    assert info_dict["absolute_orbit_number"] == "022237"
    assert info_dict["mission_data_take_id"] == "0267eb"


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
            # (4, 4),
            # (100, 100),
            # (50, 200),
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

    @pytest.mark.plot
    def test_plot_magnitude(self, single_s1a_filename):
        asf_datafile = ASFDataFile(single_s1a_filename)
        asf_datafile.load_data_array()
        ax = asf_datafile.plot_magnitude_data(log_transform=True, cmap="gray")
        plt.show()

    @pytest.mark.unit
    def test_confirm_data_loaded(self, single_s1a_filename):
        asf_datafile = ASFDataFile(single_s1a_filename)
        with pytest.raises(ValueError):
            asf_datafile._confirm_data_loaded()

        asf_datafile.load_data_array()
        asf_datafile._confirm_data_loaded()

@pytest.fixture(scope="function")
def asf_datascene(s1a_zipfile):
    asf_datascene = ASFDataScene.from_zipfile(s1a_zipfile)
    yield asf_datascene


@pytest.fixture(scope="session")
def asf_extracted_safe(s1a_zipfile):
    safe_stem = Path(s1a_zipfile).stem
    safe_dir = Path(s1a_zipfile).parent.parent / "safe"
    desired_safe_destination = safe_dir / (safe_stem + ".SAFE")
    asf_data_scene = ASFDataScene.from_zipfile(s1a_zipfile, safe_path=desired_safe_destination)
    # asf_data_scene.safe_path = Path(f"./tests/test_data/safe/{asf_data_scene.safe_path.name}")
    if not os.path.isdir(asf_data_scene.safe_path):
        print("EXTRACTING SAFE DATA FROM ZIP ARCHIVE...")
        asf_data_scene.extract()
    yield asf_data_scene.safe_path
    # As part of teardown, delete SAFE path contents. Will make tests run much longer...
    inuse = True
    while inuse:
        try:
            shutil.rmtree(asf_data_scene.safe_path)
            inuse = False
        except PermissionError as pe:
            print(f"File still in use:\n{pe}")
            time.sleep(2.0)  # Wait for system to realize it's done with this directory...

class TestASFDataScene:
    @pytest.mark.unit
    def test_create_asf_datascene_from_zipfile_only(self, s1a_zipfile):
        """
        1. Zip archive file only, no SAFE directory. Should instantiate safe_path in same parent folder.
        """
        asf_data_scene = ASFDataScene.from_zipfile(s1a_zipfile)
        assert asf_data_scene.zip_path.resolve() == Path(s1a_zipfile).resolve()
        assert not os.path.exists(asf_data_scene.safe_path)

    @pytest.mark.unit
    def test_create_asf_datascene_from_zipfile_explicit_safe_dir(self, s1a_zipfile, tmp_path):
        """
        2. Zip archive file only, with explicit path to safe dir (does not exist yet).
        """
        safe_stem = Path(s1a_zipfile).stem
        desired_safe_destination = tmp_path / (safe_stem + ".SAFE")
        asf_data_scene = ASFDataScene.from_zipfile(s1a_zipfile, safe_path=desired_safe_destination)
        assert asf_data_scene.safe_path.resolve() == desired_safe_destination.resolve()
        assert not os.path.exists(asf_data_scene.safe_path)

    @pytest.mark.unit
    def test_create_asf_datascene_from_SAFE(self, asf_extracted_safe):
        """
        3. SAFE directory only (no local zip). Zip path is None, parsing uses glob
        """
        asf_data_scene = ASFDataScene(asf_extracted_safe, extracted=True)
        assert isinstance(asf_data_scene, ASFDataScene)
        assert asf_data_scene.safe_path == asf_extracted_safe
        # assert Path(asf_datascene.zipfilename).absolute().parent == Path(s1a_zipfile).absolute().parent
        assert asf_data_scene.zip_path is None

    @pytest.mark.unit
    def test_create_asf_datascene_from_SAFE_with_zip(self, s1a_safedir):
        """
        4. SAFE directory and local zip. Expect that constructor finds zip archive and parses based on glob.
        """
        asf_data_scene = ASFDataScene(s1a_safedir, extracted=True)
        assert isinstance(asf_data_scene, ASFDataScene)
        assert asf_data_scene.safe_path.resolve() == Path(s1a_safedir).resolve()

        file_stem = Path(s1a_safedir).stem
        expected_zip_path = Path(s1a_safedir).parent / f"{file_stem}.zip"
        assert asf_data_scene.zip_path.resolve() == expected_zip_path.resolve()

    @pytest.mark.unit
    def test_asf_datascene_finds_data_files(self, asf_datascene):
        """On load, want ASFDataScene to index data files within ZIP archive"""
        assert hasattr(asf_datascene, "data_filenames")
        assert isinstance(asf_datascene.data_filenames, list)
        assert len(asf_datascene.data_filenames) == 6

    @pytest.mark.unit
    def test_asf_datascene_finds_annotation_files(self, asf_datascene):
        """On load, want ASFDataScene to index annotation files within ZIP archive"""
        assert hasattr(asf_datascene, "annotation_filenames")
        assert isinstance(asf_datascene.annotation_filenames, list)
        assert len(asf_datascene.annotation_filenames) == 6

    @pytest.mark.unit
    def test_asf_datascene_extract_selected_files(self, s1a_zipfile, tmp_path):
        out_dir = str(tmp_path / "S1A_test")
        asf_datascene = ASFDataScene.from_zipfile(s1a_zipfile)
        assert not (os.path.isdir(out_dir))
        # base = "S1A_IW_SLC__1SDV_20180606T141441_20180606T141508_022237_0267EB_EC03.SAFE/"
        subdirs = ["annotation/", "preview/", "support/"]
        zip_path = Path(s1a_zipfile)
        base = zip_path.stem + ".SAFE/"

        asf_datascene.extract(
            extract_location=out_dir,
            members=[base + subdir for subdir in subdirs]
        )
        assert os.path.isdir(out_dir)

        assert asf_datascene.safe_path is not None

        for member_dir in subdirs:
            assert os.path.isdir(f"{out_dir}\\{base}\\{member_dir}")

    @pytest.mark.unit
    def test_asf_datascene_has_four_geo_corners(self, s1a_safedir, s1a_zipfile, tmp_path):
        asf_datascene = ASFDataScene(s1a_safedir)
        assert asf_datascene.zip_path.resolve() == (Path(s1a_safedir).parent / f"{Path(s1a_safedir).stem}.zip").resolve()
        assert hasattr(asf_datascene, "corner_coordinates")
        assert isinstance(asf_datascene.corner_coordinates, list)

        zip_path = Path(s1a_zipfile)
        base = zip_path.stem + ".SAFE/"

        asf_datascene.extract(
            extract_location=tmp_path,
            members=[base + "preview/", base + "preview/map-overlay.kml"]
        )
        assert asf_datascene.corner_coordinates == []

        asf_datascene.locate()
        assert asf_datascene.corner_coordinates != []

    @pytest.mark.unit
    def test_asf_datascene_show_preview(self, s1a_safedir):
        asf_data_scene = ASFDataScene(safe_path=s1a_safedir, extracted=True)
        ax = asf_data_scene.plot_preview()
        assert isinstance(ax, matplotlib.axes.Axes)

    @pytest.mark.unit
    def test_asf_datascene_from_zip_creates_sub_data_files(self, s1a_zipfile):
        """Currently, we are not loading measurement and annotation files from zip. Need to be extracted"""
        asf_data_scene = ASFDataScene(s1a_zipfile)

        assert hasattr(asf_data_scene, "asf_data_files")
        assert len(asf_data_scene.asf_data_files) == 0

    @pytest.mark.unit
    def test_asf_datascene_creates_sub_data_files(self, s1a_safedir):
        asf_data_scene = ASFDataScene(s1a_safedir, extracted=True)

        assert hasattr(asf_data_scene, "asf_data_files")
        assert isinstance(asf_data_scene.asf_data_files, list)
        for data_file in asf_data_scene.asf_data_files:
            assert isinstance(data_file, ASFDataFile)
            assert isinstance(data_file.annotation_filepath, Path)
            assert isinstance(data_file.annotation_filename, str)

    # TODO: On load, want the ASFDataScene to find the subfiles and store a dict of {stem:geo_bounds}
    # @pytest.mark.unit
    # def test_asf_datascene_finds_geo_bounds(self, s1a_zipfile):
    #     asf_datascene = ASFDataScene(s1a_zipfile)
    #     assert hasattr(asf_datascene, "stem_geo_bounds")

    # TODO: Add methods for "mosaicing" multiple georeferenced coords into a plottable structure?
    #  Pre-prepared plotting methods (Magnitude top, phase bottom; columns for VH versus VV)
    #  False color composite ability
