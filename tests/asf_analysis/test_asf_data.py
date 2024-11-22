import pytest

from asf_analysis.asf_data import ASFData


@pytest.fixture
def s1a_filename():
    yield "/s1a-iw1-slc-vh-20180606t141443-20180606t141508-022237-0267eb-001.tiff"

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
    def test_parse_info_from_stem(self, s1a_filename):
        stem = s1a_filename.split(".")[0]
        asf_data = ASFData()
        asf_data.parse_info_from_stem(stem)

        assert asf_data.absolute_orbit_number == "022237"

    @pytest.mark.unit
    def test_asf_data_has_annotation_xml(self):
        asf_data = ASFData()
        asf_data.load("./data/sentinel-1/")


# my_data = ASFData()
#
# my_data.load(filepath)