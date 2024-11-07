import pytest

from src.asf_analysis.asf_data import ASFData


class TestASFData:
    @pytest.mark.unit
    def test_create_asf_data(self, tmpdir):
        asf_data = ASFData()
        assert asf_data.filepath is None

    @pytest.mark.unit
    def test_load_asf_file(self, tmpdir):
        path_to_file = tmpdir + "/s1a-iw1-slc-vh-20180606t141443-20180606t141508-022237-0267eb-001.tiff"
        asf_data = ASFData()

        asf_data.load(path_to_file)
        assert asf_data.filepath == path_to_file

    @pytest.mark.unit
    def test_parse_info_from_stem(self):
        stem = "s1a-iw1-slc-vh-20180606t141443-20180606t141508-022237-0267eb-001"
        asf_data = ASFData()
        asf_data.parse_info_from_stem(stem)

        assert asf_data.absolute_orbit_number == "022237"

# my_data = ASFData()
#
# my_data.load(filepath)