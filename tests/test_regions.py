from PIL import Image

from spatial_kis.regions import grid_crops, region_ids


def test_grid_has_nine_overlapping_crops():
    crops = grid_crops(Image.new("RGB", (300, 300)), 0.12)
    assert len(crops) == 9
    assert crops[4].size[0] > 100


def test_named_left_region_maps_to_column():
    assert region_ids("left") == (1, 4, 7)
