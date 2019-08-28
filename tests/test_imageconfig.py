from xzgen import ImageData, Dimension
from pathlib import Path

def test_pass():
    assert True

def test_dimension():
    sku_info_path = str(Path.cwd().joinpath(
        'tests',
        'test_data',
        'imageconfig',
        'hs_sku_info.csv'))

    dimension = Dimension(sku_info_path)
    assert True


