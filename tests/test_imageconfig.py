import shutil
from xzgen import ImageData, Dimension
from pathlib import Path

TEST_PATH = Path.cwd().joinpath('tests', 'test_data', 'imageconfig')
SKU_INFO_PATH = TEST_PATH.joinpath('hs_sku_info.csv')
PATH_POS = TEST_PATH.joinpath('test_pos')
PATH_NEG = TEST_PATH.joinpath('NEG_SKU')
PATH_NOSKU = TEST_PATH.joinpath('NEG_BG')
PATH_OCC = TEST_PATH.joinpath('OCC_DIR')
PATH_BG = TEST_PATH.joinpath('Background')

def test_pass():
    assert True

def test_Dimension_init():
    dimension = Dimension(str(SKU_INFO_PATH))
    assert 'hs_smooth' in dimension.valid_sku
    assert 'hs_anti' in dimension.valid_sku
    assert 'hs_black' in dimension.valid_sku
    assert isinstance(dimension.sku_list, dict)
    assert 'hs_smooth' in dimension.sku_list
    assert 'hs_anti' in dimension.sku_list

def test_ImageData():
    dimension = Dimension(SKU_INFO_PATH)
    image_data = ImageData(
        path_pos = str(PATH_POS),
        path_neg = str(PATH_NEG),
        path_nosku = str(PATH_NOSKU),
        path_occ = str(PATH_OCC),
        path_bg = str(PATH_BG),
        sku_list = dimension.sku_list,
        root_dir = TEST_PATH)
    assert TEST_PATH.joinpath('dataset').is_dir()
    assert (TEST_PATH
        .joinpath(image_data.folder_name)
        .joinpath('classes.csv')
        .exists())
    assert (TEST_PATH
        .joinpath(image_data.folder_name)
        .joinpath('train_annotations.csv')
        .exists())
    assert len(image_data.neg_files_list) > 0
    assert len(image_data.pos_files_list) > 0
    assert len(image_data.sku_neg_list) > 0

    shutil.rmtree(TEST_PATH.joinpath('dataset'))
