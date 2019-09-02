import os
import shutil
from pathlib import Path
import numpy as np
from xzgen import ImageData, ImageObject, Dimension, Scene

TEST_PATH_CONFIG = Path.cwd().joinpath('tests', 'test_data', 'imageconfig')
SKU_INFO_PATH = TEST_PATH_CONFIG.joinpath('hs_sku_info.csv')
PATH_POS = TEST_PATH_CONFIG.joinpath('test_POS')
PATH_NEG = TEST_PATH_CONFIG.joinpath('NEG_SKU')
PATH_NOSKU = TEST_PATH_CONFIG.joinpath('NEG_BG')
PATH_OCC = TEST_PATH_CONFIG.joinpath('OCC_DIR')
PATH_BG = TEST_PATH_CONFIG.joinpath('Background')

dimension = Dimension(str(SKU_INFO_PATH))
image_data = ImageData(
    path_pos = str(PATH_POS),
    path_neg = str(PATH_NEG),
    path_nosku = str(PATH_NOSKU),
    path_occ = str(PATH_OCC),
    path_bg = str(PATH_BG),
    sku_list = dimension.sku_list,
    root_dir = TEST_PATH_CONFIG)

def test_scene():
    scene = Scene(dimension, image_data, 7, 0.8, upper_limit=45)
    assert True
