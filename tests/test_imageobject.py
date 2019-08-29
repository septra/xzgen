from xzgen import ImageObject
from pathlib import Path
import os
import numpy as np

TEST_PATH = Path.cwd().joinpath('tests', 'test_data', 'imageobject')

img = ImageObject.frompath(str(TEST_PATH.joinpath('hs_smooth.png')))
occ = ImageObject.frompath(str(TEST_PATH.joinpath('occ0.jpg')))

def test_frompath():
    assert isinstance(img, ImageObject)

def test_findmask():
    boundRect, mask = img.find_mask()
    assert len(boundRect) == 4
    assert isinstance(mask, np.ndarray)

def test_add_occlusion():
    assert img.shape == img.add_occlusion(occ).shape

def test_augment_object():
    assert img.shape == img.augment_obj().shape

