from xzgen import ImageObject
from pathlib import Path
import os
import numpy as np

def test_frompath():
    img = ImageObject.frompath(
        str(Path.cwd().joinpath(
            'tests',
            'test_data',
            'imageobject',
            'hs_smooth.png')))
    assert isinstance(img, ImageObject)

def test_findmask():
    img = ImageObject.frompath(
        str(Path.cwd().joinpath(
            'tests',
            'test_data',
            'imageobject',
            'hs_smooth.png')))
    boundRect, mask = img.find_mask()
    assert len(boundRect) == 4
    assert isinstance(mask, np.ndarray)
