import importlib_resources
from pathlib import Path
from tifffile import imread

CACHED = {}

def get_image_sample(ndim = 4):
    global CACHED
    if "test_4d_im.tif" not in CACHED:
        p = importlib_resources.files("napari_yolo5_mitosis_detector") / "_tests" /  "test_4d_im.tif"
        CACHED["test_4d_im.tif"]  = imread(p)
    if ndim ==4:
        return CACHED["test_4d_im.tif"].copy()
    elif ndim ==3:
        return CACHED["test_4d_im.tif"][6,...].copy()
    elif ndim ==2:
        return CACHED["test_4d_im.tif"][6,4,...].copy()
    else :
        raise NotImplementedError()