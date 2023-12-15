import napari_yolo5_mitosis_detector.core as core
from napari_yolo5_mitosis_detector._tests.utility import get_image_sample


def test_load_model():    
    core._model_loaded()
    assert core._model is not None

def test_yolo5_bbox_2d_mitosis():
    # because our "widget" is a pure function, we can call it and
    # test it independently of napari
    im_data = core.napari.layers.Image(get_image_sample(ndim = 2))
    empty_bbox_layer = core.yolo5_bbox_mitosis(im_data)[0]
    assert empty_bbox_layer.ndim == 2

def test_yolo5_bbox_3d_mitosis():
    # because our "widget" is a pure function, we can call it and
    # test it independently of napari
    im_data =  core.napari.layers.Image(get_image_sample(ndim = 3))
    empty_bbox_layer = core.yolo5_bbox_mitosis(im_data)[0]
    assert empty_bbox_layer.ndim == 3

def test_yolo5_bbox_4d_mitosis():
    im_data =  core.napari.layers.Image(get_image_sample(ndim = 4))
    empty_bbox_layer = core.yolo5_bbox_mitosis(im_data)[0]
    assert empty_bbox_layer.ndim == 4

def test_maxproj_3d():
    # because our "widget" is a pure function, we can call it and
    # test it independently of napari
    im_data =  core.napari.layers.Image(get_image_sample(ndim = 3))
    proj_layer = core.max_intensity_projection(im_data)
    assert proj_layer.ndim == 3
    assert proj_layer.data.shape[0]==1

def test_maxproj_4d_mitosis():
    im_data =  core.napari.layers.Image(get_image_sample(ndim = 4))
    proj_layer = core.max_intensity_projection(im_data)
    assert proj_layer.ndim == 4
    assert proj_layer.data.shape[1]==1