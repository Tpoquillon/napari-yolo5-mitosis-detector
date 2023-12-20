import napari_yolo5_mitosis_detector.core as core
from napari_yolo5_mitosis_detector._tests.utility import get_image_sample
import pytest




def test_load_model():    
    core._model_loaded()
    assert core._model is not None

@pytest.mark.parametrize("ndim,monolayer",[(2,False),(3,False),(4,False),(4,True)])
def test_yolo5_bbox_ndim(ndim,monolayer):
    im_data = core.napari.layers.Image(get_image_sample(ndim))
    outputs = core.yolo5_bbox_mitosis(im_data,monolayer)
    for i,layer in enumerate(outputs):
        assert type(layer) is (core.napari.layers.Tracks  if i==2 else core.napari.layers.Shapes)
        assert layer.name == ("%s_mitosis-bbox"%im_data.name if i==0 else "%s_nuclei-bbox"%im_data.name if i==1 else "%s_tracks"%im_data.name)
        assert (layer.scale==im_data.scale)[-2:].all()
        assert (layer.translate==im_data.translate)[-2:].all()
@pytest.mark.parametrize("ndim",[3,4])
def test_maxproj_ndim(ndim):
    im_data =  core.napari.layers.Image(get_image_sample(ndim))
    proj_layer = core.max_intensity_projection(im_data)
    assert proj_layer.ndim == ndim
    assert proj_layer.data.shape[ndim-3]==1

if __name__ == "__main__":
    retcode = pytest.main([__file__])