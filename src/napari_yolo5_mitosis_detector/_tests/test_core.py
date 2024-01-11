import napari_yolo5_mitosis_detector.core as core
from napari_yolo5_mitosis_detector._tests.utility import get_image_sample
import pytest




def test_load_model():    
    core._model_loaded()
    assert core._model is not None

@pytest.mark.parametrize("ndim,monolayer,return_surface",[(2,False,False),(3,False,False),(3,False,True),(4,False,False),(4,True,False)])
def test_yolo5_bbox_ndim(ndim,monolayer,return_surface):
    im_data = core.napari.layers.Image(get_image_sample(ndim))
    outputs = core.yolo5_bbox_mitosis(im_data,monolayer,return_surface)
    for i,layer in enumerate(outputs):
        if i<2:
            assert type(layer) is core.napari.layers.Shapes
            assert layer.name == "%s_%s-bbox"%(im_data.name,"mitosis" if i%2==0 else "nuclei")
        elif return_surface and i<4:
            assert type(layer) is core.napari.layers.Surface
            assert layer.name == "%s_%s-3dbox"%(im_data.name,"mitosis" if i%2==0 else "nuclei")
        else:
            assert type(layer) is core.napari.layers.Tracks
            assert layer.name == "%s_tracks"%im_data.name
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