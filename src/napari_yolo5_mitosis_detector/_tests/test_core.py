import napari_yolo5_mitosis_detector.core as core
def test_load_model():
    
    core._model_loaded()
    assert core._model is not None
