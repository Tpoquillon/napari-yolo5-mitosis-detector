import numpy as np
import pandas as pd
from pathlib import Path
import napari
import threading
import importlib_resources
from .post_process import row_to_rect,zyx_pandas_post_process
_model = None

def _load_model():
    import torch
    from torch.hub import load
    global _model
    _model = load('ultralytics/yolov5', 'custom', path=importlib_resources.files("napari_yolo5_mitosis_detector") / ".models" / "Yolo5" / "mito-nuclei-detection"/ "weights" / "best.pt",force_reload=False)
    if torch.cuda.is_available():
        _model = _model.to(torch.device("cuda"))
loader = threading.Thread(target=_load_model)
loader.start()

def _model_loaded():
    if _model is None:
        loader.join()


def _im_qnorm(im):
    q99 = np.quantile(im,0.999)
    q01 = np.quantile(im, 0.001)
    return (((im-q01)/(q99-q01)).clip(0,1)*255).astype(np.uint8)

def _zyx_to_pandas(im):
    _model_loaded()

    batch_size=8
    im = _im_qnorm(im)
    batches = [im[i:i+batch_size] for i in range(0,len(im),batch_size)]
    results_batch = sum([[_model([el for el in imb] )] for imb in batches],[])
    results_pandas = sum([el.pandas().xyxy for el in results_batch],[])

    for i, el in enumerate(results_pandas):
        if not el.empty:
            el["z"] = i
    df = pd.concat(results_pandas,ignore_index=True)
    return df




def _pandas_to_layer(df:pd.DataFrame,ndim=2):
    lay =  napari.layers.Shapes(ndim=ndim)
    rectangle_data_mito = np.array([row_to_rect(row,ndim) for _,row in df.iterrows() if row["class"]==0])
    rectangle_data_nuc = np.array([row_to_rect(row,ndim)for _,row in df.iterrows() if row["class"]==1])
    if len(rectangle_data_mito>0 ):
        lay.add_rectangles((rectangle_data_mito),edge_width=4, edge_color="green", face_color="#ffffff32", z_index=2)    
    if len(rectangle_data_nuc>0 ):
        lay.add_rectangles((rectangle_data_nuc),edge_width=2, edge_color="red", face_color="#ffffff32", z_index=1)
        return lay
    
    
def _yx_to_rectangle(im:np.ndarray):
    assert len(im.shape)==2 , 'img should have 2 dimention'
    df = _zyx_to_pandas(im[None,...])
    lay =  _pandas_to_layer(df,2)
    return lay

def _zyx_to_rectangle(im:np.ndarray):
    assert len(im.shape)==3 , 'img should have 3 dimention'
    df = _zyx_to_pandas(im)
    df = zyx_pandas_post_process(df,threshold_overlap=0.5)
    lay =  _pandas_to_layer(df,3)
    return lay

def yolo5_bbox_mitosis(img:np.ndarray):
    if len(img.shape)==2:
        return _yx_to_rectangle(img)
    elif len(img.shape)==3:
        return _zyx_to_rectangle(img)
    else:
        return None