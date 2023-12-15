import numpy as np
import pandas as pd
from pathlib import Path
import napari
import threading
import importlib_resources
from .post_process import row_to_rect,zyx_pandas_post_process, tyx_pandas_post_process
import dask
from skimage.transform import resize
from scipy.spatial.distance import cdist
_model = None
def _load_model():
    import torch
    global _model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    import sys
    sys.path.append(str(importlib_resources.files("napari_yolo5_mitosis_detector") / "_models" / "Yolo5"))
    from napari_yolo5_mitosis_detector._models.Yolo5.models.common import AutoShape
    from napari_yolo5_mitosis_detector._models.Yolo5.models.experimental import attempt_load
    model = attempt_load(
    importlib_resources.files("napari_yolo5_mitosis_detector") / "_models" / "Yolo5" / "mito-nuclei-detection"/ "weights" / "best.pt",
    device=device, fuse=True)
    _model = AutoShape(model)
    sys.path.remove(str(importlib_resources.files("napari_yolo5_mitosis_detector") / "_models" / "Yolo5"))
loader = threading.Thread(target=_load_model)
loader.start()

def _model_loaded():
    if _model is None:
        loader.join()
    if _model is not None:
        return
    _load_model()

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
    result_filtered=[]
    for i, el in enumerate(results_pandas):
        if not el.empty:
            el["z"] = i
            result_filtered.append(el)
    df = pd.concat(result_filtered,ignore_index=True)
    return df

def _pandas_to_layer(df:pd.DataFrame,ndim=2):
    lay1, lay2 =  napari.layers.Shapes(ndim=ndim), napari.layers.Shapes(ndim=ndim)
    rectangle_data_mito = np.array([row_to_rect(row,ndim) for _,row in df.iterrows() if row["class"]==0])
    rectangle_data_nuc = np.array([row_to_rect(row,ndim)for _,row in df.iterrows() if row["class"]==1])
    if len(rectangle_data_mito>0 ):
        lay1.add_rectangles((rectangle_data_mito),edge_width=4, edge_color="green", face_color="#ffffff32", z_index=2)    
    if len(rectangle_data_nuc>0 ):
        lay2.add_rectangles((rectangle_data_nuc),edge_width=2, edge_color="red", face_color="#ffffff32", z_index=1)
    return lay1,lay2

def _add_centroids(df:pd.DataFrame):
    df.loc[:,"x"] = (df["xmin"]+(df["xmax"]-df["xmin"])/2).copy().astype(int)
    df.loc[:,"y"] = (df["ymin"]+(df["ymax"]-df["ymin"])/2).copy().astype(int)
    return df

def _nearest_neigbour(df:pd.DataFrame):
    dist = cdist(df[["x","y"]].values,df[["x","y"]].values)
    dist[dist==0]=99999.
    df["nn"]  = np.min(dist,axis = 0)
    return df["nn"]
      
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

def _tzyx_to_rectangle(im:np.ndarray):
    assert len(im.shape)==4 , 'img should have 4 dimention'
    df = []
    for t, imt in enumerate(im):
        dft = _zyx_to_pandas(imt)
        dft = zyx_pandas_post_process(dft,threshold_overlap=0.5)
        dft["t"]=t
        df.append(dft)
    df = pd.concat(df,axis=0)
    lay =  _pandas_to_layer(df,4)
    return lay

def _tzyx_monolayer_resized_to_rectangle(im:np.ndarray):
    p_sizes = min(512,im.shape[-2]), min(512,im.shape[-1])
    yx_scale_factors = im.shape[-2]/p_sizes[-2], im.shape[-1]/p_sizes[-1]
    im_flat = resize(im,(im.shape[0],1,p_sizes[-2],p_sizes[-1]),order=1)
    df = _zyx_to_pandas(im_flat[:,0,...])
    df["t"] = df["z"]
    df["z"] = 0
    df["xmax"],df["xmin"] = (df["xmax"]*yx_scale_factors[1]).astype(int),(df["xmin"]*yx_scale_factors[1]).astype(int)
    df["ymax"],df["ymin"] = (df["ymax"]*yx_scale_factors[0]).astype(int),(df["ymin"]*yx_scale_factors[0]).astype(int)
    df = tyx_pandas_post_process(df,threshold_overlap=0.3)
    return _pandas_to_layer(df,4)


def yolo5_bbox_mitosis(img_layer:napari.layers.Image, monolayer=False):
    img = img_layer.data
    scale = np.asarray(img_layer.scale).copy()
    translate = np.asarray(img_layer.translate).copy()
    
    if type(img) == dask.array.core.Array:
        img = np.asarray(img)
    shape = img.shape
    if len(img.shape)==2:
        detection_shape_layers = _yx_to_rectangle(img)
    elif len(shape)==3:
        detection_shape_layers = _zyx_to_rectangle(img)
    elif len(shape)==4 and not monolayer:
        detection_shape_layers = _tzyx_to_rectangle(img)
    elif len(shape)==4 and monolayer:
        detection_shape_layers = _tzyx_monolayer_resized_to_rectangle(img)
        scale[1] *= shape[1]    
        translate[1] += scale[1]/2
    else:
        raise NotImplementedError("%dd image not suported yet"%len(img.shape))
    for i, lay in enumerate(detection_shape_layers):
        lay.scale = scale
        lay.translate = translate
        lay.name = img_layer.name+("_mitosis-bbox" if i==0 else "_nuc-bbox")
    return detection_shape_layers

def max_intensity_projection(img_layer:napari.layers.Image):
    img = img_layer.data
    scale = np.asarray(img_layer.scale).copy()
    shape = img.shape
    translate = np.asarray(img_layer.translate).copy()
    if type(img) == dask.array.core.Array:
        img = np.asarray(img)
    if len(shape)==3:#zyx
        proj_np = img.max(axis=0)[None,...]
    elif len(shape)==4:#tzxy
        proj_np = proj_np = img.max(axis=1)[:,None,...]
    else:
        raise NotImplementedError("%dd image not suported yet"%len(img.shape))
    scale[-3] *= shape[-3]    
    translate[-3] += scale[-3]/2
    projection = napari.layers.Image(proj_np)
    projection.scale = scale
    projection.translate = translate
    projection.name = img_layer.name+"_zprojection"

    return projection





 