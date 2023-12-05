import numpy as np
import pandas as pd
from pathlib import Path
import napari
import threading
model = None

def load_model():
    import torch
    from torch.hub import load
    global model
    model = load('ultralytics/yolov5', 'custom', path=Path(r"D:\Data\Models\Yolo5\mito-nuclei-detection") / "weights" / "best.pt",force_reload=False)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
loader = threading.Thread(target=load_model)
loader.start()

def model_loaded():
    if model is None:
        loader.join()


def im_preproc(im):
    q99 = np.quantile(im,0.999)
    q01 = np.quantile(im, 0.001)
    return (((im-q01)/(q99-q01)).clip(0,1)*255).astype(np.uint8)

def zyx_to_pandas(im):
    model_loaded()

    batch_size=8
    im = im_preproc(im)
    batches = [im[i:i+batch_size] for i in range(0,len(im),batch_size)]
    results_batch = sum([[model([el for el in imb] )] for imb in batches],[])
    results_pandas = sum([el.pandas().xyxy for el in results_batch],[])

    for i, el in enumerate(results_pandas):
        if not el.empty:
            el["z"] = i
    df = pd.concat(results_pandas,ignore_index=True)
    return df


def row_to_rect(row:pd.Series, ndim:int):
    if ndim == 2:
        return [[row.ymin,row.xmin],[row.ymin,row.xmax],[row.ymax,row.xmax],[row.ymax,row.xmin]]
    elif ndim ==3:
        return [[row.z,row.ymin,row.xmin],[row.z,row.ymin,row.xmax],[row.z,row.ymax,row.xmax],[row.z,row.ymax,row.xmin]]
    else:
        raise Exception("ndim must be 2 or 3, not %d"%ndim)

def pandas_to_layer(df:pd.DataFrame,ndim=2):
    lay =  napari.layers.Shapes(ndim=ndim)
    rectangle_data_mito = np.array([row_to_rect(row,ndim) for _,row in df.iterrows() if row["class"]==0])
    rectangle_data_nuc = np.array([row_to_rect(row,ndim)for _,row in df.iterrows() if row["class"]==1])
    if len(rectangle_data_mito>0 ):
        lay.add_rectangles((rectangle_data_mito),edge_width=4, edge_color="green", face_color="#ffffff32", z_index=2)    
    if len(rectangle_data_nuc>0 ):
        lay.add_rectangles((rectangle_data_nuc),edge_width=2, edge_color="red", face_color="#ffffff32", z_index=1)
        return lay
    
    
def yx_to_rectangle(im:np.ndarray):
    assert len(im.shape)==2 , 'img should have 2 dimention'
    df = zyx_to_pandas(im[None,...])
    lay =  pandas_to_layer(df,2)
    return lay

def zyx_to_rectangle(im:np.ndarray):
    assert len(im.shape)==3 , 'img should have 3 dimention'
    df = zyx_to_pandas(im)
    lay =  pandas_to_layer(df,3)
    return lay

def yolo5_bbox_mitosis(img:np.ndarray):
    if len(img.shape)==2:
        return yx_to_rectangle(img)
    elif len(img.shape)==3:
        return zyx_to_rectangle(img)
    else:
        return None