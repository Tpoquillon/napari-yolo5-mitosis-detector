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

def zxy_to_pandas(im):
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
def xy_to_rectangle(im:np.ndarray):
    assert len(im.shape)==2 , 'img should have 2 dimention'
    lay =  napari.layers.Shapes(ndim=2)
    df = zxy_to_pandas(im[None,...])

    rectangle_data_mito = np.array([[[row.ymin,row.xmin],[row.ymax,row.xmax]] for _,row in df.iterrows() if row["class"]==0])
    if len(rectangle_data_mito>0 ):
        lay.add_rectangles((rectangle_data_mito),edge_width=4, edge_color="green", face_color="#ffffff32", z_index=2)
    
    rectangle_data_nuc = np.array([[[row.ymin,row.xmin],[row.ymax,row.xmax]] for _,row in df.iterrows() if row["class"]==1])
    if len(rectangle_data_nuc>0 ):
        lay.add_rectangles((rectangle_data_nuc),edge_width=2, edge_color="red", face_color="#ffffff32", z_index=1)
    return lay