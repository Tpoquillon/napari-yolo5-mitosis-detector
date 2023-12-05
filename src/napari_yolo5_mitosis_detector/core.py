import numpy as np
import pandas as pd
import torch
from torch.hub import load
from pathlib import Path
import napari
model = load('ultralytics/yolov5', 'custom', path=Path(r"D:\Data\Models\Yolo5\mito-nuclei-detection") / "weights" / "best.pt",force_reload=False)

if torch.cuda.is_available(): 
    model = model.to(torch.device("cuda"))
def im_preproc(im):
    q99 = np.quantile(im,0.999)
    q01 = np.quantile(im, 0.001)
    return (((im-q01)/(q99-q01)).clip(0,1)*255).astype(np.uint8)

def zxy_to_pandas(im):
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
    df = zxy_to_pandas(im[None,...])
    rectangle_data = np.array([[[row.ymin,row.xmin],[row.ymax,row.xmax]] for _,row in df.iterrows()])
    return napari.layers.Shapes(data=rectangle_data,ndim=2,edge_color="red")