

import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List
import numpy as np
import pandas as pd


def _overlap_ratio(poly1, poly2):
    min_area = min(poly1.area,poly2.area)
    intersection = poly1.intersection(poly2)
    return intersection.area / min_area

def row_to_rect(row:pd.Series, ndim:int):
    if ndim == 2:
        return [[row.ymin,row.xmin],[row.ymin,row.xmax],[row.ymax,row.xmax],[row.ymax,row.xmin]]
    elif ndim ==3:
        return [[row.z,row.ymin,row.xmin],[row.z,row.ymin,row.xmax],[row.z,row.ymax,row.xmax],[row.z,row.ymax,row.xmin]]
    elif ndim ==4:
        return [[row.t, row.z,row.ymin,row.xmin],[row.t, row.z,row.ymin,row.xmax],[row.t, row.z,row.ymax,row.xmax],[row.t, row.z,row.ymax,row.xmin]]
    else:
        raise Exception("ndim must be 2 or 3, not %d"%ndim)
    
def _find_overlapping_rectangles(rectangle_layers: List[List[np.ndarray]], threshold_overlap:float=-1., threshold_distance:float=-1.):
    # Build graph
    G = nx.Graph()
    plygon_dict={}
    polygon_layer = [[Polygon(el)for el in layer] for layer in rectangle_layers]
    for z in range(0,len(polygon_layer)-1):
        for i, rect1 in enumerate(polygon_layer[z]):
            for j, rect2 in enumerate(polygon_layer[z+1]):
                d = rect1.distance(rect2)
                o = _overlap_ratio(rect1,rect2)
                G.add_edge((z,i), (z+1,j),**{"distance": d,"overlap":o})
                plygon_dict[(z+1,j)]=rect2
                if z==0:
                    plygon_dict[(z,i)]=rect1


    # Find connected components
    
    G_filtered = nx.subgraph_view(G,filter_edge = lambda X,Y:G[X][Y]["overlap"]>threshold_overlap)
    if threshold_distance>=0 and threshold_overlap<=0:
        G_filtered = nx.subgraph_view(G_filtered,filter_edge = lambda X,Y:G[X][Y]["d"]<threshold_distance)
    components = list(nx.connected_components(G_filtered))

    return components

def _add_layer_ids(df:pd.DataFrame):
    df = df.set_index(["z"])
    df["layer_id"] = -1
    for z in range(int(df.index.max()+1)):
        df.loc[z:z,"layer_id"] = list(range(len(df.loc[z:z])))
    return df.set_index("layer_id",append=True)
    


def zyx_pandas_post_process(df:pd.DataFrame,**kwargs):
    df = _add_layer_ids(df).sort_index()
    rectangle_layers = [[ row_to_rect(row,2)  for _,row in df.loc[z:z].iterrows()] for z in range(int(df.index.levels[0].max()+1))]
    components = _find_overlapping_rectangles(rectangle_layers,**kwargs)
    for c , comp in enumerate(components):
        df.loc[list(comp),"class"] = np.round(np.mean(df.loc[list(comp),"class"])**2)
    return df.reset_index()
    
