

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

def _build_overlapping_graph(rectangle_layers: List[List[np.ndarray]]):
    G = nx.DiGraph()
    polygon_layer = [[Polygon(el)for el in layer] for layer in rectangle_layers]
    for l in range(0,len(polygon_layer)-1):
        for i, rect1 in enumerate(polygon_layer[l]):
            for j, rect2 in enumerate(polygon_layer[l+1]):
                d = rect1.distance(rect2)
                o = _overlap_ratio(rect1,rect2)
                G.add_edge((l,i), (l+1,j),**{"distance": d,"overlap":o})
    return G    

def _get_large_duplicat(rectangle_layers: List[List[np.ndarray]]):
    polygon_layer = [[Polygon(el)for el in layer] for layer in rectangle_layers]
    overlapping =[]
    for l in range(0,len(polygon_layer)):
        lay = polygon_layer[l]
        for i in range(len(lay)):
            rect1 = lay[i]
            for j in range(i+1,len(lay)):
                rect2 = lay[j]
                if _overlap_ratio(rect1,rect2) >0.8:
                    overlapping.append((l,i) if rect1.area>rect2.area else (l,j))
    return overlapping


def _filter_overlaping_graph(G:nx.Graph,threshold_overlap:float=-1., threshold_distance:float=-1.):
    G_filtered = nx.subgraph_view(G,filter_edge = lambda X,Y:G[X][Y]["overlap"]>threshold_overlap)
    if threshold_distance>=0 and threshold_overlap<=0:
        G_filtered = nx.subgraph_view(G_filtered,filter_edge = lambda X,Y:G[X][Y]["d"]<threshold_distance)
    return G_filtered

def _find_overlapping_rectangles(rectangle_layers: List[List[np.ndarray]], threshold_overlap:float=-1., threshold_distance:float=-1.):
    # Build graph
    G = _build_overlapping_graph(rectangle_layers)
    # Find connected components    
    G_filtered = _filter_overlaping_graph(G, threshold_overlap,threshold_distance)
    components = list(nx.connected_components(G_filtered.to_undirected()))

    return components


def _add_layer_ids(df:pd.DataFrame, index_col = "z"):
    df = df.set_index([index_col])
    df["layer_id"] = -1
    for l in range(int(df.index.max()+1)):
        df.loc[l:l,"layer_id"] = list(range(len(df.loc[l:l])))
    return df.set_index("layer_id",append=True)
    

def tyx_pandas_post_process(df:pd.DataFrame,**kwargs):
    df = _add_layer_ids(df,"t").sort_index()
    rectangle_layers = [[ row_to_rect(row,2)  for _,row in df.loc[t:t].iterrows()] for t in range(int(df.index.levels[0].max()+1))]
    

    G = _build_overlapping_graph(rectangle_layers)
    G_filtered = _filter_overlaping_graph(G, **kwargs)
    components = list(nx.connected_components(G_filtered.to_undirected()))
    for c , comp in enumerate(components):
        sub = df.loc[list(comp),"class"]
        list_mito_comp = list(sub[sub==0].index)
        subgraph = G_filtered.subgraph(comp)
        nodlist = set(sum([list(nx.ego_graph(subgraph.reverse(),el,4).nodes) for el in list_mito_comp ],[]))
        df.loc[list(nodlist),"class"] = 0
    df = df.drop( _get_large_duplicat(rectangle_layers))
    return df.reset_index()


def zyx_pandas_post_process(df:pd.DataFrame,**kwargs):
    df = _add_layer_ids(df).sort_index()
    rectangle_layers = [[ row_to_rect(row,2)  for _,row in df.loc[z:z].iterrows()] for z in range(int(df.index.levels[0].max()+1))]
    components = _find_overlapping_rectangles(rectangle_layers,**kwargs)
    for c , comp in enumerate(components):
        print(comp)
        df.loc[list(comp),"class"] = np.round(np.mean(df.loc[list(comp),"class"])**2)
    return df.reset_index()
    
