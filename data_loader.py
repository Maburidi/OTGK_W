import networkx as nx
import numpy as np
import ot
import time

from Graph_model import Graph
from utlis import *



def load_local_data(data_path,name,one_hot=False,attributes=True,use_node_deg=False,wl=0):
    if name=='mutag':
        path=data_path+'/MUTAG_2/'
        dataset=build_MUTAG_dataset(path,one_hot=one_hot)






def build_MUTAG_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'MUTAG_graph_labels.txt')
    adjency=compute_adjency(path,'MUTAG_A.txt')
    data_dict=graph_indicator(path,'MUTAG_graph_indicator.txt')
    node_dic=node_labels_dic(path,'MUTAG_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],7)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data



