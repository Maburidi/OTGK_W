import networkx as nx
import numpy as np
import ot
import time

from Graph_model import Graph
from utils import *



def load_local_data(data_path,name,one_hot=False,attributes=True,use_node_deg=False,wl=0):
    if name=='mutag':
        path=data_path+'/MUTAG_2/'
        dataset=build_MUTAG_dataset(path,one_hot=one_hot)
    if name=='enzymes':
        path=data_path+'/ENZYMES_2/'
        if attributes:
            dataset=build_ENZYMES_dataset(path,type_attr='real')
        else:
            dataset=build_ENZYMES_dataset(path)
    
    if name=='ptc':
        path=data_path+'/PTC_MR/'
        dataset=build_PTC_dataset(path,one_hot=one_hot)
    if name=='protein':
        path=data_path+'/PROTEINS_full/'
        if attributes:
            dataset=build_PROTEIN_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_PROTEIN_dataset(path)
    if name=='bzr':
        path=data_path+'/BZR/'
        if attributes:
            dataset=build_BZR_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_BZR_dataset(path)
    if name=='cox2':
        path=data_path+'/COX2/'
        if attributes:
            dataset=build_COX2_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_COX2_dataset(path)

    X,y=zip(*dataset)
    
    if wl!=0:
        X=label_wl_dataset(X,h=wl)
    return np.array(X),np.array(y)




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




def build_ENZYMES_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'ENZYMES_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'ENZYMES_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'ENZYMES_node_attributes.txt')
    adjency=compute_adjency(path,'ENZYMES_A.txt')
    data_dict=graph_indicator(path,'ENZYMES_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_PTC_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'PTC_MR_graph_labels.txt')
    adjency=compute_adjency(path,'PTC_MR_A.txt')
    data_dict=graph_indicator(path,'PTC_MR_graph_indicator.txt')
    node_dic=node_labels_dic(path,'PTC_MR_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],18)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data


def build_PROTEIN_dataset(path,type_attr='label',use_node_deg=False):
    if type_attr=='label':
        node_dic=node_labels_dic(path,'PROTEINS_full_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'PROTEINS_full_node_attributes.txt')
    graphs=graph_label_list(path,'PROTEINS_full_graph_labels.txt')
    adjency=compute_adjency(path,'PROTEINS_full_A.txt')
    data_dict=graph_indicator(path,'PROTEINS_full_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data
