### Graph_model


import networkx as nx
import numpy as np
import ot
import time
from scipy.sparse.csgraph import shortest_path
from scipy import sparse


class NoAttrMatrix(Exception):
    pass




class Graph():
    """ Graph is a class that model all the graphs used in the experiments.
    
    Attributes
    ----------
    nx_graph : a networkx graph, optionnal
               The networkx graph
    C : ndarray
        The structure matrix of the graph. Initalize at None
    name_struct_dist : string
                       The name of the method used to compute the structure matrix
    name : string,
           Name of the graph because life without name has no meaning.
    """
    def __init__(self,nx_graph=None):
        if nx_graph is not None:
            self.nx_graph=nx.Graph(nx_graph)
        else:
            self.nx_graph=nx.Graph()
        self.name='A graph as no name'
        self.log={}
        self.log['pertoperdistance']=[]
        self.log['pathtime']=[]
        self.log['attridist']=[]
        self.C=None
        self.name_struct_dist='No struct name for now'


    def __eq__(self, other) :
        return self.nx_graph == other.nx_graph

    def __hash__(self):
        return hash(str(self))

    def characterized(self):
        if self.name!='A graph as no name':
            return self.name
        else:
            return self

    def nodes(self):
        """ returns the vertices of a graph """
        return dict(self.nx_graph.nodes())

    def edges(self):
        """ returns the edges of a graph """
        return self.nx_graph.edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self.nodes():
            self.nx_graph.add_node(vertex)

    def values(self):
        """ returns a list of all the features of the graph"""
        return [v for (k,v) in nx.get_node_attributes(self.nx_graph,'attr_name').items()]

    def add_nodes(self, nodes):
        self.nx_graph.add_nodes_from(nodes)

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        (vertex1, vertex2) = tuple(edge)
        self.nx_graph.add_edge(vertex1,vertex2)

    def add_one_attribute(self,node,attr,attr_name='attr_name'):
        self.nx_graph.add_node(node,attr_name=attr)

    def add_attibutes(self,attributes):
        attributes=dict(attributes)
        for node,attr in attributes.items():
            self.add_one_attribute(node,attr)

    def get_attr(self,vertex):
        return self.nx_graph.node[vertex]
    
    def reshaper(self,x):
        try:
            a=x.shape[1]
            return x
        except IndexError:
            return x.reshape(-1,1)

    def distance_matrix(self,method='shortest_path',changeInf=True,maxvaluemulti=10,force_recompute=False):
        """ Compute the structure matrix of the graph.
        It aims at comparing nodes between them using a notion of similarity defined by the "method" parameter
        
        Parameters
        ----------
        method : string, default shortest_path. choices : shortest_path, square_shortest_path, weighted_shortest_path, adjency, harmonic_distance
               The method used to compute the structure matrix of the graph :
                   - shortest_path : compute all the shortest_path between the nodes
                   - square_shortest_path : same but squared
                   - weighted_shortest_path : compute the shortest path of the weighted graph with weights the distances between the features of the nodes
                   - adjency : compute the adjency matrix of the graph
                   - harmonic_distance : harmonic distance between the nodes
        changeInf : bool
                    If true when the graph has disconnected parts it replaces inf distances by a maxvaluemulti times the largest value of the structure matrix
        force_recompute : force to recompute de distance matrix. If False the matrix is computed only if not already compute or if the method used for computing it changes
        Returns
        -------
        C : ndarray, shape (n_nodes,n_nodes)
            The structure matrix of the graph
        Set also the attribute C of the graph if C does not exist or if force_recompute is True
        """
        start=time.time()
        if (self.C is None) or force_recompute:

            A=nx.adjacency_matrix(self.nx_graph)

            if method=='harmonic_distance':

                A=A.astype(np.float32)
                D=np.sum(A,axis=0)
                L=np.diag(D)-A

                ones_vector=np.ones(L.shape[0])
                fL=np.linalg.pinv(L)

                C=np.outer(np.diag(fL),ones_vector)+np.outer(ones_vector,np.diag(fL))-2*fL
                C=np.array(C)
                
            if method=='shortest_path':
                C=shortest_path(A)
             
            if method=='square_shortest_path':
                C=shortest_path(A)
                C=C**2
                
            if method=='adjency':
                return A.toarray()
                
            if method=='weighted_shortest_path':
                d=self.reshaper(np.array([v for (k,v) in nx.get_node_attributes(self.nx_graph,'attr_name').items()]))
                D= ot.dist(d,d)
                D_sparse=sparse.csr_matrix(D)
                C=shortest_path(A.multiply(D_sparse))
                
            if changeInf==True:
                C[C==float('inf')]=maxvaluemulti*np.max(C[C!=float('inf')]) # à voir
                
            self.C=C
            self.name_struct_dist=method
            end=time.time()
            self.log['allStructTime']=(end-start)
            return self.C

        else :
            end=time.time()
            self.log['allStructTime']=(end-start)
            return self.C


    def all_matrix_attr(self,return_invd=False):
        d=dict((k, v) for k, v in self.nx_graph.nodes.items())
        x=[]
        invd={}
        try :
            j=0
            for k,v in d.items():
                x.append(v['attr_name'])
                invd[k]=j
                j=j+1
            if return_invd:
                return np.array(x),invd
            else:
                return np.array(x)
        except KeyError:
            raise NoAttrMatrix
            
#%%

