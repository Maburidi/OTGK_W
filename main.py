
import ot
import ot.plot

import numpy as np
import os, sys

from data_loader import load_local_data
from sklearn.model_selection import train_test_split
import networkx as nx

import time
from sklearn.svm import SVC


def hamming_dist(x,y):
    #print('x',len(x[-1]))
    #print('y',len(y[-1]))
    return len([i for i, j in zip(x, y) if i != j])


def assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    a=X.dtype.char in np.typecodes['AllFloat']
    b=np.isfinite(X.sum())
    c=np.isfinite(X).all()

    if (a and not b and not c):
        return False
    else :
        return True


class InfiniteException(Exception):
    pass


def compute_matching_similarity(g1, g2, dist = "uniform",features_metric ='hamming' , discrete = True, alg = 'wasserstein', sinkhorn_lambda=1e-2 ):
    nodes1 = g1.nodes()
    nodes2 = g2.nodes()
    startstruct=time.time()

    if dist == "uniform":
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)
    elif dist == "node_cent":
        A1=nx.adjacency_matrix(g1.nx_graph)
        D1=np.sum(A1,axis=0)
        A2=nx.adjacency_matrix(g2.nx_graph)
        D2=np.sum(A2,axis=0)

        t1masses = D1/sum(D1)
        t2masses = D2/sum(D2)
    elif dist == "btws":
        centrality1 = nx.betweenness_centrality(g1.nx_graph, normalized=True, k=len(g1.nodes())-1)
        centrality2 = nx.betweenness_centrality(g2.nx_graph, normalized=True, k=len(g2.nodes())-1)

        t1masses = [x/sum(list(centrality1.values())) for x in list(centrality1.values())]
        t2masses = [x/sum(list(centrality2.values())) for x in list(centrality2.values())]

    #ground_distance = 'hamming' if discrete else 'euclidean'

    if features_metric=='euclidean':
        costs = ot.dist(g1.all_matrix_attr(), g2.all_matrix_attr(), metric=features_metric)

    elif features_metric=='hamming':
        f = lambda x,y: hamming_dist(x,y)
        costs = ot.dist(g1.all_matrix_attr(), g2.all_matrix_attr(), metric=f)

    if alg == "sinkhorn":
        # mat = ot.sinkhorn( t1masses  , t2masses, costs, sinkhorn_lambda,  numItermax=50)
        # dist_ = np.sum(np.multiply(mat, costs))
        dist_ = ot.sinkhorn2( t1masses  , t2masses, costs, sinkhorn_lambda,  numItermax=50)
    else:
        dist_ = ot.emd2(t1masses,t2masses , costs)

    return dist_


def compute_distances_btw_graphs(X,Y,fm, otsolver):

    X = X.reshape(X.shape[0],)
    Y = Y.reshape(Y.shape[0],)
    if np.all(X==Y):
        M = np.zeros((X.shape[0], Y.shape[0]))      # similarity matrix
        Q = np.zeros((X.shape[0], Y.shape[0]))
        for i, x1 in enumerate(X):
            for j,x2 in enumerate(Y):
                if j>=i:
                    dist_ = compute_matching_similarity(x1, x2, dist="uniform", features_metric = fm, alg = otsolver)
                    M[i, j] = dist_
            if i % 20 == 0:
                print(f'Processed {i} graphs out of {len(X)}')

        np.fill_diagonal(Q,np.diagonal(M))     # this is to avoid
        M = M+M.T-Q

    else:
        M = np.zeros((X.shape[0], Y.shape[0]))
        for i, x1 in enumerate(X):
            row=[compute_matching_similarity(x1, x2, dist="uniform") for j,x2 in enumerate(Y)]
            M[i,:]=row

            if i % 20 == 0:
                print(f'Processed {i} graphs out of {len(X)}')


    M[np.abs(M)<=1e-15]=0 #threshold due to numerical precision
    return M





def main():
    name='enzymes'    # mutag, ptc, enzymes, protein, bzr, cox2
    data_path='/content/OTGK_W/data'
    distance_ = 'euclidean'     # for discrete attributed grahs (mutag, ptc) use 'hamming', for continous (enzymes, protein, bzr, cox2) use 'euclidean'
    otsolver_ = 'wasserstein'       # sinkhorn or wasserstein
    plot = False
    wl_ = 0                    # wl=2 for discrete attributed datasets (mutag,ptc), otherwise wl = 0
    gamma=1                    # kernel parameter

    #---------------------------
    
    X,y=load_local_data(data_path,name, wl=wl_)

    X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.33, random_state=42)
    print(f"{name} dataset contains {len(X)} graphs")
    print(f"There are {len(X_train)} graphs for training")
    print(f"There are {len(X_test)} graphs for testing")


    ######################### TRAINING PART ####################
    M = compute_distances_btw_graphs(X_train,X_train,fm = distance_, otsolver= otsolver_)
    
    if plot == True:
        cmap='Reds'
        pl.imshow(M,cmap=cmap,interpolation='nearest')
        plt.imsave('/content/drive/MyDrive/simil_' + str(name) +'.jpg' ,M, dpi=100, cmap=cmap)


    Z=np.exp(-gamma*(M))
    if not assert_all_finite(Z):
        raise InfiniteException('There is Nan')

    #------------------ Classification using SVM -----------------
    
    C=1
    verbose = False

    svc=SVC(C=C,kernel="precomputed",verbose=verbose,max_iter=10000000)
    classes_ =np.array(y_train)

    svc.fit(Z, classes_)



 
    M_test = compute_distances_btw_graphs(X_test,X_train, fm =distance_, otsolver=otsolver_ )

    #cmap='Reds'
    #pl.imshow(M_test,cmap=cmap,interpolation='nearest')
    
    Z_test=np.exp(-gamma*(M_test))
    if not assert_all_finite(Z_test):
        raise InfiniteException('There is Nan')

    preds=svc.predict(Z_test)
    print(f"Accuracy =  {np.sum(preds==y_test)/len(y_test)}")

    

if __name__ == "__main__":
    main()


