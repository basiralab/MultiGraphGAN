import numpy as np
import networkx as nx

# put it back into a 2D symmetric array
def to_2d(vector):
    size = 35
    x = np.zeros((size,size))
    c = 0
    for i in range(1,size):
        for j in range(0,i):
            x[i][j] = vector[c]
            x[j][i] = vector[c]
            c = c + 1
    return x

def topological_measures(data):
    # ROI is the number of brain regions (i.e.,35 in our case)
    ROI= to_2d(data[0]).shape[0]
    CC = np.empty((0,ROI), int)
    BC = np.empty((0,ROI), int)
    EC = np.empty((0,ROI), int)
    topology = []
    for i in range(data.shape[0]):
        A = to_2d(data[i])
        np.fill_diagonal(A, 0)

        # create a graph from similarity matrix
        G = nx.from_numpy_matrix(A)
        U = G.to_undirected()

        # Centrality #
        # compute closeness centrality and transform the output to vector
        cc = nx.closeness_centrality(U)
        closeness_centrality = np.array([cc[g] for g in U])
        # compute betweeness centrality and transform the output to vector
        bc = nx.betweenness_centrality(U)
        betweenness_centrality = np.array([bc[g] for g in U])
        # compute egeinvector centrality and transform the output to vector
        ec = nx.eigenvector_centrality(U)
        eigenvector_centrality = np.array([ec[g] for g in U])
        
        # create a matrix of all subjects centralities
        CC = np.vstack((CC, closeness_centrality))
        BC = np.vstack((BC, betweenness_centrality))
        EC = np.vstack((EC, eigenvector_centrality))
        
    topology.append(CC)#0
    topology.append(BC)#1
    topology.append(EC)#2
    
    return topology