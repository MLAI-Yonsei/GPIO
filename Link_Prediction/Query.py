import scipy.sparse as sp
import numpy as np
import torch
from torch_geometric.nn import APPNP

def Query(x, edge_index):  

    graph = x
    edge = edge_index


    A = sp.coo_matrix((np.ones(edge.shape[1]), (edge[0, :], edge[1, :])),
                               shape=(graph.shape[0], graph.shape[0]),
                               dtype=np.float32)

    A = A + sp.identity(graph.shape[0])     # add self loop
    rowsum = np.array(A.sum(1)).clip(1)
    r_inv = np.power(rowsum, -0.5).flatten()     
    Dinv = sp.diags(r_inv)     # invers of degree matrix
    
    
    S = Dinv * A * Dinv
    smooth_data = S.dot(graph)  
    #smooth_data = S.dot(smooth_data)
    
        
    # norm = APPNP(10, 0.1)
    # smooth_data = norm(graph, edge)
  
    # return smooth_data.clone().detach().requires_grad_(True).type(graph.type())   # for APPNP
    return torch.tensor(smooth_data).type(graph.type())    
    # return graph  