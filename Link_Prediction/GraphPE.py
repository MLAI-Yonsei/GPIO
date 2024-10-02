from re import X
import scipy.sparse as sp
import numpy as np
import torch


def GraphPE(x, edge_index ,p_dim):  
    pos_enc_dim = p_dim
    if p_dim == 0:
      return x
    
    graph = x
    edge = edge_index


    A = sp.coo_matrix((np.ones(edge.shape[1]), (edge[0, :], edge[1, :])),
                                shape=(graph.shape[0], graph.shape[0]),
                                dtype=np.float32)

    rowsum = np.array(A.sum(1)).clip(1)
    r_inv = np.power(rowsum, -1.0).flatten()     
    Dinv = sp.diags(r_inv)     # degree matrix


    RW = A * Dinv  
    M = RW

    # Iterate
    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]   # original

    M_power = M
    for _ in range(nb_pos_enc-1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE,dim=-1)

    en_data = torch.cat((graph, PE), dim=1) 
  
    return en_data