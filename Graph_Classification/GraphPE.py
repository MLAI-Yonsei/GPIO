import scipy.sparse as sp
import numpy as np
import torch


def GraphPE(all_dataset , p_dim):  

    node_pe = []

    pos_enc_dim = p_dim

    for i in range(len(all_dataset)):

      if p_dim == 0:
        node_pe.append( all_dataset[i].x )     # no positional embedding
        continue

      A = sp.coo_matrix((np.ones(all_dataset[i].edge_index.shape[1]), (all_dataset[i].edge_index[0, :], all_dataset[i].edge_index[1, :])),
                                  shape=(all_dataset[i].x.shape[0], all_dataset[i].x.shape[0]),
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

      node_pe.append( torch.cat((all_dataset[i].x, PE), dim=1) )
  
    return node_pe