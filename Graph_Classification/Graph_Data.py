from torch.utils.data import Dataset
import scipy.sparse as sp
import numpy as np
import torch

class Graph_Data(Dataset):
  def __init__(
    self,
    node_pe, 
    label
  ):
    super(Graph_Data, self).__init__()


    self.node_pe = node_pe
    self.label = label


  def __getitem__(self, idx):
    node_pe = self.node_pe[idx]
    label = self.label[idx].y

    return node_pe, label

  def __len__(self):
    return len(self.node_pe)