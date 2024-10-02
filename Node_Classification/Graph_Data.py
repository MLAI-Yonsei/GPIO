from torch.utils.data import Dataset
from torch_geometric.utils.subgraph import k_hop_subgraph
import random
import torch


class Graph_Data(Dataset):
  def __init__(
    self,
    data, 
    label,
  ):
    super(Graph_Data, self).__init__()
    self.data = data
    self.label = label
    


  def __getitem__(self, idx):
    data = self.data
    label = self.label

    return data, label

  def __len__(self):
    return 1