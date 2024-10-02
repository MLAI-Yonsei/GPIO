from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gc
from model import PerceiverIO
from Graph_Data import Graph_Data
from GraphPE import GraphPE
from util import k_fold, data_processing
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import argparse

gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='**test**')
parser.add_argument('--l_d', required=True, type=int, help='latent_dim')
parser.add_argument('--n_l', required=True, type=int, help='num_latents')
parser.add_argument('--l_dh', required=True, type=int, help='latent_dim_head')
parser.add_argument('--c_dh', required=True, type=int, help='cross_dim_head')
parser.add_argument('--l_h', required=True, type=int,  help='latent_heads')
parser.add_argument('--c_h', required=True, type=int,  help='cross_heads')
parser.add_argument('--depth', required=True, type=int,  help='self_per_cross_attn')
parser.add_argument('--lr', required=True, type=float,  help='learning_rate')
parser.add_argument('--wd', required=True, type=float,  help='weight_decay')
parser.add_argument('--p_dim', required=True, default=64, type=int,  help='pe_dimension')
parser.add_argument('--epoch', required=True, default=300, type=int,  help='epoch')
parser.add_argument('--seed', required=True, default=2025,type=int,  help='seed')
parser.add_argument('--dataset', required=True, default='MUTAG',  help='dataset')

args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random_seed = args.seed
dataset_name = args.dataset
print('seed:', random_seed)
print('dataset: ', dataset_name)


torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


dataset = TUDataset(root='data/TUDataset', name=dataset_name)

dataset = data_processing(dataset, args.dataset)
node_pe = GraphPE(dataset, args.p_dim)

x_dim = dataset[0].x.shape[1]    
n_class = dataset.num_classes


criterion = nn.CrossEntropyLoss() 


@torch.no_grad()
def test(data_loader):
    model.eval()
    count =0
    accs = 0
    losses = 0
    for d, l in data_loader:
      data_, label = d.to(device), l.to(device)
      out  = model(data_)
      loss = criterion(out.view(1,-1), label.view(1))
      accs += (out.argmax(-1) == label).sum() 
      count+= data_.shape[0]
      losses += loss
    
    return float(losses/count), float(accs/count)



best_acc_list = []
min_loss_list = []


for fold, (train_idx, test_idx,
          val_idx) in enumerate(zip(*k_fold(dataset, 10, random_seed))):


  train_data = [node_pe[i] for i in list(train_idx)]
  test_data = [node_pe[i] for i in list(test_idx)]
  val_data = [node_pe[i] for i in list(val_idx)]

  train_label = dataset[train_idx]
  test_label  = dataset[test_idx]
  val_label  = dataset[val_idx]

  train_size = len(train_data)
  train_batch_size = 128

  step = train_size // train_batch_size
  rem = train_size - step*train_batch_size
  to = step*train_batch_size


  train_loader = DataLoader(Graph_Data(train_data, train_label),
                              batch_size=1, shuffle=True)

  test_loader = DataLoader(Graph_Data(test_data, test_label),
                              batch_size=1, shuffle=False)
                              
  val_loader = DataLoader(Graph_Data(val_data, val_label),
                              batch_size=1, shuffle=False)

  model = PerceiverIO(
          depth=args.depth,
          dim = args.p_dim + x_dim,
          queries_dim = 2,
          logits_dim = n_class,
          num_latents = args.n_l,
          latent_dim = args.l_d,
          cross_heads = args.c_h,
          latent_heads = args.l_h,
          cross_dim_head = args.c_dh,
          latent_dim_head = args.l_dh,
          weight_tie_layers = False,
          decoder_ff = False
      ).to(device)
      

  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
  print("fold: ", fold+1)

  min_loss = 1000
  best_acc = 0

  for epoch in range(1, args.epoch+1):
    count =0
    losses = 0
    accs = 0
    optimizer.zero_grad()
    for d, l in train_loader:
      data_, label = d.to(device), l.to(device)
      model.train()
      out = model(data_)
      loss = criterion(out.view(1,-1), label.view(1))

      count+= data_.shape[0]

      if count > to:
        loss = loss / rem
        loss.backward()
        losses += (loss * rem)
      else:
        loss = loss / train_batch_size
        loss.backward()
        losses += (loss * train_batch_size)

      accs += (out.argmax(-1) == label).sum() 

      if (count) % (train_batch_size) == 0 or count == train_size:
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 50 == 0:
      for param_group in optimizer.param_groups:
          param_group['lr'] = 0.5 * param_group['lr']   


    test_loss, test_acc = test(test_loader)
    val_loss, val_acc = test(val_loader)


    if val_loss < min_loss :
      min_loss = val_loss
      best_acc = test_acc
    
  print("best_acc: ", best_acc)

  
  best_acc_list.append(best_acc)
  min_loss_list.append(min_loss)

print("best_acc_mean: ", np.mean(best_acc_list))
print("best_acc_std: ", np.std(best_acc_list))
print("min_loss_mean: ",  np.mean(min_loss_list))
