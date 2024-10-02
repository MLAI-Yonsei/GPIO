from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import PerceiverIO
from Graph_Data import Graph_Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import random
import argparse
from GraphPE import GraphPE
from random_mask import random_planetoid_splits
from Query import Query
import gc

gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='**test**')
parser.add_argument('--l_d', required=True, type=int, help='latent_dim')
parser.add_argument('--n_l', required=True, type=int, help='num_latents')
parser.add_argument('--l_dh', required=True, type=int, help='latent_dim_head')
parser.add_argument('--c_dh', required=True, type=int, help='cross_dim_head')
parser.add_argument('--l_h', required=True, type=int,  help='latent_heads')
parser.add_argument('--c_h', required=True, type=int,  help='cross_heads')
parser.add_argument('--depth', required=True, type=int,  help='depth')
parser.add_argument('--lr', required=True, type=float,  help='learning_rate')
parser.add_argument('--wd', required=True, type=float,  help='weight_decay')
parser.add_argument('--p_dim', required=True, type=int,  help='pe_dimension')
parser.add_argument('--random_splits', required=True, type=bool,  help='random_split')
parser.add_argument('--epoch', required=True, default=200, type=int,  help='epoch')
parser.add_argument('--seed', required=True, default=2025,type=int,  help='seed')
parser.add_argument('--dataset', required=True, default='Cora',  help='dataset')


args = parser.parse_args()



random_seed = args.seed
dataset_name = args.dataset
print('seed:', random_seed)
print('dataset: ', dataset_name)
print('rand_split: ', args.random_splits)

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)



dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=T.NormalizeFeatures())
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_dim = data.x.shape[1]    
n_class = dataset.num_classes


criterion = nn.CrossEntropyLoss() 


out_queries = Query(data).to(device)
input_data = GraphPE(data , args.p_dim)
input_data = input_data.view(1,*input_data.shape)



def evaluate(model, input_data):
    model.eval()
    input_data = input_data.to(device)
    with torch.no_grad():
      logits = model(input_data, queries = out_queries).squeeze()

    outs = {}
    for key in ['val', 'test']:
      mask = data[f'{key}_mask']
      loss = criterion(logits[mask], data.y[mask].to(device)).item()
      pred = logits[mask].max(1)[1]
      acc = pred.eq(data.y[mask].to(device)).sum().item() / mask.sum().item()

      outs[f'{key}_loss'] = loss
      outs[f'{key}_acc'] = acc

    return outs


runs = 100
min_loss_list = []
best_acc_list = []

for r in range(runs):
  # create Perciver model
  model = PerceiverIO(
          depth=args.depth,
          dim = args.p_dim + x_dim,
          queries_dim =  x_dim,
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
  
  
  if args.random_splits:
      data = random_planetoid_splits(data, n_class)
      
  min_loss = 1000.0
  best_acc = 0    
  
  for epoch in range(1, args.epoch+1):
    count = data['train_mask'].sum().item()
    losses = 0
    accs = 0
    optimizer.zero_grad()

    in_data, label = input_data.to(device), data.y.to(device)
    model.train()
    out = model(in_data, queries=out_queries).squeeze()
    loss = criterion(out[data.train_mask], label[data.train_mask])
    loss.backward()

    accs += (out[data.train_mask].argmax(-1) == label[data.train_mask]).sum() 
      
    optimizer.step()
    

    eval_info = evaluate(model, input_data)

    
    if eval_info['val_loss'] < min_loss :
      min_loss = eval_info['val_loss']
      best_acc = eval_info['test_acc']
    

  
  min_loss_list.append(min_loss)
  best_acc_list.append(best_acc)


print("min_val_mean: ", np.mean(min_loss_list))
print("best_acc_mean: ", np.mean(best_acc_list))
print("best_acc_std: ", np.std(best_acc_list))
