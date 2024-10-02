from re import X
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import PerceiverIO
from Graph_Data import Graph_Data
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import train_test_split_edges
import torch_geometric.transforms as T
from torch_geometric.nn import GAE
import random
import argparse
from GraphPE import GraphPE
from Query import Query
import gc

gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='**test**')
parser.add_argument('--l_d', required=True, type=int, help='latent_dim')
parser.add_argument('--n_l', required=True, type=int, help='num_latents')
parser.add_argument('--l_dh', required=True, type=int, help='latent_dim_head')
parser.add_argument('--c_dh', required=True, type=int, help='cross_dim_head')
parser.add_argument('--c_h', required=True, default=4, type=int,  help='cross_heads')
parser.add_argument('--l_h', required=True,  type=int,  help='latent_heads')
parser.add_argument('--depth', required=True, type=int,  help='depth')
parser.add_argument('--lr', required=True,  type=float,  help='learning_rate')
parser.add_argument('--wd', required=True, type=float,  help='weight_decay')
parser.add_argument('--p_dim', required=True, type=int,  help='pe_dimension')
parser.add_argument('--training_rate', required=True, type=float,  help='training_rate')
parser.add_argument('--epoch', required=True, type=int,  help='epoch')
parser.add_argument('--seed', default=2025, type=int,  help='seed')
parser.add_argument('--dataset', required=True, help='dataset')


args = parser.parse_args()


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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_rate = args.training_rate
val_ratio = (1-args.training_rate) / 3
test_ratio = (1-args.training_rate) / 3 * 2

transform = T.Compose([
    T.NormalizeFeatures(),
    T.RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])


if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid('data/Planetoid', args.dataset, 'public', transform=transform)


train_data, val_data, test_data = dataset[0]


x_dim = train_data.x.shape[1]    
n_class = dataset.num_classes

criterion = nn.CrossEntropyLoss() 


x, train_pos_edge_index = train_data.x, train_data.edge_index


out_queries_train = Query(x, train_data.edge_index).to(device)
out_queries_val = Query(x, val_data.edge_index).to(device)   
out_queries_test = Query(x, test_data.edge_index).to(device)   

x = GraphPE(x, train_pos_edge_index, args.p_dim).to(device)


def test(data):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, queries=out_queries_test)
        auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return loss, auc , ap

def val(data):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, queries=out_queries_val)
        auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return loss, auc , ap


runs = 10

val_list = []
roc_list = []
ap_list = []

for r in range(runs):
  # create Perciver model
  encoder = PerceiverIO(
          depth=args.depth,
          dim = args.p_dim + x_dim,
          queries_dim =  x_dim,
          logits_dim = None,
          num_latents = args.n_l,
          latent_dim = args.l_d,
          cross_heads = args.c_h,
          latent_heads = args.l_h,
          cross_dim_head = args.c_dh,
          latent_dim_head = args.l_dh,
          weight_tie_layers = False,
          decoder_ff = False
      ).to(device)

  model = GAE(encoder).to(device)
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)     
  

  best = 0
  best_AUC = 0 
  best_AP = 0

  for epoch in range(1, args.epoch+1):
    model.train()
    optimizer.zero_grad()
    z  = model.encode(x, queries=out_queries_train)  
    loss = model.recon_loss(z, train_pos_edge_index.to(device))
    loss.backward()
    optimizer.step()
    

    with torch.no_grad():
        test_loss, test_auc, test_ap = test(test_data)
        val_loss, val_auc, val_ac = val(val_data)

        if val_auc > best:
            best = val_auc
            best_AUC = test_auc
            best_AP = test_ap



  roc_list.append(best_AUC)
  ap_list.append(best_AP)
  val_list.append(float(best))


print("best_roc_mean: ", np.mean(roc_list))
print("best_roc_std: ", np.std(roc_list))
print("best_ap_mean: ", np.mean(ap_list))
print("best_ap_std: ", np.std(ap_list))
