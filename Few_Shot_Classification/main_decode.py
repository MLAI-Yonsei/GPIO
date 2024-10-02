from data import MiniImagenetLoader
from model_decode import PerceiverIO
from extractor import FeatureExtractor
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import shutil
import os
import random
import torch
import numpy as np
from util import build_sim

class ModelTrainer(object):
    def __init__(self,
                 feature_extractor,
                 model,
                 data_loader):
        
        self.feature_extractor = feature_extractor
        self.model = model


        # get data loader
        self.data_loader = data_loader

        # set optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)   


        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

        self.edge_loss = nn.BCELoss(reduction='none')
        self.sig = nn.Sigmoid()

    def train(self):
        val_acc = self.val_acc

        num_supports = args.num_ways_train * args.num_shots_train
        num_queries = args.num_ways_train * 1   
        num_samples = num_supports + num_queries


        support_edge_mask = torch.zeros(args.meta_batch_size, num_samples, num_samples).to(args.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask

        

        # for each iteration
        for iter in range(self.global_step + 1, args.train_iteration + 1):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step = iter

            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader['train'].get_task_batch(num_tasks=args.meta_batch_size,
                                                                     num_ways=args.num_ways_train,
                                                                     num_shots=args.num_shots_train,
                                                                     seed=iter + args.seed,
                                                                     device=args.device)
            
            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)  
        

            # set as train mode

            self.model.train()


            full_data = [self.feature_extractor(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1) # batch_size x num_samples x featdim



            out_query = full_data

            logit, out_node = self.model(full_data[:, num_supports:], queries=out_query)    


            loss = self.edge_loss((logit), (full_edge))  


            pos_loss = torch.sum(loss * query_edge_mask * full_edge ) / torch.sum(query_edge_mask * full_edge )
            neg_loss = torch.sum(loss * query_edge_mask * (1-full_edge)) / torch.sum(query_edge_mask * (1-full_edge))
            pos_neg_loss = pos_loss+neg_loss

            total_edge_loss = pos_neg_loss



            edge_acc = self.hit(logit, full_edge.long()) 
            query_edge_acc = torch.sum(edge_acc * query_edge_mask ) / torch.sum(query_edge_mask ) 


            sim = torch.matmul(out_node, out_node.transpose(1, 2))
            logit_node = self.sig(sim)   


            loss_node = self.edge_loss((logit_node), (full_edge))  

            pos_loss_node = torch.sum(loss_node * query_edge_mask * full_edge) / torch.sum(query_edge_mask * full_edge )
            neg_loss_node = torch.sum(loss_node * query_edge_mask * (1-full_edge)) / torch.sum(query_edge_mask * (1-full_edge))
            pos_neg_loss_node = pos_loss_node+neg_loss_node

            total_node_loss = pos_neg_loss_node

            logit_edge_and_node = logit*args.weight + logit_node*(1-args.weight)

            query_node_pred= torch.bmm(logit_edge_and_node[:,num_supports:, :num_supports], self.one_hot_encode(args.num_ways_train, support_label.long())) 
            query_node_acc = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean() 


            total_loss = total_edge_loss*args.weight + total_node_loss*(1-args.weight)

            # update model
            total_loss.backward()
            self.optimizer.step()
                

            self.adjust_learning_rate(optimizers=[self.optimizer],
                                      lr=args.lr,
                                      iter=self.global_step)


            # evaluation
            if self.global_step % args.test_interval == 0:
                node_mean, node_std, edge_mean, edge_std = self.eval(partition='val')

                is_best = 0

                if node_mean >= self.val_acc:    
                    self.val_acc = node_mean
                    torch.save(self.model.state_dict(), f'./GIPO+_PARAMS'+str(args.run)+'.pt')
                    is_best = 1
                
                
                print("global: ", self.global_step)
                print("val_node_acc: ",self.val_acc)
                print("train loss: ", total_loss.item())
                print("train node acc: ", query_node_acc.item())
                print("train edge acc: ", query_edge_acc.item())     
                
                print()




    def eval(self, partition='test', log_flag=True):
        best_acc = 0

        num_supports = args.num_ways_test * args.num_shots_test
        num_queries = args.num_ways_test * 1
        num_samples = num_supports + num_queries


        support_edge_mask = torch.zeros(args.test_batch_size, num_samples, num_samples).to(args.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask

        edge_acc_list = []
        node_acc_list = []

        # for each iteration
        for iter in range(args.test_iteration//args.test_batch_size):
            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader[partition].get_task_batch(num_tasks=args.test_batch_size,
                                                                       num_ways=args.num_ways_test,
                                                                       num_shots=args.num_shots_test,
                                                                       seed=iter)

            # set as single data
            full_data = torch.cat([support_data, query_data], 1)
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)  


            init_edge = full_edge.clone()

     
            # set as train mode
            self.feature_extractor.eval()
            self.model.eval()

            full_data = [self.feature_extractor(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
            full_data = torch.stack(full_data, dim=1) # batch_size x num_samples x featdim
            out_query = full_data

            logit, out_node = self.model(full_data[:, num_supports:], queries=out_query)    


            edge_accr = self.hit(logit, full_edge.long()) 
            query_edge_accr = torch.sum(edge_accr * query_edge_mask ) / torch.sum(query_edge_mask ) 


            sim = torch.matmul(out_node, out_node.transpose(1, 2))
            logit_node = self.sig(sim)   

            logit_edge_and_node = logit*args.weight + logit_node*(1-args.weight)

            query_node_pred = torch.bmm(logit_edge_and_node[:,num_supports:, :num_supports], self.one_hot_encode(args.num_ways_train, support_label.long())) 
            query_node_accr = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean() 

            node_acc_list.append(query_node_accr.item())
            edge_acc_list.append(query_edge_accr.item())   
      
        return np.array(node_acc_list).mean() * 100, np.array(node_acc_list).std() * 100, np.array(edge_acc_list).mean() * 100, np.array(edge_acc_list).std() * 100



    def adjust_learning_rate(self, optimizers, lr, iter):
        new_lr = lr * (0.5 ** (int(iter / args.dec_lr)))

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def label2edge(self, label):
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge
        edge = torch.eq(label_i, label_j).float().to(args.device)

        return edge    

    def hit(self, logit, label):
        logit_ = logit.clone()
        logit_[logit_>=0.5] = 1
        logit_[logit_<0.5] = 0
        hit = torch.eq(logit_, label).float()
        return hit

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].to(args.device)



if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='**test**')
    # model parameter related


    parser.add_argument('--l_d', type=int, help='latent_dim')
    parser.add_argument('--n_l', type=int, help='num_latents')
    parser.add_argument('--l_dh', type=int, help='latent_dim_head')
    parser.add_argument('--c_dh', type=int, help='cross_dim_head')
    parser.add_argument('--l_h', type=int,  help='latent_heads')
    parser.add_argument('--c_h', type=int,  help='cross_heads')
    parser.add_argument('--depth', type=int,  help='depth')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset_root', default='/data/mini/')

    parser.add_argument('--num_ways', default=5, type=int)
    parser.add_argument('--num_shots', default=1, type=int)
    parser.add_argument('--num_unlabeled', default=0, type=int)

    parser.add_argument('--meta_batch_size', default=40, type=int)
    parser.add_argument('--seed', default=2025, type=int)

    parser.add_argument('--train_iteration', default=10000, type=int)
    parser.add_argument('--test_iteration', default=1000, type=int)
    parser.add_argument('--test_interval', default=2000, type=int)

    parser.add_argument('--test_batch_size', default=10, type=int)
    parser.add_argument('--dec_lr', default=5000, type=int)
    parser.add_argument('--run', default=1, type=int)

    args = parser.parse_args()
    
    
    
    args.weight = 0.99

    args.num_ways_train = args.num_ways
    
    args.num_ways_test = args.num_ways

    args.num_shots_train = args.num_shots
    args.num_shots_test = args.num_shots



    #set random seed

    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


    train_loader = MiniImagenetLoader(root=args.dataset_root, partition='train')
    valid_loader = MiniImagenetLoader(root=args.dataset_root, partition='val')


    data_loader = {'train': train_loader,
                  'val': valid_loader
                   }


    # Initialize the model
    full_model = models.vgg16(pretrained=True)
    f_extractor = FeatureExtractor(full_model)
    f_extractor = f_extractor.to(args.device)

    for param in f_extractor.parameters():   # Freeze param of feature extractor *****
        param.requires_grad  = False


    perceiver = PerceiverIO(
          depth=args.depth,
          dim = 4096,
          queries_dim =  4096,
          edge_logits_dim = None,
          node_logits_dim = None,
          num_latents = args.n_l,
          latent_dim = args.l_d,
          cross_heads = args.c_h,
          latent_heads = args.l_h,
          cross_dim_head = args.c_dh,
          latent_dim_head = args.l_dh,
          weight_tie_layers = False,
          decoder_ff = False
      ).to(args.device)

    # create trainer
    trainer = ModelTrainer(feature_extractor=f_extractor,
                           model=perceiver,
                           data_loader=data_loader)


    trainer.train()

    #### test ####

    perceiver = PerceiverIO(
          depth=args.depth,
          dim = 4096,
          queries_dim =  4096,
          edge_logits_dim = None,
          node_logits_dim = None,
          num_latents = args.n_l,
          latent_dim = args.l_d,
          cross_heads = args.c_h,
          latent_heads = args.l_h,
          cross_dim_head = args.c_dh,
          latent_dim_head = args.l_dh,
          weight_tie_layers = False,
          decoder_ff = False
      ).to(args.device)


    test_loader = MiniImagenetLoader(root=args.dataset_root, partition='test')
    data_loader = {'test': test_loader}

    perceiver.load_state_dict(torch.load('./GIPO+_PARAMS'+str(args.run)+'.pt'))
    tester = ModelTrainer(feature_extractor=f_extractor,
                           model=perceiver,
                           data_loader=data_loader)

    node_mean, node_std, edge_mean, edge_std  = tester.eval(partition='test')
    print("test node acc :", node_mean)
    print("test edge acc :" , edge_mean)
    print()



