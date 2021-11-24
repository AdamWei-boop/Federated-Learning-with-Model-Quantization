#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
import os
import argparse
import copy
import pandas as pd
import numpy as np
import random
import torch
from torchvision import datasets, transforms

from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from update import LocalUpdate
from fednets import MlPModel, MLPMnist, CNNMnist, CNN_test, CNNCifar, MLPAdult,CNNFashionMnist
from averaging import average_weights
from calculate import subtract, add
from quantilization_sche import quant_process
from sklearn.model_selection import train_test_split
    
def main(args): 
    
    # if not os.path.exists('./experiresult'):
    #     os.mkdir('./experiresult')

    # load dataset and split users
    dict_users_train, dict_users_test = {},{}
    dataset_train,dataset_test = [],[]
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./dataset/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]))
        dataset_test = datasets.MNIST('./dataset/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]))
        if args.iid:
            dict_users_train = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            dict_users_test = mnist_iid(args, dataset_test, args.num_users, args.num_items_test) 
        else:
            dict_users_train = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_users_test = mnist_noniid(args, dataset_test, args.num_users, args.num_items_test)
        
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./dataset/cifar/', train=True, transform=transform, target_transform=None, download=True)
        dataset_test = datasets.CIFAR10('./dataset/cifar/', train=False, transform=transform, target_transform=None, download=True)
        if args.iid:
            dict_users_train = cifar_iid(dataset_train, args.num_users, args.num_items_train)
            dict_users_test = cifar_iid(dataset_test, args.num_users, args.num_items_test)
        else:
            dict_users_train = cifar_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_users_test = cifar_noniid(args, dataset_test, args.num_users, args.num_items_test)
            
    elif args.dataset == 'FashionMNIST':
        dataset_train = datasets.FashionMNIST('./dataset/fashion_mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]))
        print('dataset_train = ',dataset_train)
        dataset_test = datasets.FashionMNIST('./dataset/fashion_mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]))
        if args.iid:
            dict_users_train = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            dict_users_test = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)
        else:
            dict_users_train = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_users_test = mnist_noniid(args, dataset_test, args.num_users, args.num_items_test)
            
    elif args.dataset == 'Adult':  #
        dt = pd.read_csv("./datasets/Bin_Adultall.csv")
        data_set = dt.values
        # print(data_set)
        X = data_set[:, :-1].astype(float)
        Y = data_set[:, -1:].astype(int)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, shuffle=True)
        X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)
        X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)

        Y_train = Y_train.reshape(len(Y_train),)
        Y_test = Y_test.reshape(len(Y_test), )
        dataset_train =[]
        dataset_test =[]
        for i in range(len(X_train)):
            dataset_train.append([X_train[i], Y_train[i]])
        for i in range(len(X_test)):
            dataset_test.append([X_test[i], Y_test[i]])
        
        if args.iid:
            dict_users_train = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            dict_users_test = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)
        else:
            dict_users_train = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_users_test = mnist_noniid(args, dataset_test, args.num_users, args.num_items_test)

    img_size = dataset_train[0][0].shape    
    final_train_loss = np.zeros([len(args.set_degree_noniid), len(args.set_quant_level), len(args.set_quant_sche)])
    final_train_acc = np.zeros([len(args.set_degree_noniid), len(args.set_quant_level), len(args.set_quant_sche)])
    final_test_loss = np.zeros([len(args.set_degree_noniid), len(args.set_quant_level), len(args.set_quant_sche)])
    final_test_acc = np.zeros([len(args.set_degree_noniid), len(args.set_quant_level), len(args.set_quant_sche)])
    
    num_combin = len(args.set_degree_noniid)*len(args.set_quant_level)*len(args.set_quant_sche)
    for s in range(num_combin):
        
        index0 = int(s / (len(args.set_quant_level) * len(args.set_quant_sche)))
        index1 = int((s-index0) / len(args.set_quant_sche))
        index2 = s-index0-index1
        args.degree_noniid = args.set_degree_noniid[index0]
        args.quant_sche = args.set_quant_sche[index2] 
        args.quant_level = args.set_quant_level[index1]                
        
        print('\nNon-i.i.d. degree: {}, quantization scheme: {}, quantization level: {}'.format(args.degree_noniid, args.quant_sche, args.quant_level) )
        
        loss_test, loss_train = [], []
        acc_test, acc_train = [], []

        for m in range(args.num_experiments):
            # with open('./Data_distribution/{}_dict_users_save_{}.pkl'.format(args.dataset,m),'rb') as f:
            #     dict_users = pickle.load(f)   
            # with open('./Data_distribution/{}_dict_server_save_{}.pkl'.format(args.dataset,m),'rb') as f:
            #     dict_sever = pickle.load(f) 
            # print('dict_users =',dict_users)
            # print('dict_sever =',dict_sever)
            # build model
            net_glob = None
            
            if args.model == 'mlp':
                
                input_dim = 1
                for x in img_size:
                    input_dim *= x
                
                if args.gpu != -1:
                    torch.cuda.set_device(args.gpu)
                    net_glob = MlPModel(input_dim=input_dim,
                                        hidden_units = args.hidden_units,
                                        num_classes=args.num_classes).cuda()
                else:
                    net_glob = MlPModel(input_dim=input_dim,
                                        hidden_units = args.hidden_units,
                                        num_classes=args.num_classes)
                
            elif args.model == 'cnn' and args.dataset == 'mnist':
                if args.gpu != -1:
                    torch.cuda.set_device(args.gpu)
                    # net_glob = CNNMnist(args=args).cuda()
                    net_glob = CNN_test(args=args).cuda()
                else:
                    net_glob = CNNMnist(args=args)
                    #torch.save(net_glob.state_dict(), './net_glob/cnn_mnist_glob.pth')
            # elif args.model == 'mlp' and args.dataset == 'mnist':
            #     len_in = 1
            #     for x in img_size:
            #         len_in *= x
                    
                    
            #     if args.gpu != -1:
            #         torch.cuda.set_device(args.gpu)
            #         net_glob = MLPMnist(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).cuda()
            #     else:
            #         net_glob = MLPMnist(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes)
            #         #torch.save(net_glob.state_dict(), './net_glob/mlp_mnist_glob.pth')
            
            elif args.model == 'cnn' and args.dataset == 'cifar':
                if args.gpu != -1:
                    net_glob = CNNCifar(args).cuda()
                else:
                    net_glob = CNNCifar(args)
            # elif args.model == 'mlp' and args.dataset == 'FashionMNIST':
            #     len_in = 1
            #     for x in img_size:
            #         len_in *= x
            #     if args.gpu != -1:
            #         torch.cuda.set_device(args.gpu)
            #         net_glob = MLPMnist(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).cuda()
            #     else:
            #         net_glob = MLPMnist(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes)
            #         #torch.save(net_glob.state_dict(), './net_glob/mlp_FashionMNIST_glob.pth')
            elif args.model == 'cnn' and args.dataset == 'FashionMNIST':
                if args.gpu != -1:
                    torch.cuda.set_device(args.gpu)
                    # net_glob = CNNMnist(args=args).cuda()
                    net_glob = CNNFashionMnist(args=args).cuda()
                else:
                    net_glob = CNNFashionMnist(args=args)
                    #torch.save(net_glob.state_dict(), './net_glob/cnn_FashionMNIST_glob.pth')
            # elif args.model == 'mlp' and args.dataset == 'Adult':
            #     if args.gpu != -1:
            #         torch.cuda.set_device(args.gpu)
            #         # net_glob = CNNMnist(args=args).cuda()
            #         net_glob = MLPAdult(args=args).cuda()
            #     else:
            #         net_glob = MLPAdult(args=args)
            #         #torch.save(net_glob.state_dict(), './net_glob/cnn_FashionMNIST_glob.pth')
            else:
                exit('Error: unrecognized model')
            #print("Nerual Net:",net_glob)
        
            net_glob.train()  #Train() does not change the weight values
            # copy weights
            # net_glob.load_state_dict(torch.load('./net_glob/{}_{}_glob.pth'.format(args.model, args.dataset)))
            w_glob = net_glob.state_dict()

                
            # w_size = 0
            # w_size_all = 0
            # for k in w_glob.keys():
            #     size = w_glob[k].size()
            #     if(len(size)==1):
            #         nelements = size[0]
            #     else:
            #         nelements = size[0] * size[1]
            #     w_size += nelements*4
            #     w_size_all += nelements
            
            # print("Size ", k, ": ",nelements*4)
            # print("\n Weight Size:", w_size, " bytes")
            # print("\n Weight & Grad Size:", w_size*2, " bytes")
            # print("\n Each user Training size:", 784* 8/8* args.local_bs, " bytes")
            # print("\n Total Training size:", 784 * 8 / 8 * 60000, " bytes")
            
            # training
            ###  FedAvg Aglorithm  ###                                     
            for iter in range(args.epochs):
                print('\n','*' * 20,f'Experiment: {m}/{args.num_experiments}, Epoch: {iter}/{args.epochs}','*' * 20)
                time_start = time.time() 
                if  args.num_chosenusers < args.num_users:
                    chosenUsers = random.sample(range(args.num_users),args.num_chosenusers)
                    chosenUsers.sort()
                else:
                    chosenUsers = range(args.num_users)
                # print("\nChosen users:", chosenUsers)
                
                train_loss_locals_list, train_acc_locals_list = [], []
                mse_errors = [] 
                w_update_locals = []
                for idx in range(len(chosenUsers)):
                    local = LocalUpdate(args=args, 
                                        dataset=dataset_train, 
                                        idxs=dict_users_train[chosenUsers[idx]], 
                                        )
                    w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                    
                    w_update_local = subtract(w, w_glob)
                    local_quant = quant_process(args.quant_sche, w_update_local, args.quant_level)
                    w_update_local, mse_error = local_quant.quant()
                    
                    train_loss_locals_list.append(loss)
                    train_acc_locals_list.append(acc)                                
                    w_update_locals.append(w_update_local)
                    mse_errors.append(mse_error)
                    
                print('MSE errors:', mse_errors)
                        
                w_glob_update = average_weights(w_update_locals)                                    
                
                w_glob = add(w_glob, w_glob_update)

                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)
                # global test
                test_loss_locals_list, test_acc_locals_list = [], []   
                net_glob.eval()
                for c in range(args.num_users):
                    net_local = LocalUpdate(args=args, 
                                            dataset=dataset_test, 
                                            idxs=dict_users_test[idx])
                    acc, loss = net_local.test(net=net_glob)                  
                    test_acc_locals_list.append(acc)  
                    test_loss_locals_list.append(loss) 
  

                train_loss_avg = sum(train_loss_locals_list) / len(train_loss_locals_list)
                train_acc_avg = sum(train_acc_locals_list) / len(train_acc_locals_list)
                
                test_loss_avg = sum(test_loss_locals_list) / len(test_loss_locals_list)
                test_acc_avg = sum(test_acc_locals_list) / len(test_acc_locals_list)
  
                time_end = time.time()
   
                print('\nRunning time = {:.2f}s'.format(time_end-time_start))
                
                print("\nTrain loss: {}, Train acc: {}".format(train_loss_avg, train_acc_avg))
                print("\nTest  loss: {}, Test acc:  {}".format(test_loss_avg, test_acc_avg))
                
            loss_train.append(train_loss_avg)
            acc_train.append(train_acc_avg)
            loss_test.append(test_loss_avg)
            acc_test.append(test_acc_avg)
        
        final_train_loss[index0][index1][index2] = sum(loss_train) / len(loss_train)
        final_train_acc[index0][index1][index2] = sum(loss_train) / len(loss_train)
        final_test_loss[index0][index1][index2] = sum(loss_test) / len(loss_test)
        final_test_acc[index0][index1][index2] = sum(acc_test) / len(acc_test)
        
        print('\nFinal train loss:', final_train_loss)
        print('\nFinal train acc:', final_train_acc)
        print('\nFinal test loss:', final_test_loss)
        print('\nFinal test acc:', final_test_acc)


if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='FL with model quantilization')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--dataset', default='mnist', help='mnist or FashionMNIST or cifar or Adult')
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--iid', default=True)
    parser.add_argument('--strict_iid', default=True)     
    parser.add_argument('--model', default='mlp', help='mlp or cnn')
    parser.add_argument('--hidden_units', type=list, default=[128,64], help='mlp or cnn')
    parser.add_argument('--lr', default=0.002, help='learning rate')    
    parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')  
    parser.add_argument('--local_ep', type=int, default=5)     
    parser.add_argument('--num_users', type=int, default=50) 
    parser.add_argument('--num_chosenusers', type=int, default=5) 
    parser.add_argument('--num_items_train', type=int, default=800)
    parser.add_argument('--num_items_test', type=int, default=128)    
    parser.add_argument('--local_bs', type=int, default=128)
    
    parser.add_argument('--set_quant_level', type=list, default=[64])
    parser.add_argument('--set_quant_sche', type=list, \
                       default=['bucket_quantile', 'uniform_quantization', 'SVD_Split',\
                                'count_sketch', 'QSGD', 'kmeans'])
    parser.add_argument('--set_degree_noniid', type=list, default=[0])    
    parser.add_argument('--num_experiments', type=int, default=1)
    args = parser.parse_args() 
 
    print('\nStart training')
    run_start_time = time.time()   
    
    main(args)
    
    print('\nExperiment finished')
    run_end_time = time.time()
    print('\nRun time = {} h'.format((run_end_time-run_start_time)/3600))
    
