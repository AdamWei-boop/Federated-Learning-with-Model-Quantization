#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import copy

def args_set_quantile_level():
    set_quantile_level = [8]   # level
    return set_quantile_level

def args_model():
    model = 'cnn'  # 'mlp' 'cnn'
    return model

def args_parameter_ways():
    parameter_ways = 'diff_parameter'  #  'orig_parameter' , 'diff_parameter'
    return parameter_ways
    
def args_lr():
    lr = 0.02
    return lr

def args_dataset():
    dataset = 'cifar'     # 'mnist' 'FashionMNIST' 'cifar'
    return dataset

def args_num_Chosenusers():
    num_Chosenusers = 6
    print('num_Chosenusers =',num_Chosenusers)
    return num_Chosenusers
    
def args_num_users():
    num_users = 20
    print('num_users =',num_users)
    return num_users

def args_num_experiments():
    num_experiments = 1
    return num_experiments

def args_globel_ep():
    globel_ep = 200          # numb of globel iters
    return globel_ep

def args_local_ep():
    local_ep = 5          # numb of local(device) iters
    return local_ep

def args_dataset_iid():
    dataset_iid = False     #  dataset_iid = 'True'  nom-iid ='False'   
    return dataset_iid

def args_degree_noniid():
    degree_noniid = [1]     # iid = 0           non-iid =1 
    return degree_noniid

def args_num_items_train():
    num_items_train = 400  # numb of local data size # 
    print('num_items_train = ',num_items_train)
    return num_items_train

def  args_num_items_test():
    num_items_test =  256 #
    print( 'num_items_test = ',num_items_test)
    return num_items_test

def args_local_bs():
    local_bs = 64        # Local Batch size (1200 = full dataset)
    print( 'local_bs =',local_bs)
    return local_bs       # size of a user for mnist, 2000 for cifar)

    