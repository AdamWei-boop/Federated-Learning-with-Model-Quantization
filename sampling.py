#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import copy
import itertools
from random import shuffle
from torchvision import datasets, transforms

# np.random.seed(1)
def unique_index(L,f):
    return [i for (i,value) in enumerate(L) if value==f]

def mnist_iid(args,dataset, num_users, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """ 
    print('len(dataset) =',len(dataset))      
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]     
    if len(dataset) == 60000:
        if args.strict_iid == True:
            labels = dataset.train_labels.numpy()
            classes = np.unique(labels)
            classes_index = []
            for i in range(len(classes)):
                classes_index.append(unique_index(labels, classes[i]))
            for i in range(num_users):
                num_items_i= num_items
                num_digits = int(num_items_i/10)
                c = []
                for j in range(10):
                    b = (np.random.choice(classes_index[j], num_digits,\
                                          replace=False))
                    for m in range(num_digits):
                        c.append(b[m])
                # print(c)
                dict_users[i] = set(c)
        else:
            dict_users, all_idxs = {}, [i for i in range(len(dataset))]
            for i in range(num_users):
                num_items_i= num_items
                dict_users[i] = set(np.random.choice(all_idxs, num_items_i, replace=False))
                if num_users*num_items_i <= len(dataset):
                    all_idxs = list(set(all_idxs) - dict_users[i])
    else:
        c = set(np.random.choice(all_idxs, num_items, replace=False))
        for i in range(num_users):  
            dict_users[i] = copy.deepcopy(c)
            # print("\nDivide", len(all_idxs))                      
    return dict_users

def mnist_noniid(args, dataset, num_users, num_items):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if len(dataset) == 60000:
        # divide and assign
        num_digit_noniid = 5
        print('non-iid labels = ',num_digit_noniid)
        dict_users = {}
        labels = dataset.train_labels.numpy()

        classes = np.unique(labels)
        classes_index = []
        for i in range(len(classes)):
            classes_index.append(unique_index(labels, classes[i]))
        
        digit_ch_list = list(itertools.combinations(range(len(classes)), num_digit_noniid))
        digit_ch_idx = [i for i in range(len(digit_ch_list))]
        shuffle(digit_ch_idx)
        digit_ch_list_stor = copy.deepcopy(digit_ch_list)
        
        k = 0
        for i in digit_ch_idx:
            digit_ch_list[i]=copy.deepcopy(digit_ch_list_stor[k])
            k += 1

        num_group = int(num_users/len(args.ratio_train))
        for i in range(num_users):
            num_items_i= int(args.ratio_train[int(i/num_group)]*num_items)
            num_items_iid = int(np.ceil((1-args.degree_noniid)*num_items_i/len(classes)))
            num_items_noniid = int(np.ceil(args.degree_noniid*num_items_i/num_digit_noniid))            
            
            c = []
            for j in range(len(classes)):
                b = (np.random.choice(classes_index[j],int(num_items_iid),\
                                      replace=False))
                classes_index[j]=list(set(classes_index[j])-set(b))
                for m in range(num_items_iid):
                    c.append(b[m])
            for j in list(digit_ch_list[i]):
                b = (np.random.choice(classes_index[j],int(num_items_noniid),\
                                      replace=False))
                classes_index[j]=list(set(classes_index[j])-set(b))
                for m in range(num_items_noniid):
                    c.append(b[m])
            dict_users[i] = set(c)
        
    else:
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        c = set(np.random.choice(all_idxs, num_items, replace=False))
        for i in range(num_users):
            dict_users[i] = copy.deepcopy(c)
#            if num_users*num_items <= len(dataset):
#                all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_iid(dataset, num_users, num_items):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# Divide into 100 portions of total data. Allocate 2 random portions for each user
def cifar_noniid(args, dataset, num_users, num_items):
    num_noniid = 8
    print('non-iid labels = ',num_noniid)
    dict_users = {}
    # labels = dataset.train_labels.numpy()
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])
    # print(labels)
        
    classes = np.unique(labels)
    classes_index = []
    for i in range(len(classes)):
        classes_index.append(unique_index(labels, classes[i]))
    
    cifar_ch_list = list(itertools.combinations(range(len(classes)), num_noniid))
    cifar_ch_idx = [i for i in range(len(cifar_ch_list))]
    shuffle(cifar_ch_idx)
    cifar_ch_list_stor = copy.deepcopy(cifar_ch_list)
    
    k = 0
    for i in cifar_ch_idx:
        cifar_ch_list[i]=copy.deepcopy(cifar_ch_list_stor[k])
        k += 1

    num_group = int(num_users/len(args.ratio_train))
    for i in range(num_users):
        num_items_i= int(args.ratio_train[int(i/num_group)]*num_items)
        num_items_iid = int(np.ceil((1-args.degree_noniid)*num_items_i/len(classes)))
        num_items_noniid = int(np.ceil(args.degree_noniid*num_items_i/num_noniid))            
        
        c = []
        for j in range(len(classes)):
            b = (np.random.choice(classes_index[j],int(num_items_iid),\
                                  replace=False))
            classes_index[j]=list(set(classes_index[j])-set(b))
            for m in range(num_items_iid):
                c.append(b[m])
        for j in list(cifar_ch_list[i]):
            b = (np.random.choice(classes_index[j],int(num_items_noniid),\
                                  replace=False))
            classes_index[j]=list(set(classes_index[j])-set(b))
            for m in range(num_items_noniid):
                c.append(b[m])
        dict_users[i] = set(c)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)