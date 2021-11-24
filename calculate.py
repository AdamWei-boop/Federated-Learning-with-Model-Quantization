# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:19:31 2019

@author: WEIKANG
"""
import torch
import numpy as np
import copy
import random
import math

def add(params_a, params_b):
    w = copy.deepcopy(params_a)
    for k in w.keys():
            w[k] += params_b[k]
    return w

def subtract(params_a, params_b):
    w = copy.deepcopy(params_a)
    for k in w.keys():
            w[k] = w[k] - params_b[k]
    return w

def dict_to_list(params_a):
    val_a = []
    for i in params_a.keys():
        val_a += list(params_a[i].view(-1).cpu().numpy())  
    return val_a

def sample_para(w_locals,indices_samp):
    X = []
    for i in range(len(w_locals)):
        val_i = dict_to_list(w_locals[i])  
        X.append([val_i[j] for j in indices_samp])
    return X

def minkowski_distance(params_a, params_b, p):   
    val_a, val_b = [], []
    for i in params_a.keys():
        val_a += list(params_a[i].view(-1).cpu().numpy())
        val_b += list(params_b[i].view(-1).cpu().numpy())
    val_c = [val_a[i]-val_b[i] for i in range(len(val_a))]    
    val_dist = np.linalg.norm(val_c, ord=p)
    return val_dist

def mahala_distance(params_a, params_b, w_locals, num_samp):   
 
    val_a = dict_to_list(params_a) 
    val_b = dict_to_list(params_b)        
    val_c = [val_a[i]-val_b[i] for i in range(len(val_a))] 
    indices_samp = random.sample(range(len(val_a)),num_samp)
    
    # val_a = np.array([val_a[i] for i in indices_samp])
    # val_b = np.array([val_b[i] for i in indices_samp])
    val_c = np.array([val_c[i] for i in indices_samp])
    
    # a_ = val_a[:,np.newaxis]
    # b_ = val_b[:,np.newaxis]
    # X = np.hstack((a_, b_))
    X = sample_para(w_locals,indices_samp)
    X = np.array(X)
    cov_matr = np.cov(X.T)
    inver_cov_matr = np.array(np.matrix(cov_matr).I)  
    val_dist = np.dot(np.dot(val_c, inver_cov_matr), val_c.T)

    return np.sqrt(val_dist)

def DAC_AGR(w_locals):
    
    w_locals_filter = w_locals
    
    return w_locals_filter

def get_l2_norm(args, params_a):
    sum = 0
    if args.gpu != -1:
        tmp_a = np.array([v.detach().cpu().numpy() for v in params_a])
    else:
        tmp_a = np.array([v.detach().numpy() for v in params_a])
    a = []
    for i in tmp_a:
        x = i.flatten()
        for k in x:
            a.append(k)
    for i in range(len(a)):
        sum += (a[i] - 0) ** 2
    norm = np.sqrt(sum)
    return norm

def get_1_norm(params_a):
    sum = 0
    if isinstance(params_a,np.ndarray) == True:
        sum += pow(np.linalg.norm(params_a, ord=2),2) 
    else:
        for i in params_a.keys():
            if len(params_a[i]) == 1:
                sum += pow(np.linalg.norm(params_a[i].cpu().numpy(), ord=2),2)
            else:
                a = copy.deepcopy(params_a[i].cpu().numpy())
                for j in a:
                    x = copy.deepcopy(j.flatten())
                    sum += pow(np.linalg.norm(x, ord=2),2)                  
    norm = np.sqrt(sum)
    return norm

def get_2_norm(params_a, params_b):
    sum = 0
    for i in params_a.keys():
        if len(params_a[i]) == 1:
            sum += pow(np.linalg.norm(params_a[i].cpu().numpy()-\
                params_b[i].cpu().numpy(), ord=2),2)
        else:
            a = copy.deepcopy(params_a[i].cpu().numpy())
            b = copy.deepcopy(params_b[i].cpu().numpy())
            x = []
            y = []
            for j in a:
                x.append(copy.deepcopy(j.flatten()))
            for k in b:          
                y.append(copy.deepcopy(k.flatten()))
            for m in range(len(x)):
                sum += pow(np.linalg.norm(x[m]-y[m], ord=2),2)            
    norm = np.sqrt(sum)
    return norm

def inner_product(params_a, params_b):
    sum = 0
    for i in params_a.keys():
        sum += np.sum(np.multiply(params_a[i].cpu().numpy(),\
                params_b[i].cpu().numpy()))     
    return sum

def avg_grads(g):
    grad_avg = copy.deepcopy(g[0])
    for k in grad_avg.keys():
        for i in range(1, len(g)):
            grad_avg[k] += g[i][k]
        grad_avg[k] = torch.div(grad_avg[k], len(g))
    return grad_avg

def noise_add(args, w, noise_scale, malicious_users):
    w_noise = copy.deepcopy(w)
    if isinstance(w[0],np.ndarray) == True:
        for k in malicious_users:
            noise = np.random.normal(0,noise_scale,w.size())
            w_noise[k] = w_noise[k] + noise
    else:
        for k in malicious_users:
            for i in w[k].keys():
               noise = np.random.normal(0,noise_scale,w[k][i].size())
               if args.gpu != -1:
                   noise = torch.from_numpy(noise).float().cuda()
               else:
                   noise = torch.from_numpy(noise).float()
               w_noise[k][i] = w_noise[k][i] + noise
    return w_noise
