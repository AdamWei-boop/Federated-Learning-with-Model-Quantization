# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 19:31:22 2021

@author: adamwei
"""

import numpy as np
import copy
import pandas as pd
import torch
import math

from numpy import linalg 
from sklearn.cluster import KMeans
from csvec import CSVec
from calculate import get_2_norm


def kmeans_opt(quantile_level, orig_values_increment):
    quantile_buck = np.zeros(quantile_level+1)
    quantile_buck[-1] = max(orig_values_increment)+(1e-5)
    quantile_buck[0] = min(orig_values_increment)-(1e-5)
    kmeans = KMeans(n_clusters = quantile_level)
    x=np.array(orig_values_increment).reshape(-1,1).astype(np.float64)
    kmeans.fit(x) 
    quantile_value_1 = kmeans.cluster_centers_
    quantile_value = sorted([quantile_value_1[i][0] for i in range(quantile_level)])
    for i in range(quantile_level-1):
        quantile_buck[i+1] = (quantile_value[i]+quantile_value[i+1])/2
        
    quantile_index = pd.cut(orig_values_increment, quantile_buck, right=True, labels=range(len(quantile_buck)-1))
    values_increment = [quantile_value[i] for i in quantile_index]
    
    return quantile_buck, quantile_index, values_increment


def QSGD(quantile_level,orig_values_increment):
    
    norm = np.linalg.norm(orig_values_increment, ord=2, axis=None, keepdims=False)
    interval = 2*norm/quantile_level
    quantile_buck = [-norm+i*interval for i in range(int(quantile_level/2))] + [i*interval for i in range(int(quantile_level/2)+1)]
    quantile_index = pd.cut(orig_values_increment, quantile_buck, right=True, labels=range(quantile_level))

    quantile_index_l = [abs(int(i-quantile_level/2+0.5)) for i in quantile_index]
    dec_value = [quantile_index_l[i]+1-0.5*quantile_level*abs(orig_values_increment[i])/norm for i in range(len(orig_values_increment))]
    random_value = np.random.rand(len(orig_values_increment))
    
    values_increment = copy.deepcopy(orig_values_increment)    
    for i in range(len(orig_values_increment)):
        if random_value[i] < dec_value[i]:
            values_increment[i] = quantile_buck[quantile_index[i]]
        else:
            values_increment[i] = quantile_buck[quantile_index[i]+1]
    
    return quantile_buck, quantile_index, values_increment

def SVD_Split(quantile_level,orig_values_increment):
    #-------input arr handle  -----------------------
    length=len(orig_values_increment)
    #print('length =',length)
    #print('sqrt = ',np.ceil(math.sqrt(length)))
    matrix_size = int(np.ceil(math.sqrt(length)))
    orig_arr = np.zeros(matrix_size*matrix_size)
    for i in range(len(orig_values_increment)):
        orig_arr[i] =orig_values_increment[i]
    orig_matrix =orig_arr.reshape(matrix_size,matrix_size)
    #-----------svd split ---------
    u,sigma,vt = linalg.svd(orig_matrix)
    #print(orig_values_increment)
    S = np.zeros([len(sigma),len(sigma)])
    R = int(np.ceil(len(orig_values_increment)*math.log2(quantile_level)/(32*(2*matrix_size+1))))
    #R = 4
    #print('math.log2(quantile_level)= ',math.log2(quantile_level))
    # print( 'R = ',R)
    u1 =u[:,:R]
    vt1 =vt[:R,:]
    #print('\n -------------sigma= ',sigma[:R])
    #print('\n -------------sigma= ',sigma[-R:])
    S1 = np.zeros([R,R])
    for i in range(R):
        S1[i][i] =sigma[i]
    tmp = np.dot(u1,S1)
    values_increment_ =np.dot(tmp,vt1).reshape(1,matrix_size*matrix_size)
    values_increment = []
    for i in range(length):
        values_increment.append(values_increment_[0,i])
    #print('\n---------\n',values_increment)

    quantile_err = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
    #print('\n err = ',quantile_err)
    transmist_bit = (2*R*len(sigma)+R)*32
    # print('transmit_bit = ',transmist_bit,'bit')
    return values_increment


def count_sketch(grad, table_size):
    device = 'cpu'

    # compress the gradient if needed
    sketch = CSVec(d=len(grad), c=table_size[1],
        r=table_size[0], device=device,
        numBlocks=table_size[2])
    sketch.accumulateVec(grad)
    # gradient clipping

    hash_table = sketch.table
    unSketch_grad = sketch.unSketch(k=len(grad))
    #print('grad = ',update.size())
    #print('grad = ',grad.size())
    #print('g = ',g)
    #print(unSketch_grad)
    return hash_table, unSketch_grad, grad


def quant_recover_boundary(w, bucket_boundary):

    quant_update = copy.deepcopy(w)
    for i in w.keys():
        for j in range(len(bucket_boundary)-1):
            locations_bucket = (quant_update[i] > bucket_boundary[j]) & (quant_update[i] <= bucket_boundary[j+1])
            quant_update[i][locations_bucket] = (bucket_boundary[j] + bucket_boundary[j+1])/2
    
    return quant_update

def quant_recover_values(w, quant_values):
    quant_update = copy.deepcopy(w)
    m = 0
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            quant_update[i][index] = torch.tensor(quant_values[m])
            m = m + 1
            
    return quant_update

class quant_process(object):
    def __init__(self, sketch_sche, w_update, quant_level, base_bits):
        
        self.sketch_sche = sketch_sche
        self.quant_level = quant_level
        self.w_update = w_update
        self.base_bits = base_bits
        self.table_size = [quant_level, 200, 1]
    
        self.values_update = []                          
        for i in w_update.keys():
            self.values_update += list(w_update[i].view(-1).cpu().numpy())
    
    def quant(self):
            
        if self.sketch_sche == 'bucket_quantile':
            
            q = [i/self.quant_level for i in range(self.quant_level)]
            quantile_bucket = np.quantile(self.values_update, q, axis = None)
            
            
            min_value, max_value = min(self.values_update)-(1e-5), max(self.values_update)+(1e-5)
            quantile_bucket = [min_value] + quantile_bucket + [max_value]
 
            quant_w = quant_recover_boundary(self.w_update, quantile_bucket)
            
            communication_cost = self.base_bits * self.quant_level + np.ceil(np.log2(len(self.values_update)))
        
        elif self.sketch_sche == 'bucket_uniform':
            
            min_value, max_value = min(self.values_update)-(1e-5), max(self.values_update)+(1e-5)
            _, uniform_bucket = np.histogram(self.values_update,
                                                bins=self.quant_level,
                                                range=[min_value, max_value],
                                                weights=None,
                                                density=False)
            quant_w = quant_recover_boundary(self.w_update, uniform_bucket)
            
            communication_cost = self.base_bits * self.quant_level + np.ceil(np.log2(len(self.values_update)))
        
        elif self.sketch_sche == 'kmeans':
        
            _, _, quant_values = kmeans_opt(self.quant_level, self.values_update)
            quant_w = quant_recover_values(self.w_update, quant_values)
            
            communication_cost = self.base_bits * self.quant_level + np.ceil(np.log2(len(self.values_update)))

        elif self.sketch_sche == 'QSGD':

            _, _, quant_values = QSGD(self.quant_level, self.values_update)
            quant_w = quant_recover_values(self.w_update, quant_values)
            
            communication_cost = self.base_bits * self.quant_level + np.ceil(np.log2(len(self.values_update)))
            
        elif self.sketch_sche == 'count_sketch':

            grad = torch.from_numpy(np.array(self.values_update))
            #g =sketch_new(w_increas,orig_values_increment,num_rows,num_cols, compute_grad=True)
            # grad_size =  len(self.values_update)
            hash_table, grad_unsketched, grad = count_sketch(grad, self.table_size)
            quant_values = grad_unsketched.numpy()
            quant_w = quant_recover_values(self.w_update, quant_values)

            communication_cost = self.base_bits * self.table_size[0] * self.table_size[1]\
                + np.ceil(np.log2(self.table_size[0] * self.table_size[1]))
            
        elif self.sketch_sche == 'SVD_Split':
            
            quant_values = SVD_Split(self.quant_level, self.values_update)
            quant_w = quant_recover_values(self.w_update, quant_values)
            
            communication_cost = self.base_bits * len(quant_values)
            
        else:
            
            print('\nNotice: no quantization')
            quant_w = self.w_update
            communication_cost = self.base_bits * len(self.values_update)
            
        mse_error = get_2_norm(quant_w, self.w_update)
        
        return quant_w, communication_cost, mse_error
