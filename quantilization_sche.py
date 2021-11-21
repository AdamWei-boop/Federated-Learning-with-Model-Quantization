# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 19:31:22 2021

@author: adamwei
"""

import numpy as np
import copy
import pandas as pd
from sklearn.cluster import KMeans

def quantile_uniform(quantile_level,value_sequence_,sorted_id_):
    quantile_buck = np.zeros(quantile_level+1)
    quantile_index = np.zeros(len(value_sequence_))
    values_increment = np.zeros(len(value_sequence_))
    sorted_id =copy.deepcopy(sorted_id_)
    max_value = max(value_sequence_)+0.000001
    min_value = min(value_sequence_)-0.000001
    delta = (max_value-min_value)/quantile_level
    for i in range(quantile_level+1):
        quantile_buck[i] = min_value + delta*i

    a0= min_value
    a1= min_value + delta
    code = 0
    #print('\n value_sequence_ =',value_sequence_)
    for i in range(len(value_sequence_)):
            X =1
            while X :
                if value_sequence_[i] >= a0 and value_sequence_[i] < a1 :
                    quantile_index[sorted_id[i]] = code  # 二进制编码
                    values_increment[sorted_id[i]] = (a0+a1)/2
                    #print('\n a0= ',a0,'a1=',a1,'   value_sequence_[i] = ',value_sequence_[i])
                    X =0
                else:
                    a0= a1
                    a1= a1 + delta
                    code = code +1
                    X = 1            
    return quantile_buck,quantile_index,values_increment

def quantile_bucket(quantile_level,value_sequence, sorted_id):
    quantile_buck = []
    quantile_index = copy.deepcopy(sorted_id)
    values_increment = copy.deepcopy(value_sequence)
    k, k_ = 0, 0
    for i in range(len(value_sequence)):
        if i == 0:
            quantile_buck.append(value_sequence[i]-0.00000001)
        elif (i+1)%quantile_level == 0:
            if i == len(value_sequence)-1:
                quantile_buck.append(value_sequence[i]+0.0000001)
            else :
                quantile_buck.append(value_sequence[i])
            #----------------- test add ---------------
            #print('\n i = ',i,'quantile_level =',quantile_level)
            c0 = value_sequence[i+1-quantile_level:i].copy()
            #code_value = loss_min(c0)
            code_value = (c0[0]+c0[-1])/2
            #print('\n code value: ',code_value)
            #------------------test ------------------------------
            for j in range(k_,i+1):
                quantile_index[sorted_id[j]] = k
                #values_increment[sorted_id[j]] = (quantile_buck[-1]+quantile_buck[-2])/2
                values_increment[sorted_id[j]] = code_value
            k += 1
            k_ = i
        elif len(value_sequence)%quantile_level != 0 and i == len(value_sequence)-1:
            quantile_buck.append(value_sequence[i] +0.0000001)
            #------------------- test add  -------------------------
            c1 = value_sequence[-(len(value_sequence)%quantile_level):].copy()
            code_value = (c1[0]+c1[-1])/2
            #print('\n code value: ',code_value)
            #code_value = loss_min(c1)
            #--------------------test add ---------------------- 
            for j in range(k_,i+1):
                quantile_index[sorted_id[j]] = k
                #values_increment[sorted_id[j]] = (quantile_buck[-1]+quantile_buck[-2])/2
                values_increment[sorted_id[j]] = code_value 
            k += 1
            k_ = i  
                
    return quantile_buck, quantile_index, values_increment

def kmeans_opt(quantile_level,orig_values_increment):
    quantile_buck = np.zeros(quantile_level+1)
    quantile_buck[-1] = max(orig_values_increment)+0.000001
    quantile_buck[0] = min(orig_values_increment)-0.000001
    kmeans = KMeans(n_clusters = quantile_level)
    x=np.array(orig_values_increment).reshape(-1,1).astype(np.float64)
    kmeans.fit(x) 
    quantile_value_1 = kmeans.cluster_centers_
    quantile_value = sorted([quantile_value_1[i][0] for i in range(quantile_level)])
    for i in range(quantile_level-1):
        quantile_buck[i+1] = (quantile_value[i]+quantile_value[i+1])/2
        
    quantile_index = pd.cut(orig_values_increment, quantile_buck, right=True, labels=range(len(quantile_buck)-1))
    values_increment = [quantile_value[i] for i in quantile_index]
    
    #---------debug-----------
    #print('final kmeans_err = ', np.linalg.norm(np.array(values_increment)-np.array(orig_values_increment), ord=1)) 
    #---------debug------------
    return quantile_buck,quantile_index,values_increment


def QSGD_(quantile_level,orig_values_increment):
    
    norm = np.linalg.norm(orig_values_increment, ord=2, axis=None, keepdims=False) #二范数
    interval = 2*norm/quantile_level
    quantile_buck = [-norm+i*interval for i in range(int(quantile_level/2))] + [i*interval for i in range(int(quantile_level/2)+1)]
    quantile_index = pd.cut(orig_values_increment, quantile_buck, right=True, labels=range(quantile_level))

    #print('quantile_index, quantile_buck, norm =',quantile_index)
    #print('quantile_buck',quantile_buck)
    #print('norm=',norm)
    quantile_index_l = [abs(int(i-quantile_level/2+0.5)) for i in quantile_index]
    dec_value = [quantile_index_l[i]+1-0.5*quantile_level*abs(orig_values_increment[i])/norm for i in range(len(orig_values_increment))]
    random_value = np.random.rand(len(orig_values_increment))
    
    #print("\n ***********\n       dec_value =",dec_value)
    #print('random_value =',random_value) #quantile_index_l, quantile_index)
    
    values_increment = copy.deepcopy(orig_values_increment)    
    for i in range(len(orig_values_increment)):
        if random_value[i] < dec_value[i]:
            values_increment[i] = quantile_buck[quantile_index[i]]
        else:
            values_increment[i] = quantile_buck[quantile_index[i]+1]
    
    return quantile_buck,quantile_index,values_increment