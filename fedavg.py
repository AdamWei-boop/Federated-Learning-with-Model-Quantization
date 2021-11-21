#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from operator import length_hint
import time
import matplotlib
import sys
import pylab
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
plt.switch_backend('agg')
#matplotlib.use('Agg')
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
import copy
import pandas as pd
import math
import numpy as np
import random
import collections
import torch
import torch.nn.functional as F
import sympy as sy
from numpy import linalg as la
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from tqdm import tqdm
from torch import autograd
from tensorboardX import SummaryWriter
from sympy import solve
from sympy.abc import P, y
from scipy import optimize

from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP1, CNNMnist, CNN_test, CNNCifar, MLP3, MLP_triple ,MLP_regression,CNNFashionMnist,VGG
from averaging import average_weights, average_weights_orig
from Calculate import minkowski_distance, mahala_distance, noise_add, sample_para
import pickle
from csvec import CSVec
from sklearn.model_selection import train_test_split
#from utils import get_param_vec, set_param_vec, get_grad, _topk, clip_grad



# 带入不同的X的值，求得在X下不同的loss 总和
def loss_sum(X,value_sequence):
    loss = 0
    for i in range(len(value_sequence)):
        loss =loss + abs(value_sequence[i] -X)
    return loss


# 加速求解 最优 x的值
# 输入： 
# value_sequence  ：任意一维数组
# acc_level ： 无用参数，主要是为了函数方便修改
# 输出：
# x_min  ： loss 最小时，对应的x的值
def fast_find_x_opt(values_increment_,acc_level=5):
    X= 1
    a = min(values_increment_)
    b = max(values_increment_)
    #itertimes = 0
    while X :
        #x1 = a + 0.382*(b-a)
        #x2 = a + 0.618*(b-a)
        x1 = a + 0.382*(b-a)
        x2 = a + 0.618*(b-a)
        x1_loss =loss_sum(x1,values_increment_)
        x2_loss =loss_sum(x2,values_increment_)
        #---------------debug-----------
        #itertimes = itertimes +1
        #plt.scatter(x1,x1_loss,color='b')
        #plt.scatter(x2,x2_loss,color='b')
        #plt.show()
        #print('x1 =',x1,'x2=',x2)
        #print('x1_loss =',x1_loss,'x2_loss=',x2_loss)
        #-------------debug-------------
        if x1_loss >= x2_loss :
            a= x1
            opt_value = x2
            #opt_loss = x2_loss
        else:
            b= x2
            opt_value = x1
            #opt_loss = x1_loss
        if abs(x1-x2) < 0.0001 :
            X = 0
    #---------debug ---------
    #print('itertimes =',itertimes)
    #print('opt_value =',opt_value,'opt_loss=',opt_loss)
    #plt.scatter(opt_value,opt_loss,color='r')
    #plt.show()
    #--------debug ----------
    return opt_value



#  均匀间隔量化：
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组【数组满足有小到大排序】  
#  sorted_id        : 整理好的数组索引

#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数
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
    #量化
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

#  均匀间隔量化--优化
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组【数组满足有小到大排序】  
#  sorted_id        : 整理好的数组索引

#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数

def quantile_uniform_opt(quantile_level,value_sequence_,sorted_id_,orig_values_increment):
    quantile_buck_0 = np.zeros(quantile_level+1)
    quantile_buck_1 = np.zeros(quantile_level+1)
    quantile_index = np.zeros(len(value_sequence_))
    values_increment = np.zeros(len(value_sequence_))
    sorted_id =copy.deepcopy(sorted_id_)

    max_value = max(value_sequence_)+0.000001
    min_value = min(value_sequence_)-0.000001
    
    hist, quantile_buck_ = np.histogram(value_sequence_,bins=quantile_level,range=[min_value,max_value],weights=None,density=False)
    quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(quantile_level,orig_values_increment,sorted_id,quantile_buck_)
    
    quantile_buck =copy.deepcopy(quantile_buck_0) #不调整边界，只做一次优化
    #------------debug------------
    #loss_adapt_err_0 = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
    #print('opt quantile_err = ',loss_adapt_err_0)
    #------------debug------------                
    return quantile_buck,quantile_index,values_increment
    
    

# 基于边界值量化（根据边界调整后的值）
# 输入 ： 
# quantile_level ：量化等级
# value_sequence_：原始参数
# sorted_id_ ： 排序好的数，注：排序由小到大进行排序
# quantile_buck_ ：输入的边界值
# quantile_buck_tmp ： 优化的区间内的值

# 输出 ：
# values_increment ：对原始值基于边界量化后输出的值

def quantile_border_replace(quantile_level,value_sequence_,sorted_id_,quantile_buck_,quantile_buck_tmp):
    quantile_buck_0 = copy.deepcopy(quantile_buck_)
    quantile_buck_1 = np.zeros(quantile_level+1)
    quantile_index = np.zeros(len(value_sequence_))
    values_increment = np.zeros(len(value_sequence_))
    sorted_id =copy.deepcopy(sorted_id_)
    #max_value = max(value_sequence_)+0.000001
    #min_value = min(value_sequence_)-0.000001
    #delta = (max_value-min_value)/quantile_level

    #量化
    quantile_buck_id =0
    a0= quantile_buck_0[0]
    a1= quantile_buck_0[1]
    code = 0
    arr_tmp = []
    #quantile_buck_tmp.append(value_sequence_[0])
    #print('\n value_sequence_ =',value_sequence_)
    for i in range(len(value_sequence_)):
        X =1
        while X :
            if value_sequence_[i] >= a0 and value_sequence_[i] < a1 :
                arr_tmp.append(value_sequence_[i])
                quantile_index[sorted_id[i]] = code  # 二进制编码
                #values_increment[sorted_id[i]] = (a0+a1)/2
                #print('\n a0= ',a0,'a1=',a1,'   value_sequence_[i] = ',value_sequence_[i])
                X =0
            else:
                i_1 = i
                i_0 = int(i - len(arr_tmp))
                if i_0 !=  i_1 :
                    #print('\n len(arr_tmp) = ',len(arr_tmp))
                    #print('\n arr_tmp = ',arr_tmp)
                    #print('i_0 =',i_0,'i_1 =',i_1)
                    code_value =quantile_buck_tmp[code]
                    for index in range(i_0,i_1):
                        values_increment[sorted_id[index]] = code_value
                        #print('code_value =',code_value)
                arr_tmp = []
                quantile_buck_id = quantile_buck_id +1
                a0= a1
                a1= quantile_buck_0[quantile_buck_id+1]
                code = code +1
                X = 1

    if len(arr_tmp) == 0 :
        code_value =code_value =quantile_buck_tmp[-1]
    else :
        i_1 = i
        i_0 = int(i - len(arr_tmp))
        if i_0 !=  i_1 :
            #print('\n len(arr_tmp) = ',len(arr_tmp))
            #print('\n arr_tmp = ',arr_tmp)
            #print('i_0 =',i_0,'i_1 =',i_1)
            code_value =quantile_buck_tmp[-1]
            for index in range(i_0+1,i_1+1):
                values_increment[sorted_id[index]] = code_value

    return values_increment

#  均匀间隔量化优化---ways = 2
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组【数组满足有小到大排序】  
#  sorted_id        : 整理好的数组索引
#  quantile_buck_   : 上次优化后，调整好的边界值

#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数
def quantile_border_opt(quantile_level,value_sequence_,sorted_id_,quantile_buck_):
    quantile_buck_0 = copy.deepcopy(quantile_buck_)
    quantile_buck_1 = np.zeros(quantile_level+1)
    
    quantile_buck_1[0] =quantile_buck_0[0]
    quantile_buck_1[-1]=quantile_buck_0[-1]
  
       
    quantile_buck_tmp = []
    
    cats = pd.cut(value_sequence_, quantile_buck_0,right=True, labels=range(len(quantile_buck_0)-1))
    for i in range(len(quantile_buck_0)-1):
        #time_start = time.time()
        value_bin = np.array(value_sequence_)[cats == i]   
        a = len(value_bin)
        b = -2*sum(value_bin)
        #c = pow(np.linalg.norm(value_bin, ord=2), 2)
        #print(value_bin, quantile_buck_0[i], quantile_buck_0[i+1])
        if a > 0:
            quantile_buck_tmp.append(-b/(2*a))
        else:
            quantile_buck_tmp.append((quantile_buck_0[i]+quantile_buck_0[i+1])/2)
    
    for i in range(len(quantile_buck_0)-2):
        quantile_buck_1[i+1] = (quantile_buck_tmp[i] + quantile_buck_tmp[i+1])/2
    
    values_increment = [quantile_buck_tmp[i] for i in cats]
    
    return quantile_buck_0,quantile_buck_1,cats,values_increment,quantile_buck_tmp


#  均匀间隔量化优化---ways = 2
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组【数组满足有小到大排序】  
#  sorted_id        : 整理好的数组索引

#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数
#----------------------debug ----------------------

def quantile_uniform_opt_proposed(quantile_level,value_sequence_,sorted_id_,orig_values_increment,err = 0.1):
    quantile_buck_0 = np.zeros(quantile_level+1)
    quantile_buck_1 = np.zeros(quantile_level+1)
    quantile_index = np.zeros(len(value_sequence_))
    values_increment = np.zeros(len(value_sequence_))
    sorted_id =copy.deepcopy(sorted_id_)

    max_value = max(value_sequence_)+0.000001
    min_value = min(value_sequence_)-0.000001
    
    hist, quantile_buck_ = np.histogram(value_sequence_,bins=quantile_level,range=[min_value,max_value],weights=None,density=False)
    quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(quantile_level,orig_values_increment,sorted_id,quantile_buck_)
    quantile_buck_ =copy.deepcopy(quantile_buck_1)
    loss_adapt_err_0 = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
    #----------debug-----------------------
    #print('orig quantile_err = ',loss_adapt_err_0)
    #-----------debug-------------------
    #  step1 : quantile_border_opt 调整边界量化
    ###--------------------------------------------###
    ###--------------------------------------------###
    X = 1
    while X:
        #time_start = time.time()  
        quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(quantile_level,orig_values_increment,sorted_id,quantile_buck_)
        #time_end = time.time()
        #print('\nquantile_border_opt time = ',time_end-time_start,' seound')
        #time_start = time.time() 
        loss_adapt_err_1 = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
        #print('\nadapt quantile_err = ',loss_adapt_err_1)
        quantile_buck_ =copy.deepcopy(quantile_buck_1)
        if loss_adapt_err_0 - loss_adapt_err_1 < err:           
            break
        loss_adapt_err_0 = loss_adapt_err_1

    #--------------debug-----------------
    #loss_adapt_err = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
    #print('final quantile_err = ',loss_adapt_err)
    #------------Debug ------------------------------------------------
    return quantile_buck_1,quantile_index,values_increment


def value_replace_2(w, value_sequence):  # w模型形式 ,value_sequence 数组形式，
    w_rel = copy.deepcopy(w)
    m =0
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()): #顺序获取每一个值
            w_rel[i][index] = torch.tensor(value_sequence[m])
            m =m +1
    #print('\n m=',m,'len(value_sequence) = ',len(value_sequence))    
    #if m != len(value_sequence):
    #    print('Quantile Error',len(value_sequence),m)
    return w_rel


def value_replace_diff(w, value_sequence,values_glob):  # w模型形式 ,value_sequence 数组形式，
    w_rel = copy.deepcopy(w)
    m =0
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()): #顺序获取每一个值
            w_rel[i][index] = torch.tensor(value_sequence[m]+values_glob[m])
            m =m +1
    #print('\n m=',m,'len(value_sequence) = ',len(value_sequence))    
    #if m != len(value_sequence):
    #    print('Quantile Error',len(value_sequence),m)
    return w_rel




#  桶量化：
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组  （注：数组必须由有小到大排序）
#  sorted_id        : 整理好的数组索引
#  实现量化方式：    ways1: 根据桶的数目做量化，每个区间内量化的个数相等，但是边界值间隔不相等
#                                    quantile_level = len(value_sequence)/quantile_level  设置每个桶的数目
#                   ways2: 均匀量化， quantile_level = quantile_level
#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数
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



#  桶量化-opt：
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组  （注：数组必须由有小到大排序）
#  sorted_id        : 整理好的数组索引
#  实现量化方式：    ways1: 根据桶的数目做量化，每个区间内量化的个数相等，但是边界值间隔不相等
#                                    quantile_level = len(value_sequence)/quantile_level  设置每个桶的数目
#                   ways2: 均匀量化， quantile_level = quantile_level
#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数
def quantile_bucket_opt(quantile_level,value_sequence, sorted_id):
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
            code_value = fast_find_x_opt(c0,1)
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
            code_value =fast_find_x_opt(c1,1) # acc_level: 5
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



#  基于（bucket）桶间隔量化优化---ways = 2
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组【数组满足有小到大排序】  
#  sorted_id        : 整理好的数组索引

#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数

def quantile_bucket_opt_proposed(quantile_level,value_sequence_,sorted_id_,orig_values_increment,err = 0.001):
    quantile_buck_0 = np.zeros(quantile_level+1)
    quantile_buck_1 = np.zeros(quantile_level+1)
    quantile_index = np.zeros(len(value_sequence_))
    values_increment = np.zeros(len(value_sequence_))
    sorted_id =copy.deepcopy(sorted_id_)

    quantile_buck_ = np.zeros(quantile_level+1)
    #  step0 : 基于桶边界量化优化，得到初始的桶优化初始边界方案 （本步骤主要目的（标）： 主要得到 quantile_buck_ 的初始输入值）
    quantile_buck_, quantile_index, values_increment=  quantile_bucket(int(np.ceil(len(value_sequence_)/quantile_level)),value_sequence_, sorted_id)
    loss_adapt_err_0 =0
    #for i in range(len(values_increment)):
    #    loss_adapt_err_0 =loss_adapt_err_0 + abs(values_increment[i]-orig_values_increment[i])
    loss_adapt_err_0 = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
    #print('\n orig bucket opt quantile_err = ',loss_adapt_err_0)
    #print('quantile_buck_ = ',quantile_buck_)
    # quantile_buck_ 边界值
    #  step1 : quantile_border_opt 调整边界量化 
    #quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(quantile_level,value_sequence_,sorted_id,quantile_buck_)
    quantile_buck_1 =copy.deepcopy(quantile_buck_)
    X =1
    while X :
        for i in range(len(quantile_buck_1)-1,1,-1):
            if  quantile_buck_1[i-1]==quantile_buck_1[i] :
                #print('quantile_buck_1[i-1]',quantile_buck_1[i-1])
                #print('quantile_buck_1[i]',quantile_buck_1[i])
                #print('quantile_buck_1[i+1]',quantile_buck_1[i+1])
                quantile_buck_1[i] = (quantile_buck_1[i]+quantile_buck_1[i+1])/2
                quantile_buck_ = sorted([d for d in quantile_buck_1], reverse=False)  # value sequence  reverse=False 有小到大排列
                quantile_buck_1 = copy.deepcopy(quantile_buck_)
                # quantile_buck_  整理，是因为在优化的时候，可能已经不是一个由小到大的边界值了，所以需要重新整理排序一次
                #loss_adapt_err_0 = loss_adapt_err
                #quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=bucket_border_opt(quantile_level,value_sequence_,sorted_id,quantile_buck_)
                #quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(len(quantile_buck_)-1,orig_values_increment,sorted_id,quantile_buck_)       
                #loss_adapt_err =loss_adapt_err = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
                #print('\n temp quantile_err = ',loss_adapt_err)
                X =1
                break
            else :
                X =0
    
    
    quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(len(quantile_buck_)-1,orig_values_increment,sorted_id,quantile_buck_)
    #bucket_border_opt(quantile_level,value_sequence_,sorted_id,quantile_buck_)
    quantile_buck_ =copy.deepcopy(quantile_buck_1)
    #loss_adapt_err =0
    #for i in range(len(values_increment)):
    #    loss_adapt_err =loss_adapt_err + abs(values_increment[i]-orig_values_increment[i])
    loss_adapt_err = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
    #print('\n adapt one quantile_err = ',loss_adapt_err)
    X =1
    while X :
        for i in range(len(quantile_buck_1)-1):
            if  quantile_buck_1[i]==quantile_buck_1[i+1] :
                quantile_buck_1[i+1] = quantile_buck_1[i+2]/2
                print('quantile_buck_1[{}]={}',i,quantile_buck_1[i])
                print('quantile_buck_1[{}]={}',i+1,quantile_buck_1[i+1]) 
                quantile_buck_ = sorted([d for d in quantile_buck_1], reverse=False)  # value sequence  reverse=False 有小到大排列
                quantile_buck_1 = copy.deepcopy(quantile_buck_)
                # quantile_buck_  整理，是因为在优化的时候，可能已经不是一个由小到大的边界值了，所以需要重新整理排序一次
                #loss_adapt_err_0 = loss_adapt_err
                #quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=bucket_border_opt(quantile_level,value_sequence_,sorted_id,quantile_buck_)
                #quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(len(quantile_buck_)-1,orig_values_increment,sorted_id,quantile_buck_)       
                #loss_adapt_err =loss_adapt_err = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
                #print('\n temp quantile_err = ',loss_adapt_err)
                X =1
                break
            else :
                X =0
    #-----------------Debug --------------------------------------
    #print('quantile_buck_0 = ',quantile_buck_0)
    #print('quantile_buck_1 = ',quantile_buck_1)
    #print('quantile_buck_tmp = ',quantile_buck_tmp)
    #print('values_increment = ',values_increment) 
    #final_values_increment = copy.deepcopy(values_increment)
    X =1
    while X:
        if loss_adapt_err_0 > loss_adapt_err and (loss_adapt_err_0 - loss_adapt_err)>err :  #超出边界误差
        #if loss_adapt_err_0 > loss_adapt_err :  #超出边界误差
        #if quantile_boundary_err >= 0.01 : #超出边界误差
            loss_adapt_err_0 = loss_adapt_err
            #final_values_increment = copy.deepcopy(values_increment)
            quantile_buck_ =copy.deepcopy(quantile_buck_1)
            #quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=bucket_border_opt(quantile_level,value_sequence_,sorted_id,quantile_buck_)
            quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(len(quantile_buck_)-1,orig_values_increment,sorted_id,quantile_buck_)
            #print('\n quantile_buck_tmp =',quantile_buck_1)
            #print('\n values_increment =',values_increment)
            loss_adapt_err =loss_adapt_err = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
            #print('\n convergence quantile_err = ',loss_adapt_err)
            X =1
        else:
            X =0
    #------------    Debug ------------------------------------------------
    #print('\n final values_increment =',values_increment)
    #loss_adapt_err = 0
    #for i in range(len(final_values_increment)):
    #    loss_adapt_err =loss_adapt_err + abs(final_values_increment[i]- orig_values_increment[i])
    #loss_adapt_err =loss_adapt_err = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
    #print('\n     final loss_adapt_err = ',loss_adapt_err)
    #------------Debug ------------------------------------------------
    return quantile_buck_1,quantile_index,values_increment






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

"""
def count_sketch(quantile_level,orig_values_increment,d):
    
    hash_table = np.random.rand(d, quantile_level)
    
    d_list = [[j for j in range(len(orig_values_increment))] for i in range(d)]
    for i in range(d):
        random.shuffle(d_list[i])
    
    sign_list = 2*np.random.randint(2,size=[d,len(orig_values_increment)])-1
    
    
    # j-th hash(i): (hash(i)+d_list[j])%quantile_level
    
    for i in range(len(orig_values_increment)):
        for j in range(d):
            hash_table[j][(hash(i)+d_list[j][i])%quantile_level] += sign_list[j][i]*orig_values_increment[i]
          
    values_increment = copy.deepcopy(orig_values_increment)        
    for i in range(len(orig_values_increment)):
        value_d_list = []
        for j in range(d):
            value_d_list.append(sign_list[j][i]*hash_table[j][(hash(i)+d_list[j][i])%quantile_level])
        # print(value_d_list)
        
        #---median value---#
        value_d_list.sort()
        values_increment[i] = value_d_list[int(d/2)]
        
        #---minimum value---#
        # values_increment[i] = min(value_d_list)
    
    return values_increment
    """

def count_sketch_behind(quantile_level,orig_values_increment,d):
    
    hash_table = np.random.rand(d, quantile_level)
    d_list = np.random.choice(range(len(orig_values_increment)),d)
    sign_list = 2*np.random.randint(2,size=len(orig_values_increment))-1
    
    
    # j-th hash(i): (hash(i)+d_list[j])%quantile_level
    
    for i in range(len(orig_values_increment)):
        for j in range(d):
            #hash_table[j][(hash(i)+d_list[j])%quantile_level] += sign_list[i]*orig_values_increment[i]
            hash_table[j][(hash(i+d_list[j]))%quantile_level] += sign_list[i]*orig_values_increment[i]

    values_increment = copy.deepcopy(orig_values_increment)        
    for i in range(len(orig_values_increment)):
        value_d_list = []
        for j in range(d):
            value_d_list.append(sign_list[i]*hash_table[j][(hash(i+d_list[j]))%quantile_level])
        # print(value_d_list)
        
        #---median value---#
        value_d_list.sort()
        #values_increment[i] = value_d_list[int(d/2)]
        values_increment[i] = min(value_d_list)
        
        #---minimum value---#
       # values_increment[i] = min(value_d_list)
    
    return values_increment

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
    u,sigma,vt = la.svd(orig_matrix)
    #print(orig_values_increment)
    S = np.zeros([len(sigma),len(sigma)])
    R = int(np.ceil(len(orig_values_increment)*math.log2(quantile_level)/(32*(2*matrix_size+1))))
    #R = 4
    #print('math.log2(quantile_level)= ',math.log2(quantile_level))
    print( 'R = ',R)
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
    print('transmit_bit = ',transmist_bit,'bit')
    return values_increment

def clip_grad(l2_norm_clip, record):
    try:
        l2_norm = torch.norm(record)
    except:
        l2_norm = record.l2estimate()
    if l2_norm < l2_norm_clip:
        return record
    else:
        return record / float(torch.abs(torch.tensor(l2_norm) / l2_norm_clip))



def get_grad_vec(model):
    grad_vec = []
    with torch.no_grad():
        # flatten
        for p in model.parameters():
            if p.requires_grad:
                if p.grad is None:
                    grad_vec.append(torch.zeros_like(p.data.view(-1)))
                else:
                    grad_vec.append(p.grad.data.view(-1).float())
        # concat into a single vector
        grad_vec = torch.cat(grad_vec)
    return grad_vec


def get_param_vec(model):
    param_vec = []
    for p in model.parameters():
        if p.requires_grad:
            param_vec.append(p.data.view(-1).float())
    return torch.cat(param_vec)


def get_grad(model, args):
    weights = get_param_vec(model)
    grad_vec = get_grad_vec(model)
    if args.weight_decay != 0:
        grad_vec.add_(args.weight_decay / args.num_workers, weights)
    return grad_vec.to(args.device)

def count_sketch(model, args, grad, compute_grad=True):
    device = args.device
    # grad = get_grad(model, args)

    #print(grad)

    # compress the gradient if needed
    sketch = CSVec(d=args.grad_size, c=args.num_cols,
        r=args.num_rows, device=args.device,
        numBlocks=args.num_blocks)
    sketch.accumulateVec(grad)
    # gradient clipping
    if compute_grad and args.max_grad_norm is not None:
        sketch = clip_grad(args.max_grad_norm, sketch)
    hash_table = sketch.table
    unSketch_grad = sketch.unSketch(k=len(grad)) # 100% 全部还原，无top-k
    #print('grad = ',update.size())
    #print('grad = ',grad.size())
    #print('g = ',g)
    #print(unSketch_grad)
    return hash_table, unSketch_grad, grad


'''
**********************************************
Input must be a pytorch tensor
**********************************************
'''
def quantize(x,input_compress_settings={}):
    compress_settings={'n': 2}
    compress_settings.update(input_compress_settings)
    #assume that x is a torch tensor
    
    n=compress_settings['n']
    #print('n:{}'.format(n))
    x=x.float()
    x_norm=torch.norm(x,p=float('inf'))
    
    sgn_x=((x>0).float()-0.5)*2
    
    p=torch.div(torch.abs(x),x_norm)
    renormalize_p=torch.mul(p,n)
    floor_p=torch.floor(renormalize_p)
    compare=torch.rand_like(floor_p)
    final_p=renormalize_p-floor_p
    margin=(compare < final_p).float()
    xi=(floor_p+margin)/n
    
    
    
    Tilde_x=x_norm*sgn_x*xi
    
    return Tilde_x

if __name__ == '__main__': 
    # return the available GPU
    
    
    """
    av_GPU = torch.cuda.is_available()
    if  av_GPU == False:
        exit('No available GPU')
    print("\n GPU is running !!! \n ")
    """
    

    run_start_time = time.time()   
    # parse args
    args = args_parser()
    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')

    #------------
    args.device = "cpu"
    
    args.num_cols = 500
    args.num_rows = 500
    args.num_blocks = 1
    args.grad_size =0
    args.weight_decay =1
    args.max_grad_norm = 1e10
    args.num_workers =10
    #----------

    args.gpu = -1              # -1 (CPU only) or GPU = 0
    args.lr = 0.02             # 0.001 for cifar dataset
    args.model = 'mlp'         # 'mlp' or 'cnn' or ' MLP_triple' or 'MLP_regression'
    args.dataset = 'Adult'     # 'mnist' FashionMNIST cifar Adult
    

    args.num_users = 20        # numb of users cautious : the number must more than 10
    args.num_Chosenusers = 6 
    args.epochs = 100          # numb of global iters
    args.local_ep = 5          # numb of local iters
    args.num_experiments = 1


    args.num_items_train = 400 # numb of local data size # 
    args.num_items_test =  256
    args.local_bs = 64        # Local Batch size (1200 = full dataset)
                               # size of a user for mnist, 2000 for cifar)
    
                               
    args.set_epochs = [10]
    
    args.degree_noniid = 0
    args.set_degree_noniid = [0]
    args.strict_iid = True
    args.iid = True

    # -------设置加入动量-------
    args.set_momentum = True  # nedd_add_momentum = True   no_add_momentum = False  
    args.momentum_beta= 0.5   # set momentum_beta = 1-w
    momentum_beta = args.momentum_beta                          

    args.quantile_level = 2
    args.set_quantile_level = [2]



    args.ratio_train = [1,1,1,1,1]
    
    
    


    args.parameter_ways = 'diff_parameter'  #  'orig_parameter' , 'diff_parameter'
    
    

    #args.set_sketch_sche = ['orig','bucket_quantile','uniform_quantile','kmeans_opt','quantile_bucket_opt_proposed','uniform_quantile_opt_propose','QSGD','count_sketch',]
    #args.set_sketch_sche = ['uniform_quantile_opt_propose_5.0','uniform_quantile_opt_propose_2.0','uniform_quantile_opt_propose_1.0','uniform_quantile_opt','uniform_quantile_opt_propose_0.5','uniform_quantile_opt_propose_0.3','uniform_quantile_opt_propose']
    #args.set_sketch_sche = ['uniform_quantile_opt','uniform_quantile_opt_propose']
    #args.set_sketch_sche = ['quantile_bucket_opt_proposed','bucket_quantile','orig','uniform_quantile_opt_propose','SVD_Split']
    args.set_sketch_sche = ['kmeans_opt','uniform_quantile_opt_propose','quantile_bucket_opt_proposed','count_sketch','QSGD','bucket_quantile']
    
    #args.set_sketch_sche = ['bucket_quantile']
     
    args.sketch_sche = 'orig'
    
    
    
    args.delta = 0.0001
    
    args.set_variable = args.set_degree_noniid
    args.set_variable0 = copy.deepcopy(args.set_quantile_level)
    args.set_variable1 = copy.deepcopy(args.set_sketch_sche)

    hash_deepth =5

    print('--------------- information------------\n ways =',args.set_sketch_sche)
    print('          quantile_level = ',args.set_quantile_level)
    print('          learning rate  = ',args.lr)
    print('          num_experiments= ',args.num_experiments)
    print('          global_epochs  = ',args.epochs)
    print('          date_set       = ',args.dataset)
    print('          model          = ',args.model)
    print('          parameter_ways = ',args.parameter_ways)
    print('       set_degree_noniid = ',args.set_degree_noniid)
    print('                args.iid = ',args.iid)
    print('       args.set_momentum = ',args.set_momentum)
    if args.set_momentum == True :
        print('      args.momentum_beta = ',args.momentum_beta)
    print('----------------information------------\n')
    

    #加载数据集
    apply_transform1 = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    apply_transform2 = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    
# def main(args): 
    #####-Choose Variable-#####

    
    if not os.path.exists('./experiresult'):
        os.mkdir('./experiresult')



    # load dataset and split users
    dict_users,dict_users_train,dict_users_test = {},{},{}
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
            # sample users
        if args.iid:
            dict_users = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test) 
            dict_sever = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)
        else:
            dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_sever = mnist_noniid(args, dataset_test, args.num_users, args.num_items_test)
        
    elif args.dataset == 'cifar':
        dict_users_train, dict_sever = {},{}
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./dataset/cifar/', train=True, transform=transform, target_transform=None, download=True)
        dataset_test = datasets.CIFAR10('./dataset/cifar/', train=False, transform=transform, target_transform=None, download=True)
        
        #dataset_test = copy.deepcopy(dataset_train)
        #datasets.CIFAR100('./dataset/cifar100/', train=True, transform=None, target_transform=None, download=True)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, args.num_items_train)
            dict_sever = cifar_iid(dataset_test, args.num_users, args.num_items_test)
            """
            num_train = int(0.6*args.num_items_train)
            for idx in range(args.num_users):
                dict_users_train[idx] = set(list(dict_users[idx])[:num_train])
                dict_sever[idx] = set(list(dict_users[idx])[num_train:])
            """
        else:
            dict_users = cifar_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_sever = cifar_noniid(args, dataset_test, args.num_users, args.num_items_test)
            """
            dict_test = []
            num_train = int(0.6*args.num_items_train)
            for idx in range(args.num_users):
                dict_users_train[idx] = set(list(dict_users[idx])[:num_train])
                dict_sever[idx] = set(list(dict_users[idx])[num_train:])
            """
    elif args.dataset == 'FashionMNIST':  # 照着mnist 写 FashionMNIST
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
            dict_users = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            #print('dict_users = ',dict_users)
            # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test) 
            dict_sever = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)
        else:
            dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_sever = mnist_noniid(args, dataset_test, args.num_users, args.num_items_test)
    elif args.dataset == 'Adult':  #
        # 读取csv 格式文件
        dt = pd.read_csv("Bin_Adultall.csv")
        dt.head()
        # print(dt.head())   #打印标记文件头
        data_set = dt.values
        # print(data_set)
        X = data_set[:, :-1].astype(float)  # X输入的数据点（向量值），前n- 列都是输入X： 最后一列是输出： Y
        #print(X)
        Y = data_set[:, -1:].astype(int)  # Y 是输出输出结果，取出最后一列的值 [-1:]
        #print(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, shuffle=True)  # 设置测试集百分比
        X_train, Y_train = torch.FloatTensor(X_train), torch.LongTensor(Y_train)  # 训练集的输入： tensor浮点数 训练集的输出： tensor 整数
        X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)  # 测试集的输入： tensor浮点数 测试集的输出： tensor 整数


        #print('len X_train = ', len(X_train))
        # 转换韦pytorch 的张量
        #print('X_train = ', X_train)
        #print('X_test = ', X_test)
        #print('Y_train = ', Y_train)
        #print('Y_test = ', Y_test)

        Y_train = Y_train.reshape(len(Y_train),)
        Y_test = Y_test.reshape(len(Y_test), )
        dataset_train =[]
        dataset_test =[]
        for i in range(len(X_train)):
            dataset_train.append([X_train[i], Y_train[i]])
        for i in range(len(X_test)):
            dataset_test.append([X_test[i], Y_test[i]])
        
        if args.iid:
            dict_users = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            #print('dict_users = ',dict_users)
            # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test) 
            dict_sever = mnist_iid(args, dataset_test, args.num_users, args.num_items_test)
        else:
            dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_sever = mnist_noniid(args, dataset_test, args.num_users, args.num_items_test)

    
    #-----------------------------------------
    
   
    """
    for m in range(15):
        # sample users
        if args.iid:
            dict_users = mnist_iid(args,dataset_train, args.num_users, args.num_items_train)
            # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test) 
            dict_sever = mnist_iid(args,dataset_test, args.num_users, args.num_items_test)
        else:
            dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
            dict_sever = mnist_iid(args,dataset_test, args.num_users, args.num_items_test)          
        #-----------------------------------------
        with open('./Data_distribution/{}_dict_users_save_{}.pkl'.format(args.dataset,m),'wb') as f:
            pickle.dump(dict_users, f)  
        with open('./Data_distribution/{}_dict_server_save_{}.pkl'.format(args.dataset,m),'wb') as f:
            pickle.dump(dict_sever, f)
    """                                    
    

    

    #print('dict_users =',dict_users)
    #print('dict_sever =',dict_sever)
    #---------------------------
    img_size = dataset_train[0][0].shape    
    for v in range(len(args.set_variable)):
        final_train_loss = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        final_train_accuracy = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        final_test_loss = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        final_test_accuracy = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        final_quantile_err = [[0 for i in range(len(args.set_variable1))] for j in range(len(args.set_variable0))]
        args.degree_noniid = copy.deepcopy(args.set_variable[v])
        for s in range(len(args.set_variable0)):
            timeslot = time.time()
            test_acc_record, test_loss_record, quantile_err_record = [], [], []
            #-------debug---------
            percent_quantile_err_record,test_weight_sum_record =[],[]
            #-------debug---------
            for j in range(len(args.set_variable1)):
                args.sketch_sche = copy.deepcopy(args.set_variable1[j])   
                args.quantile_level = copy.deepcopy(args.set_variable0[s])
                quantile_level = args.quantile_level
                #print(args)
                loss_test, loss_train = [], []
                acc_test, acc_train = [], [] 
                com_cons = []
                fin_loss_test_list = []
                fin_acc_test_list = []   
                fin_quantile_err = []
                #---------------debug--------------
                fin_expr_weights_sum_avg_list,fin_expr_percent_quantile_err_avg_list =[],[]
                #---------------debug---------------
                for m in range(args.num_experiments):
                    # with open('./Data_distribution/{}_dict_users_save_{}.pkl'.format(args.dataset,m),'rb') as f:
                    #     dict_users = pickle.load(f)   
                    # with open('./Data_distribution/{}_dict_server_save_{}.pkl'.format(args.dataset,m),'rb') as f:
                    #     dict_sever = pickle.load(f) 
                    #print('dict_users =',dict_users)
                    #print('dict_sever =',dict_sever)
                    # build model
                    net_glob = None
                    if args.model == 'cnn' and args.dataset == 'mnist':
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            # net_glob = CNNMnist(args=args).cuda()
                            net_glob = CNN_test(args=args).cuda()
                        else:
                            net_glob = CNNMnist(args=args)
                            #torch.save(net_glob.state_dict(), './net_glob/cnn_mnist_glob.pth')
                    elif args.model == 'mlp' and args.dataset == 'mnist':
                        len_in = 1
                        for x in img_size:
                            len_in *= x
                        print('\n  mlp dim_in = ',len_in)
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).cuda()
                        else:
                            net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes)
                            #torch.save(net_glob.state_dict(), './net_glob/mlp_mnist_glob.pth')
                    elif args.model == 'cnn' and args.dataset == 'cifar':
                        if args.gpu != -1:
                            net_glob = CNNCifar(args).cuda()
                        else:
                            net_glob = CNNCifar(args)
                    elif args.model == 'mlp' and args.dataset == 'FashionMNIST':
                        len_in = 1
                        for x in img_size:
                            len_in *= x
                        print('\n  mlp dim_in = ',len_in)
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).cuda()
                        else:
                            net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes)
                            #torch.save(net_glob.state_dict(), './net_glob/mlp_FashionMNIST_glob.pth')
                    elif args.model == 'cnn' and args.dataset == 'FashionMNIST':
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            # net_glob = CNNMnist(args=args).cuda()
                            net_glob = CNNFashionMnist(args=args).cuda()
                        else:
                            net_glob = CNNFashionMnist(args=args)
                            #torch.save(net_glob.state_dict(), './net_glob/cnn_FashionMNIST_glob.pth')
                    elif args.model == 'mlp' and args.dataset == 'Adult':
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            # net_glob = CNNMnist(args=args).cuda()
                            net_glob = MLP3(args=args).cuda()
                        else:
                            net_glob = MLP3(args=args)
                            #torch.save(net_glob.state_dict(), './net_glob/cnn_FashionMNIST_glob.pth')
                    else:
                        exit('Error: unrecognized model')
                    #print("Nerual Net:",net_glob)
                

                    net_glob.train()  #Train() does not change the weight values
                    # copy weights
                    # net_glob.load_state_dict(torch.load('./net_glob/{}_{}_glob.pth'.format(args.model, args.dataset)))
                    w_glob = net_glob.state_dict()
                    values_glob =[]
                    for i in w_glob.keys():
                            values_glob += list(w_glob[i].view(-1).cpu().numpy()) 
                    print('values_glob = ',values_glob[0:10])
                    if args.set_momentum == True :
                        mountum_t_0 = [0 for i in range(len(values_glob))]
                    """"       
                    w_size = 0
                    w_size_all = 0
                    for k in w_glob.keys():
                        size = w_glob[k].size()
                        if(len(size)==1):
                            nelements = size[0]
                        else:
                            nelements = size[0] * size[1]
                        w_size += nelements*4
                        w_size_all += nelements
                    """
                        # print("Size ", k, ": ",nelements*4)
                    #print("\n Weight Size:", w_size, " bytes")
                    #print("\n Weight & Grad Size:", w_size*2, " bytes")
                    #print("\n Each user Training size:", 784* 8/8* args.local_bs, " bytes")
                    #print("\n Total Training size:", 784 * 8 / 8 * 60000, " bytes")
                    # training
                    one_expr_train_loss_avg_list, one_expr_train_acc_avg_list, one_expr_test_loss_avg_list, one_expr_test_acc_avg_list, one_expr_quantile_err_avg_list = [], [], [], [], []
                    one_expr_weights_sum_avg_list,one_expr_percent_quantile_err_avg_list =[],[]
                    quantile_err = 0
                    ###  FedAvg Aglorithm  ###    
                    #w_err = diff_values(w_glob,w_glob)
                                      
                    for iter in range(args.epochs):
                        print('\n','*' * 20,f'Experiment: {m}/{args.num_experiments}, Epoch: {iter}/{args.epochs}','*' * 20)
                        print('                       quantile_level = ',quantile_level)
                        time_start = time.time() 
                        if  args.num_Chosenusers < args.num_users:
                            #随机用户训练
                            chosenUsers = random.sample(range(args.num_users),args.num_Chosenusers)
                            chosenUsers.sort()
                            #固定用户进行训练
                            #chosenUsers = range(args.num_Chosenusers)
                        else:
                            chosenUsers = range(args.num_users)
                        print("\nChosen users:", chosenUsers)
                        print('\nsketch ways = ',args.sketch_sche)                
                        w_locals, w_locals_1ep, train_loss_locals_list, train_acc_locals_list = [], [], [], []
                        quantile_err_list = []
                        #-------------debug----------------
                        weights_sum_list,percent_quantile_err_list = [],[]
                        #-------------debug----------------
                        values_glob = []
                        for i in w_glob.keys():
                            values_glob += list(w_glob[i].view(-1).cpu().numpy()) 
                        
                        #print('values_glob = ',values_glob[0:10])
                        values_increment_list = [] # 用户模型参数list
                        for idx in range(len(chosenUsers)):
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[chosenUsers[idx]], tb=summary)
                            w_1st_ep, w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                            #注 ： 这里的acc  和  loss 都是训练得到的，所以每次要在后面求一个 append(acc) append(loss)
                            train_loss_locals_list.append(copy.deepcopy(loss))
                            # print("User ", chosenUsers[idx], " Acc:", acc, " Loss:", loss)
                            train_acc_locals_list.append(copy.deepcopy(acc))

                            
                            w_increas = copy.deepcopy(w)                                
                            values_increment = []                            
                            if args.parameter_ways == 'orig_parameter' :   #  'orig_parameter' diff_parameter
                                for i in w_increas.keys():
                                    values_increment += list(w_increas[i].view(-1).cpu().numpy())
                            elif args.parameter_ways == 'diff_parameter' :  #  'orig_parameter' diff_parameter
                                for i in w_increas.keys():
                                    values_increment += list(w_increas[i].view(-1).cpu().numpy()-w_glob[i].view(-1).cpu().numpy())

                            #print('Θ nums = ',len(values_increment))
                            

                            #-------------------debug---------------------------
                            weights_sum = sum(abs(np.array(values_increment)))
                            #print('\n min_value = ',min(values_increment))
                            #print('\n max_value = ',max(values_increment))
                            #--------------debug-----------------------------
                            #print('\n')
                            if args.sketch_sche == 'bucket_quantile':
                                #print('ways = bucket_quantile')
                                orig_values_increment = copy.deepcopy(values_increment)   
                                value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence
                                sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False)  
                                #参数数目均匀量化
                                quantile_buck, quantile_index, values_increment = quantile_bucket(int(np.ceil(len(value_sequence)/quantile_level)),value_sequence,sorted_id)
                                #print('len(quantile_buck) =',len(quantile_buck))
                                #if len(quantile_buck)!=(quantile_level+1): # 判断量化后的边界值数目和量化数目是否相等
                                #    print('Quantile error:',len(quantile_buck),quantile_level)
                            elif args.sketch_sche == 'bucket_quantile_opt':
                                #print('ways = bucket_quantile_opt')
                                orig_values_increment = copy.deepcopy(values_increment)
                                value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence
                                sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False)  
                                #参数数目均匀量化
                                quantile_buck, quantile_index, values_increment = quantile_bucket_opt(int(np.ceil(len(value_sequence)/quantile_level)),value_sequence,sorted_id)
                                #print('len(quantile_buck) =',len(quantile_buck))
                                #if len(quantile_buck)!=(quantile_level+1): # 判断量化后的边界值数目和量化数目是否相等
                                #    print('Quantile error:',len(quantile_buck),quantile_level)
                            elif args.sketch_sche == 'uniform_quantile':
                                #print('ways = uniform_quantile')
                                orig_values_increment = copy.deepcopy(values_increment)
                                value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence  reverse=False 有小到大排列
                                sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False) #reverse=False 有小到大排列
                                #区间均匀量化
                                quantile_buck, quantile_index, values_increment = quantile_uniform(quantile_level,value_sequence,sorted_id) #values_increment 的量化传输的差值
                                #print('len(quantile_buck) =',len(quantile_buck))
                            elif args.sketch_sche == 'uniform_quantile_opt':
                                #print('ways = uniform_quantile_opt')
                                orig_values_increment = copy.deepcopy(values_increment)
                                value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence  reverse=False 有小到大排列
                                sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False) #reverse=False 有小到大排列
                                #区间均匀量化
                                quantile_buck, quantile_index, values_increment = quantile_uniform_opt(quantile_level,value_sequence,sorted_id,orig_values_increment) #values_increment 的量化传输的差值                    
                                #print('len(quantile_buck) =',len(quantile_buck))
                            elif args.sketch_sche == 'uniform_quantile_opt_propose': 
                                #print('ways = uniform_quantile_opt_propose')
                                orig_values_increment = copy.deepcopy(values_increment)
                                value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence  reverse=False 有小到大排列
                                sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False) #reverse=False 有小到大排列
                                #区间均匀量化-边界优化
                                quantile_buck,quantile_index,values_increment = quantile_uniform_opt_proposed(quantile_level,value_sequence,sorted_id,orig_values_increment,err = 0.3)
                                #print('len(quantile_buck) =',len(quantile_buck))
                            elif args.sketch_sche == 'quantile_bucket_opt_proposed':
                                #print('ways = quantile_bucket_opt_proposed')
                                orig_values_increment = copy.deepcopy(values_increment)
                                value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence  reverse=False 有小到大排列
                                sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False) #reverse=False 有小到大排列
                                #区间桶量化-边界优化
                                quantile_buck,quantile_index,values_increment = quantile_bucket_opt_proposed(quantile_level,value_sequence,sorted_id,orig_values_increment,err = 0.3)   
                                #print('len(quantile_buck) =',len(quantile_buck))
                                #print('max quantile_index =',max(quantile_index))
                            elif args.sketch_sche == 'orig':
                                #print('ways = orig')
                                orig_values_increment = copy.deepcopy(values_increment) 
                            elif args.sketch_sche == 'kmeans_opt':
                                #print('ways = kmeans_opt')
                                orig_values_increment = copy.deepcopy(values_increment) 
                                quantile_buck,quantile_index,values_increment = kmeans_opt(quantile_level,orig_values_increment)
                                #print('len(quantile_buck) =',len(quantile_buck))
                            # --------------------test-----------------------------
                            elif args.sketch_sche == 'QSGD':
                                #print('ways = QSGD')
                                orig_values_increment = copy.deepcopy(values_increment) 
                                quantile_buck,quantile_index,values_increment = QSGD_(quantile_level,orig_values_increment)
                            elif args.sketch_sche == 'SVD_Split':
                                #print('ways = SVD_Split')
                                orig_values_increment = copy.deepcopy(values_increment)
                                values_increment = SVD_Split(quantile_level,orig_values_increment)
                            elif args.sketch_sche == 'count_sketch':
                                #print('ways = count_sketch')
                                orig_values_increment = copy.deepcopy(values_increment)
                                grad = torch.from_numpy(np.array(values_increment))
                                # num_rows = 128
                                # num_cols = 200
                                #g =sketch_new(w_increas,orig_values_increment,num_rows,num_cols, compute_grad=True)
                                args.grad_size =  len(orig_values_increment)
                                hash_table ,grad_unsketched, grad = count_sketch(net_glob, args, grad, compute_grad=True)
                                values_increment = grad_unsketched.numpy()
                            elif args.sketch_sche == 'QSGD_new':
                                #print('ways = QSGD_new')
                                orig_values_increment = copy.deepcopy(values_increment) 
                                #x=value_replace_2(w_increas, values_increment) # w模型形式 ,value_sequence 数组形式
                                x = torch.Tensor(values_increment) #numpy转换torch
                                values_increment_ = quantize(x,input_compress_settings={})
                                values_increment = values_increment_.numpy()
                                #print('values_increment_ max value = ',max(values_increment))
                                #print('values_increment [:]  = ',values_increment[1:20])
                                
                            w_rel = value_replace_2(w_increas, values_increment) # w模型形式 ,value_sequence 数组形式
                            #w_rel = value_replace(w_increas,quantile_buck,values_increment) # values_increment 数组形式， w_rel 模型形式
                            """
                            if args.sketch_sche != 'orig':
                                w_rel = value_replace_2(w_increas, values_increment) # w模型形式 ,value_sequence 数组形式
                            else:
                                w_rel = copy.deepcopy(w_increas)
                            """
                            # step6: 本地端接收在w_rel ，通过差值更新w_0
                            """
                            # 量化误差
                            quantile_err_ = diff_values(w_rel,w)        # step1: 模型差值 w: 实际跟新 w_rel ： 量化 更新（对差值进行量化）
                            quantile_err = weights_values(quantile_err_)# 量化误差模型绝对值求和
                            print("update_quantile_err = ",quantile_err)
                            """
                            #quantile_err += quantile_err
                            quantile_err = sum(abs(np.array(values_increment)-np.array(orig_values_increment)))
                            quantile_err_list.append(quantile_err)
                            #print("Update_quantile_err = ",quantile_err)

                            #-----------------debug---------------------
                            percent_quantile_err = 100*quantile_err/weights_sum
                            weights_sum_list.append(weights_sum)
                            percent_quantile_err_list.append(percent_quantile_err)
                            #print('\n weights_sum = ',weights_sum)
                            #print('\n percent_quantile_err = ',percent_quantile_err,'%')
                            #-----------------debug-------------------------

                            # step1:用户模型参数保存列表
                            values_increment_list.append(values_increment)
                            #print(len(values_increment_list))
                        # step2: 求平均
                        values_increment_list = np.sum(values_increment_list,axis = 0)
                        #print(values_increment_list[1])
                        values_increment_list = values_increment_list/args.num_Chosenusers
                        #添加动量操作
                        if args.set_momentum == True : 
                            values_increment_list = momentum_beta* np.array(mountum_t_0) + (1-momentum_beta)* np.array(values_increment_list)
                            mountum_t_0 =values_increment_list

                        #print(values_increment_list[1])
                        #w_rel = value_replace(w_increas,quantile_buck,values_increment) # values_increment 数组形式， w_rel 模型形式
                        #w_rel = value_replace(w_increas,quantile_buck,values_increment) # values_increment 数组形式， w_rel 模型形式

                        #w_glob = value_replace_2(w_increas, values_increment_list) # w模型形式 ,value_sequence 数组形式
                        if args.parameter_ways == 'orig_parameter' :   #  'orig_parameter' diff_parameter
                            #print('ways = orig_parameter')
                            w_glob = value_replace_2(w_increas, values_increment_list) # w模型形式 ,value_sequence 数组形式
                        elif args.parameter_ways == 'diff_parameter' :  #  'orig_parameter' diff_parameter
                            #print('ways = diff_parameter')
                            w_glob = value_replace_diff(w_increas, values_increment_list,values_glob) # w模型形式 ,value_sequence 数组形式

                        # copy weight to net_glob
                        net_glob.load_state_dict(w_glob)
                        # global test
                        test_loss_locals_list, test_acc_locals_list = [], []   # 存取的是每个用户的 测试的 list_acc 和 list_loss
                        net_glob.eval()
                        for c in range(args.num_users):
                            net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_sever[idx], tb=summary)
                            acc, loss = net_local.test(net=net_glob)                    
                            # acc, loss = net_local.test_gen(net=net_glob, idxs=dict_users[c], dataset=dataset_test)
                            test_acc_locals_list.append(acc)  #参数对所有用户的数据测试
                            test_loss_locals_list.append(loss) 
                            #print('acc =',acc,'loss =',loss)
                        # print("\nEpoch: {}, Global test loss {}, Global test acc: {:.2f}%".\
                        #      format(iter, sum(list_loss) / len(list_loss),100. * sum(list_acc) / len(list_acc)))
                        

                        # 所有用户 【本次】迭代的 train_loss  和 train_acc 求平均
                        train_loss_avg = sum(train_loss_locals_list) / len(train_loss_locals_list)
                        train_acc_avg = sum(train_acc_locals_list) / len(train_acc_locals_list)
                        
                        # 所有用户 【本次】迭代的  test_loss  和 test_acc 求平均 这个是后续需要画图的
                        test_loss_avg = sum(test_loss_locals_list) / len(test_loss_locals_list)
                        test_acc_avg = sum(test_acc_locals_list) / len(test_acc_locals_list)
                        quantile_err_avg = sum(quantile_err_list) / len(quantile_err_list)
                        #----------------debug---------------------------
                        weights_sum_avg = sum(weights_sum_list)/len(weights_sum_list)
                        percent_quantile_err_avg = sum(percent_quantile_err_list)/len(percent_quantile_err_list)
                        #---------------debug--------------------------------

                        #单次(1 次)实验的数据列表，由每次迭代出来的数据求平均进行 .appen()操作
                        one_expr_train_loss_avg_list.append(train_loss_avg)
                        one_expr_train_acc_avg_list.append(train_acc_avg) 

                        # test data 需要画图
                        one_expr_test_loss_avg_list.append(test_loss_avg) #本次迭代的平均数据，添加到数据集合中
                        one_expr_test_acc_avg_list.append(test_acc_avg)
                        one_expr_quantile_err_avg_list.append(quantile_err_avg)
                        #---------debug -----------------
                        one_expr_weights_sum_avg_list.append(weights_sum_avg)
                        one_expr_percent_quantile_err_avg_list.append(percent_quantile_err_avg)
                        #--------debug--------------------
                        time_end = time.time()
                        #本次所有用户迭代的时候，数据才会出来的结果
                        #所以这里应该是: trian loss 求平均 trian acc 求平均
                        #               test loss 求平均 test acc 求平均
                        print('\n Run time = {:.2f}S'.format(time_end-time_start))
                        
                        #print('\nAverage quantile err :',quantile_err_avg)
                        print("\nTrain loss: {}, Train acc: {}".format(train_loss_avg, train_acc_avg))
                        print("\nTest  loss: {}, Test acc:  {}".format(test_loss_avg, test_acc_avg))
                        
                        
                        # if (iter+1)%20==0:
                        #     torch.save(net_glob.state_dict(),'./Train_model/glob_model_{}epochs.pth'.format(iter))
                    print('\n-----------ways= ',args.sketch_sche,' expr ',m,' data record :---------')
                    print('expr ',m,' test_loss_avg_list    =',one_expr_test_loss_avg_list)
                    print('expr ',m,' test_acc_avg_list     =',one_expr_test_acc_avg_list)
                    print('expr ',m,' quantile_err_avg_list =',one_expr_quantile_err_avg_list)
                    
                    #------------------debug------------------------------
                    print('expr ',m,' one_expr_weights_sum_avg_list =',one_expr_weights_sum_avg_list)
                    print('expr ',m,' one_expr_percent_quantile_err_avg_list =',one_expr_percent_quantile_err_avg_list)
                    #-----------------debug-------------------------------

                    #添加每次实验的数据列表
                    fin_loss_test_list.append(one_expr_test_loss_avg_list)
                    fin_acc_test_list.append(one_expr_test_acc_avg_list)
                    fin_quantile_err.append(one_expr_quantile_err_avg_list)
                    print('\n-------------------------------------------------')
                    print('list expr 0~',m,'fin_loss_test_list =',fin_loss_test_list)
                    print('list expr 0~',m,'fin_acc_test_list  =',fin_acc_test_list)
                    print('list expr 0~',m,'fin_quantile_err   =',fin_quantile_err)

                    #------------------debug---------------------
                    fin_expr_weights_sum_avg_list.append(one_expr_weights_sum_avg_list)
                    fin_expr_percent_quantile_err_avg_list.append(one_expr_percent_quantile_err_avg_list)
                    print('list expr 0~',m,'fin_expr_weights_sum_avg_list =',fin_expr_weights_sum_avg_list)
                    print('list expr 0~',m,'fin_expr_percent_quantile_err_avg_list  =',fin_expr_percent_quantile_err_avg_list)
                    #------------------debug---------------------

                #每一种量化方案所有实验的所有数据平均
                tmp_fin_mean_loss_test = np.array(fin_loss_test_list).mean(axis=0) 
                tmp_fin_mean_acc_test  = np.array(fin_acc_test_list).mean(axis=0) 
                tmp_fin_mean_quantile_test = np.array(fin_quantile_err).mean(axis=0)                                   
                print('-------------------ways =',args.sketch_sche,' all_expr average record : -----------------')
                print('tmp_fin_average_loss_test    =',tmp_fin_mean_loss_test)
                print('tmp_fin_average_acc_test     =',tmp_fin_mean_acc_test)
                print('tmp_fin_average_quantile_test     =',tmp_fin_mean_quantile_test)    
                
                #-------------------------------debug-------------------
                tmp_fin_mean_weights_sum_test  = np.array(fin_expr_weights_sum_avg_list).mean(axis=0) 
                tmp_fin_mean_percent_quantile_err_test = np.array(fin_expr_percent_quantile_err_avg_list).mean(axis=0)                                   
                print('tmp_fin_average_weights_sum_test    =',tmp_fin_mean_weights_sum_test)
                print('tmp_fin_average_percent_quantile_err_test     =',tmp_fin_mean_percent_quantile_err_test)
                #-------------------------------debug------------------

                # 记录所有方式的实验次数的平均结果的所有列表
                test_loss_record.append(tmp_fin_mean_loss_test.tolist())
                test_acc_record.append(tmp_fin_mean_acc_test.tolist())
                quantile_err_record.append(tmp_fin_mean_quantile_test.tolist())
                #----------------debug --------------------
                test_weight_sum_record.append(tmp_fin_mean_weights_sum_test.tolist())
                percent_quantile_err_record.append(tmp_fin_mean_percent_quantile_err_test.tolist())
                #----------------debug --------------------
                print('\n\n------------------- all ways record : -----------------')
                print('test_loss_record:', test_loss_record)
                print('test_acc_record:', test_acc_record)
                print('quantile_err_record:',quantile_err_record)
                #-----------debug----------------
                print('test_weight_sum_record:', test_weight_sum_record)
                print('percent_quantile_err_record:',percent_quantile_err_record)
                #-----------debug-----------------
                print('\n\n\n-----------------------------------------------------s------------------------------')
                
                
                x = [i for i in range(args.epochs)] 
                # plot loss curve
                labels = "quantile level-{}".format(args.quantile_level)                           
                plt.figure(m)
                plt.subplot(231)
                #plt.plot(range(len(test_loss_record)), test_loss_record,label=labels)
                for i in range(len(test_loss_record)):
                    labels = "{}".format(args.set_sketch_sche[i]) 
                    plt.plot(x,test_loss_record[i],label=labels)
                plt.ylabel('test loss')
                plt.xlabel('epoches')
                plt.grid(linestyle = "--")
                # plt.legend(loc="upper left")
                plt.figure(m)
                plt.subplot(232)
                plt.title('lr = {}'.format(args.lr),fontsize=20) #标签
                #plt.plot(range(len(test_acc_record)), test_acc_record, label=labels)
                for i in range(len(test_acc_record)):
                    labels = "{}".format(args.set_sketch_sche[i]) 
                    plt.plot(x,test_acc_record[i],label=labels)
                    #plt.plot(x,test_acc_record[i],label='{}'.format(args.set_sketch_sche[i]))
                    #plt.plot(range(len(test_acc_record[0])), test_acc_record[i],label='orig')
                plt.ylabel('test accuracy')
                plt.xlabel('epoches')
                plt.grid(linestyle = "--")
                # plt.legend(loc="upper left")
                plt.figure(m)
                plt.subplot(233)
                #plt.plot(range(len(quantile_err_record)), quantile_err_record, label=labels)
                for i in range(len(quantile_err_record)):
                    labels = "{}".format(args.set_sketch_sche[i]) 
                    plt.plot(x,quantile_err_record[i],label=labels)
                plt.ylabel('quantile err')
                plt.xlabel('epoches')
                plt.grid(linestyle = "--")
                # plt.legend(loc="upper left")
                

                plt.figure(m)
                plt.subplot(234)
                #plt.plot(range(len(quantile_err_record)), quantile_err_record, label=labels)
                for i in range(len(test_weight_sum_record)):
                    labels = "{}".format(args.set_sketch_sche[i]) 
                    plt.plot(x,test_weight_sum_record[i],label=labels)
                plt.ylabel('weights_sum')
                plt.xlabel('epoches')
                plt.grid(linestyle = "--")

                plt.figure(m)
                plt.subplot(235)
                #plt.plot(range(len(quantile_err_record)), quantile_err_record, label=labels)
                for i in range(len(percent_quantile_err_record)):
                    labels = "{}".format(args.set_sketch_sche[i]) 
                    plt.plot(x,percent_quantile_err_record[i],label=labels)
                plt.ylabel('percent_quantile_err_list')
                plt.xlabel('epoches')
                plt.grid(linestyle = "--")

                plt.savefig('./experiresult/{}-quantile_level-{}training_record-{}-{}.pdf'.\
                            format(args.sketch_sche,quantile_level,iter+1,timeslot))
                
                

        # timeslot = int(time.time())
        # data_test_loss = pd.DataFrame(index = args.set_variable0, columns = args.set_variable1, data = final_train_loss)
        # data_test_loss.to_csv('./experiresult/'+'train_loss_{}_{}.csv'.format(args.set_variable[v],timeslot))
        # data_test_loss = pd.DataFrame(index = args.set_variable0, columns = args.set_variable1, data = final_test_loss)
        # data_test_loss.to_csv('./experiresult/'+'test_loss_{}_{}.csv'.format(args.set_variable[v],timeslot))
        # data_test_acc = pd.DataFrame(index = args.set_variable0, columns = args.set_variable1, data = final_train_accuracy)
        # data_test_acc.to_csv('./experiresult/'+'train_acc_{}_{}.csv'.format(args.set_variable[v],timeslot))
        # data_test_acc = pd.DataFrame(index = args.set_variable0, columns = args.set_variable1, data = final_test_accuracy)
        # data_test_acc.to_csv('./experiresult/'+'test_acc_{}_{}.csv'.format(args.set_variable[v],timeslot))
        # data_test_acc = pd.DataFrame(index = args.set_variable0, columns = args.set_variable1, data = final_quantile_err)
        # data_test_acc.to_csv('./experiresult/'+'commun_bits_{}_{}.csv'.format(args.set_variable[v],timeslot))

        plt.close()
        print('\n\n expriment finished')
        run_end_time = time.time()
        print('\n Experiment Run time = {} h'.format((run_end_time-run_start_time)/3600)) 


# if __name__ == '__main__':    
#     # return the available GPU
#     av_GPU = torch.cuda.is_available()
#     if  av_GPU == False:
#         exit('No available GPU')
#     # parse args
#     args = args_parser()
#     # define paths
#     path_project = os.path.abspath('..')

#     summary = SummaryWriter('local')
#     args.gpu = 0                # -1 (CPU only) or GPU = 0
#     args.lr = 0.02              # 0.001 for cifar dataset
#     args.model = 'mlp'          # 'mlp' or 'cnn'
#     args.dataset = 'mnist'      # 'mnist' or cifar
#     args.num_users = 5          # numb of users
#     args.num_Chosenusers = 5    # 
#     args.epochs = 10            # numb of global iters
#     args.local_ep = 10          # numb of local iters
#     args.num_items_train = 600  # numb of local data size # 
#     args.num_items_test =  512  # 
#     args.local_bs = 800         # Local Batch size (1200 = full dataset)
#                                 # size of a user for mnist, 2000 for cifar)
#     args.degree_noniid = 1      #
                               
#     args.set_epochs = [20]    
#     args.set_num_Chosenusers = [20]
    
#     args.set_degree_noniid = [0]
#     args.strict_iid = True
#     args.ratio_train = [0.5,0.75,1,1.25,1.5]
#     args.num_experiments = 1
    
#     args.iid = False   
#     main(args)
#     
