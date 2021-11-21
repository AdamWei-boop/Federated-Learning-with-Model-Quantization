#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
"""
Created on Tue May 11 21:22:29 2021

@author: weikang

"""
import torch
import os
import time
import csv
import copy
import random
import numpy as np
from numpy import linalg as la
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from FedNets import MLP_triple,MLP_triple_SVD
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# 带入不同的X的值，求得在X下不同的loss 总和
def loss_sum(X,value_sequence):
    loss = 0
    for i in range(len(value_sequence)):
        loss =loss + abs(value_sequence[i] -X)
    return loss

# 得到一个数组，获取loss 值组成的数组
def loss_array(value_sequence,delta):
    loss_list = []
    x_list =[]
    for i in range(len(value_sequence)-1):
        for j in range(delta):
            x = value_sequence[i]+j*(value_sequence[i+1]-value_sequence[i])/delta
            x_loss =loss_sum(x,value_sequence)
            loss_list.append(x_loss)
            x_list.append(x)
    return loss_list ,x_list

def find_min_code(value_sequence_,acc_level):
    value_sequence = sorted([d for d in value_sequence_], reverse=False)  # value sequence_  step0 : 排序
    loss_list_ ,x_list_=loss_array(value_sequence,acc_level)   # step1:求出所有x_list_ 和 loss_list_
    loss_list = sorted([d for d in loss_list_], reverse=False)  # step2 : 对 loss_list_排序(可以省略)
    loss_list_id = sorted(range(len(loss_list_)), key=lambda i: loss_list_[i], reverse=False)  # step3: loss_list 的索引排序 ，并且得到loss_list_ 最小索引
    x_code = x_list_[loss_list_id[0]]  # step4 ： 找到最小的x_code，并输出（返回）该值
    #--------------test -------------------
    """
    plt.scatter(x_list_,loss_list_,color='b')
    plt.scatter(x_list_[loss_list_id[0]],loss_list[0],color='r')
    plt.figure(2)
    #plt.subplot(121)
    # plot test_acc curve
    labels = "loss 拟合曲线"                  
    plt.plot(x_list_, loss_list_,label=labels)
    plt.show()
    print('\n x = ',x_code)
    """
    #----------test -------------------------
    return x_code


# 快速求解 最优 x的值
# 输入： 
# value_sequence  ：任意一维数组
# acc_level ： 二次搜索精度
# 输出：
# x_min  ： loss 最小时，对应的x的值
def fast_find_x_opt_1(value_sequence,acc_level):
    if len(value_sequence) == 1 :
        return value_sequence[0]
    loss_list_1_ = []
    loss_list_2_ = []
    x1_list =[]
    x2_list =[]
    arr_length =len(value_sequence)
    for i in range(arr_length):  # step1： 1级求解loss 列表
        x = value_sequence[i]
        #print('\n depp1: x= ',x)
        x1_loss =loss_sum(x,value_sequence)
        loss_list_1_.append(x1_loss)
        x1_list.append(x)

    #loss_list_1 = sorted([d for d in loss_list_1_], reverse=False)  # step2 : 对 loss_list_1_ 排序(可以省略)
    loss_list_1_id = sorted(range(len(loss_list_1_)), key=lambda i: loss_list_1_[i], reverse=False)  # step3: loss_list_1_ 的索引排序 ，并且得到loss_list_ 最小索引
    
    x_min = x1_list[loss_list_1_id[0]]      # step4 ： 找到第一级 loss最小的x_min
    x_sub_min = x1_list[loss_list_1_id[1]]  # step5 ： 找到第一级第二小loss 的x_sub_min
    m =1
    while x_sub_min == x_min :
        #print('\n x_sub_min == x_min')
        m = m+1
        if arr_length == m :
            return x_min
        x_sub_min = x1_list[loss_list_1_id[m]]  # step5 ： 找到第一级第二小loss 的x_sub_min 

    #print('\n x_min = ',x_min,' x_sub_min = ',x_sub_min)
    for j in range(acc_level):              # step6 ： 2级求解loss 列表
        x = x_min+j*(x_sub_min-x_min)/acc_level
        #print('\n depp2: x= ',x)
        x2_loss =loss_sum(x,value_sequence)
        loss_list_2_.append(x2_loss)
        x2_list.append(x)

    loss_list_2 = sorted([d for d in loss_list_2_], reverse=False)  # step7 : 对 loss_list_2排序(可以省略)
    loss_list_2_id = sorted(range(len(loss_list_2)), key=lambda i: loss_list_2_[i], reverse=False)  # step8: loss_list_2 的索引排序 ，并且得到loss_list_ 最小索引
    
    x_min = x2_list[loss_list_2_id[0]]      # step9 ： 找到第二级最小的x_min
    #---------------Debug------------------------------
    """
    plt.scatter(x1_list,loss_list_1_,color='b')
    plt.scatter(x2_list,loss_list_2_,color='g')
    plt.scatter(x2_list[loss_list_2_id[0]],loss_list_2[0],color='r')
    plt.show()
    """
    #---------------Debug------------------------------
    return x_min 
    
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
        if abs(x1-x2) < 0.000001 :
            X = 0
    #---------debug ---------
    #print('itertimes =',itertimes)
    #print('opt_value =',opt_value,'opt_loss=',opt_loss)
    #plt.scatter(opt_value,opt_loss,color='r')
    #plt.show()
    #--------debug ----------
    return opt_value


# 一维多边形费马点求解
# 输入： Xi  : 一维数组的所有点
# 输出： x   ：拟合的最优点
def loss_min(Xi):
    #Xi = 1000*Xi
    n=len(Xi)
    x=sum(Xi)/n
    if(x == 0):
        print('\n x= ',x)
        return x
    #print('\n Xi= ',Xi)
    loss = 0
    while True:
        xfenzi=0
        xfenmu=0
        for i in range(n):
            g=math.sqrt((x-Xi[i])**2)
            if(g != 0):
                xfenzi=xfenzi+Xi[i]/g
                xfenmu=xfenmu+1/g
        xn=xfenzi/xfenmu
        #print('\n x = ',x,)
        if abs(xn-x)<0.00001 :
            break
        else:
            x=xn
    #----- tset loss -------
    x=xn
    y =np.zeros(n)
    #plt.scatter(Xi,y,color='r')
    for i in range(len(Xi)):
        loss += abs(x-Xi[i])
    #print('\n x = ',x, ' loss = ',loss)
    #plt.scatter(x,0,color='b')
    #plt.show()
    return x


#------------------拟合值 找code_index------------
def find_code_index(A_,delta,tolerate_err):
    #A = A_.sort()
    A_min = A_.min()
    A_max = A_.max()
    loss_up = 0
    loss_down =0
    #code_index = (A_min + A_max)/2
    code_index = A_min
    print('\n A_min = ', A_min ,'A_max =',A_max)
    print('\n orig code_index = ',code_index)
    while True:
        for x, val_ in enumerate(A_):
            for y , val in enumerate(val_):
                #print(x,y, val)
                dif = A_[x,y] - code_index
                if(dif >=0 ):
                    loss_up = loss_up + dif
                else:
                    loss_down = loss_down - dif
        print('\n loss_up = ',loss_up,'loss_down = ',loss_down,' loss_all = ',loss_up +loss_down)

        if (delta !=0):
            if(code_index < A_max):
                code_index = code_index + delta
                print('code_index = ',code_index)
            else:
                break
        else:    
            if(loss_up > loss_down):  # 上界损失大于下界损失，增加code_index的上界值
                code_index = code_index + (loss_up -loss_down)/((x+1)*(y+1))
            else:
                code_index = code_index - (loss_down -loss_up)/((x+1)*(y+1))
                
        if(abs(loss_up - loss_down) <= tolerate_err):
            break
        else:
            loss_up = 0
            loss_down = 0

    return code_index


# SVD 分解
def SVD_handle(A):
    #A=np.random.randint(1,25,[5,5])
    u,sigma,vt = la.svd(A)
    length = len(sigma)
    S = np.zeros([length,length])
    #--------test-------------
    print(A)
    for i in range(len(sigma)):
        S[i][i] = sigma[i]
    print('\n S= ',S)
    print('\n u= ',u)
    print('\n vt= ',vt)
    #tmp = np.dot(u,S)
    #result =np.dot(tmp,vt)
    #print(result)
    #---------test--------------
    return u,sigma,vt

#明天写
def w_replace(w,R_1):
    w_update = copy.deepcopy(w)
    for i in w.keys():
        if 'weight' in i and 'layer_hidden2' in i:
            for index, element in np.ndenumerate(w[i].cpu().numpy()):
                #print('\n before w_update [', i , '][',index,']= ',w_update[i][index])
                w_update[i][index] = R_1[index]
                #print('\n after w_update [', i , '][',index,'] = ',w_update[i][index])
    return w_update




# dim_in, dim_hidden1,dim12_sr,dim12_sc, dim_hidden2,dim_out
# 矩阵变换，获取3个矩阵
def model_split_martix(w):
    #w_spar = copy.deepcopy(w)  # step1
    #-step2 获取3个SVD  分解矩阵 W =C R U
    #print("Model's state_dict:")
    for param_tensor in w:
        if(param_tensor == 'layer_hidden1.weight'):
            hidden1_ = w[param_tensor] # tensor 格式的数据
            C = hidden1_.tolist()  #将tensor 的值转换numpy的矩阵形式
            #C =np.transpose(C_) #矩阵转置
        elif(param_tensor == 'layer_hidden2.weight'):
            #print(param_tensor, "\t",model.state_dict()[param_tensor].size())
            hidden2_ = w[param_tensor] # tensor 格式的数据
            R = hidden2_.tolist()  #将tensor 的值转换numpy的矩阵形式
            #R = np.transpose(R_)#矩阵转置
            # E = np.dot(R,R)  #矩阵乘法 ，非点乘  测试
        elif(param_tensor == 'layer_hidden3.weight'):
            #print(param_tensor, "\t",model.state_dict()[param_tensor].size())
            hidden3_ = w[param_tensor] # tensor 格式的数据
            U = hidden3_.tolist()  #将tensor 的值转换numpy的矩阵形式
            #U = np.transpose(U_)#矩阵转置
    #print('\n C = ',C,'\n R= ',R,'\n U = ',U)
    return C , R, U


def found_sc_sr(dim_hidden1,dim_hidden2,compress_rate):
    s = (-(dim_hidden1 + dim_hidden2) + np.sqrt(pow((dim_hidden1 + dim_hidden2),2)+4*compress_rate*dim_hidden1*dim_hidden2))/2
    sc = math.ceil((dim_hidden1/dim_hidden2)*s)
    sr = math.ceil((dim_hidden2/dim_hidden1)*s)
    # verify 
    rate = (dim_hidden1 * sc +sc*sr +sr*dim_hidden2)/(dim_hidden1 *dim_hidden2)
    #print('\n rate = ' , rate )
    return sc ,sr

#求解 R_1
def replace_R(C_0,R_0,U_0,C_1,U_1):
    w_tmp_ = np.dot(U_0,R_0)  #矩阵乘法 ，非点乘
    w_tmp = np.dot(w_tmp_,C_0)
    C1_inv = np.linalg.pinv(C_1)  #求伪逆矩阵
    U1_inv = np.linalg.pinv(U_1) #求伪逆矩阵
    #-------------------test------------------
    """
    C_inv =np.dot(C_left,C_1)  #test
    print('\n C_inv = ',C_inv)
    U_inv =np.dot(U_1,U_right)  #test
    print('\n U_inv = ',U_inv)
    """
    #------------------test--------------------
    R_1_ = np.dot(U1_inv,w_tmp) #矩阵乘法 ，非点乘
    R_1 =  np.dot(R_1_,C1_inv)  #矩阵乘法 ，非点乘
    #print('\n R_1 =',R_1)      #R_1 矩阵
    return R_1


    

def noise_add(w, noise_scale, use_gpu):
    w_noise = copy.deepcopy(w)
    for i in w.keys():
       noise = np.random.normal(0,noise_scale,w[i].size())
       if use_gpu:
           noise = torch.from_numpy(noise).float().cuda()
       else:
           noise = torch.from_numpy(noise).float()
       w_noise[i] = w_noise[i] + noise
    return w_noise

def sparse_evolu(zeta,w):
    w_update = copy.deepcopy(w)
    for i in w.keys():
        if 'weight' in i and 'hidden' in i:
            values_w = list(w[i].view(-1).cpu().numpy())
            num_values = len(values_w)
            value_positive = sorted([d for d in values_w if d > 0])
            value_negative = sorted([d for d in values_w if d < 0])
            p_positive = value_positive[int(len(value_positive)*zeta)]
            p_negative = value_negative[int(len(value_negative)*(1-zeta))]
            num_remove = 0
            for index, element in np.ndenumerate(w[i].cpu().numpy()):
                if element > p_negative and element < p_positive:
                    num_remove += 1
                    w_update[i][index] = 0
            
            # values_w_update = list(w_update[i].view(-1).cpu().numpy())
            # index_nonzero = sorted([d for d in range(len(values_w_update)) if values_w_update[d] != 0])
            ini_set = [d for d in range(num_values)]
            set_gener = []
            while len(set_gener) < num_remove:
                index_gener = random.sample(ini_set,len(ini_set))
                num_add = 0
                for index, element in np.ndenumerate(w_update[i].cpu().numpy()):
                    if num_add in index_gener and w_update[i][index] == 0:
                        # print(w_update[i][index])
                        set_gener.append(num_add)
                        w_update[i][index] = np.random.randn()/10
                    num_add += 1
                ini_set = list(set(ini_set)-set(set_gener))
            
    return  w_update

def prune_reset(w,w_reset):
    for i in w.keys():
        if 'weight' in i and 'hidden' in i:
            for index, element in np.ndenumerate(w[i].cpu().numpy()):
                if w[i][index] == 0:
                    w_reset[i][index] = 0    
    
    return w_reset

def ErdosRenyi_random_graph(epsilon,w):
    w_spar = copy.deepcopy(w)
    for i in w.keys():
        if 'weight' in i and 'hidden' in i:
            h, h_ = w[i].size()
            p = epsilon*(h+h_)/(h*h_)
            print(h,h_,i, p)
            for index, element in np.ndenumerate(w[i].cpu().numpy()):
                #print(i,index ,element)
                pro_matrix = np.random.rand()
                if pro_matrix > p:
                    w_spar[i][index] = 0
    return w_spar, p

#------------------- previous func -------------------
def weights_values(w):
    weight =0
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            weight = weight +abs(w[i][index])
    #print('weight = ',weight)
    return weight

def add_values(w,w_glob):
    w_recover = copy.deepcopy(w)
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            w_recover[i][index] = w[i][index]+w_glob[i][index]
    return w_recover

# 模型差值
def diff_values(w,w_glob):
    w_increas = copy.deepcopy(w)
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            w_increas[i][index] = w[i][index]-w_glob[i][index]
    return w_increas
# 
def quantile_loss(w,w_glob):
    quantile_err = 0
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()):
            quantile_err = quantile_err + abs(w[i][index]-w_glob[i][index])
    return quantile_err

#  桶量化：
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组  
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
            quantile_buck.append(value_sequence[i]+0.000001)
        elif (i+1)%quantile_level == 0:
            quantile_buck.append(value_sequence[i])
            #----------------- test add ---------------
            #print('\n i = ',i,'quantile_level =',quantile_level)
            #code_value = (quantile_buck[-1]+quantile_buck[-2])/2
            #if abs(code_value) <= 0.00000005:
            #    code_value = code_value+ 0.0000001
            #------------------test ------------------------------
            for j in range(k_,i+1):
                quantile_index[sorted_id[j]] = k
                values_increment[sorted_id[j]] = (quantile_buck[-1]+quantile_buck[-2])/2
                #values_increment[sorted_id[j]] = code_value
            k += 1
            k_ = i
        elif len(value_sequence)%quantile_level != 0 and i == len(value_sequence)-1:
            quantile_buck.append(value_sequence[i]-0.000001)
            #------------------- test add  -------------------------
            #code_value = (quantile_buck[-1]+quantile_buck[-2])/2
            #if abs(code_value) <= 0.00000005:
            #    code_value = code_value+ 0.0000001
            #--------------------test add ---------------------- 
            for j in range(k_,i+1):
                quantile_index[sorted_id[j]] = k
                values_increment[sorted_id[j]] = (quantile_buck[-1]+quantile_buck[-2])/2
                #values_increment[sorted_id[j]] = code_value 
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
            quantile_buck.append(value_sequence[i]-0.000001)
        elif (i+1)%quantile_level == 0:
            if i == len(value_sequence)-1:
                quantile_buck.append(value_sequence[i]+0.00001)
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
            quantile_buck.append(value_sequence[i] +0.00001)
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

def quantile_uniform_opt(quantile_level,value_sequence_,sorted_id_):
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
    arr_tmp = []
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
                        code_value =fast_find_x_opt(arr_tmp,acc_level=1)
                        for index in range(i_0,i_1):
                            values_increment[sorted_id[index]] = code_value
                    arr_tmp = []
                    a0= a1
                    a1= a1 + delta
                    code = code +1
                    X = 1
    if len(arr_tmp) == 0 :
        code_value =code_value =(a0+a1)/2
    else :
        i_1 = i
        i_0 = int(i - len(arr_tmp))
        if i_0 !=  i_1 :
            #print('\n len(arr_tmp) = ',len(arr_tmp))
            #print('\n arr_tmp = ',arr_tmp)
            #print('i_0 =',i_0,'i_1 =',i_1)
            code_value =fast_find_x_opt(arr_tmp,acc_level=1)
            for index in range(i_0+1,i_1+1):
                values_increment[sorted_id[index]] = code_value
                    
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
        code_value =code_value =quantile_buck_tmp[code]
    else :
        i_1 = i
        i_0 = int(i - len(arr_tmp))
        if i_0 !=  i_1 :
            #print('\n len(arr_tmp) = ',len(arr_tmp))
            #print('\n arr_tmp = ',arr_tmp)
            #print('i_0 =',i_0,'i_1 =',i_1)
            code_value =quantile_buck_tmp[code]
            for index in range(i_0+1,i_1+1):
                values_increment[sorted_id[index]] = code_value

    return values_increment


    
#  均匀间隔量化优化---ways = 2
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组【数组满足有小到大排序】  
#  sorted_id        : 整理好的数组索引

#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数
def quantile_border_opt(quantile_level,value_sequence_,sorted_id_,quantile_buck_):
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
    quantile_buck_tmp = []
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
                    code_value =fast_find_x_opt(arr_tmp,acc_level=10)
                    quantile_buck_tmp.append(code_value)
                    for index in range(i_0,i_1):
                        values_increment[sorted_id[index]] = code_value
                        #print('code_value =',code_value)
                else:
                    quantile_buck_tmp.append((a0+a1)/2)

                arr_tmp = []
                quantile_buck_id = quantile_buck_id +1
                a0= a1
                a1= quantile_buck_0[quantile_buck_id+1]
                code = code +1
                X = 1

    if len(arr_tmp) == 0 :
        code_value =fast_find_x_opt(arr_tmp,acc_level=100)
        quantile_buck_tmp.append(code_value)
    else :
        i_1 = i
        i_0 = int(i - len(arr_tmp))
        if i_0 !=  i_1 :
            #print('\n len(arr_tmp) = ',len(arr_tmp))
            #print('\n arr_tmp = ',arr_tmp)
            #print('i_0 =',i_0,'i_1 =',i_1)
            code_value =fast_find_x_opt(arr_tmp,acc_level=100)
            quantile_buck_tmp.append(code_value)
            for index in range(i_0+1,i_1+1):
                values_increment[sorted_id[index]] = code_value
        else:
            quantile_buck_tmp.append((a0+a1)/2)
            
    #print('\n quantile_buck_tmp =',quantile_buck_tmp)
    quantile_buck_1[0] =quantile_buck_0[0]
    quantile_buck_1[-1]=quantile_buck_0[-1]

    for i in range(len(quantile_buck_0)-2):
        quantile_buck_1[i+1] = (quantile_buck_tmp[i] + quantile_buck_tmp[i+1])/2

    return quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp




#  均匀间隔量化优化---ways = 2
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组【数组满足有小到大排序】  
#  sorted_id        : 整理好的数组索引

#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数
def quantile_uniform_opt_proposed(quantile_level,value_sequence_,sorted_id_,orig_values_increment,boundary_err = 0.01):
    quantile_buck_0 = np.zeros(quantile_level+1)
    quantile_buck_1 = np.zeros(quantile_level+1)
    quantile_index = np.zeros(len(value_sequence_))
    values_increment = np.zeros(len(value_sequence_))
    sorted_id =copy.deepcopy(sorted_id_)

    max_value = max(value_sequence_)+0.000001
    min_value = min(value_sequence_)-0.000001
    delta = (max_value-min_value)/quantile_level
    #  step 0 : 设置均匀边界
    quantile_buck_ = np.zeros(quantile_level+1)
    for i in range(quantile_level+1):
            quantile_buck_[i] = min_value + delta*i
    
    #  step1 : quantile_buck_ 边界优化，
    quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(quantile_level,value_sequence_,sorted_id,quantile_buck_)

    #-----------------Debug --------------------------------------
    #print('quantile_buck_0 = ',values_increment)
    #print('quantile_buck_0 = ',quantile_buck_0)
    #print('quantile_buck_1 = ',quantile_buck_1)
    #print('quantile_buck_tmp = ',quantile_buck_tmp)
    values_increment_0 =copy.deepcopy(values_increment)
    loss_adapt_err = 0
    for i in range(len(values_increment)):                         
        loss_adapt_err =loss_adapt_err + abs(values_increment_0[i]-orig_values_increment[i])
        #print('\n value_sequence_[',i,']=',values_increment_test[i])
    print('\n initial quantile_err = ',loss_adapt_err)
    #print('\n first values_increment =',values_increment_0)
    quantile_buck_ =copy.deepcopy(quantile_buck_1)
    values_increment = quantile_border_replace(quantile_level,value_sequence_,sorted_id_,quantile_buck_,quantile_buck_tmp)
    values_increment_0 =copy.deepcopy(values_increment)
    loss_adapt_err = 0
    for i in range(len(values_increment)):
        loss_adapt_err =loss_adapt_err + abs(values_increment_0[i]-orig_values_increment[i])
    print('\n adapt quantile_err = ',loss_adapt_err)
    
    #------------------Debug------------------------------------
    
    X =1
    while X:
        quantile_boundary_err =0
        for i in range(len(quantile_buck_0)):
            quantile_boundary_err = quantile_boundary_err + abs(quantile_buck_0[i]-quantile_buck_1[i])
        print('\n boundary_err =',quantile_boundary_err)
        if quantile_boundary_err >= boundary_err :  #设置 容许的总边界误差 根据实际情况调整： 0.01 0.005
        #if quantile_boundary_err >= 0.01 : #超出边界误差 
            quantile_buck_ =copy.deepcopy(quantile_buck_1)
            quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(quantile_level,value_sequence,sorted_id,quantile_buck_)
            #print('\n quantile_buck_tmp =',quantile_buck_1)
            #print('\n values_increment =',values_increment)
            X =1
        else:
            X =0
    #------------    Debug ------------------------------------------------
    #print('\n final values_increment =',values_increment)
    loss_adapt_err = 0
    for i in range(len(values_increment)):
        loss_adapt_err =loss_adapt_err + abs(values_increment[i]- orig_values_increment[i])
    print('\n final quantile_err = ',loss_adapt_err)
    #------------Debug ------------------------------------------------
    return quantile_buck_1,quantile_index,values_increment





#  基于（bucket）桶间隔量化优化---ways = 2
#  输入:
#  quantile_level   : 量化等级    
#  value_sequence   ：排序好的数组【数组满足有小到大排序】  
#  sorted_id        : 整理好的数组索引

#  输出 :  
#  quantile_buck    ： 输出桶的边界值  
#  quantile_index   :  通信传输的编码  
#  values_increment ： 接收通过编码后，进行解码恢复的参数
def quantile_bucket_opt_proposed(quantile_level,value_sequence_,sorted_id_,orig_values_increment,boundary_err = 0.01):
    quantile_buck_0 = np.zeros(quantile_level+1)
    quantile_buck_1 = np.zeros(quantile_level+1)
    quantile_index = np.zeros(len(value_sequence_))
    values_increment = np.zeros(len(value_sequence_))
    sorted_id =copy.deepcopy(sorted_id_)

    quantile_buck_ = np.zeros(quantile_level+1)
    #  step0 : 基于桶边界量化优化，得到初始的桶优化初始边界方案 （本步骤主要目的（标）： 主要得到 quantile_buck_ 的初始输入值）
    quantile_buck_, quantile_index, values_increment=  quantile_bucket_opt(int(np.ceil(len(value_sequence)/quantile_level)),value_sequence_, sorted_id)
    #print('quantile_buck_ = ',quantile_buck_)
    #  step1 : quantile_buck_ 均匀边界
    quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(quantile_level,value_sequence_,sorted_id,quantile_buck_)

    #-----------------Debug --------------------------------------
    #print('quantile_buck_0 = ',values_increment)
    #print('quantile_buck_0 = ',quantile_buck_0)
    #print('quantile_buck_1 = ',quantile_buck_1)
    #print('quantile_buck_tmp = ',quantile_buck_tmp)
    values_increment_0 =copy.deepcopy(values_increment)
    loss_adapt_err = 0
    for i in range(len(values_increment)):                         
        loss_adapt_err =loss_adapt_err + abs(values_increment_0[i]-orig_values_increment[i])
        #print('\n value_sequence_[',i,']=',values_increment_test[i])
    print('\n initial quantile_err = ',loss_adapt_err)
    #print('\n first values_increment =',values_increment_0)
    quantile_buck_ =copy.deepcopy(quantile_buck_1)
    values_increment = quantile_border_replace(quantile_level,value_sequence_,sorted_id_,quantile_buck_,quantile_buck_tmp)
    values_increment_0 =copy.deepcopy(values_increment)
    loss_adapt_err = 0
    for i in range(len(values_increment)):
        loss_adapt_err =loss_adapt_err + abs(values_increment_0[i]-orig_values_increment[i])
    print('\n adapt quantile_err = ',loss_adapt_err)
    
    #------------------Debug------------------------------------
    
    X =1
    while X:
        quantile_boundary_err =0
        for i in range(len(quantile_buck_0)):
            quantile_boundary_err = quantile_boundary_err + abs(quantile_buck_0[i]-quantile_buck_1[i])
        print('\n boundary_err =',quantile_boundary_err)
        if quantile_boundary_err >= boundary_err :  #超出边界误差
        #if quantile_boundary_err >= 0.01 : #超出边界误差
            quantile_buck_ =copy.deepcopy(quantile_buck_1)
            quantile_buck_0,quantile_buck_1,quantile_index,values_increment,quantile_buck_tmp=quantile_border_opt(quantile_level,value_sequence,sorted_id,quantile_buck_)
            #print('\n quantile_buck_tmp =',quantile_buck_1)
            #print('\n values_increment =',values_increment)
            X =1
        else:
            X =0
    #------------    Debug ------------------------------------------------
    #print('\n final values_increment =',values_increment)
    loss_adapt_err = 0
    for i in range(len(values_increment)):
        loss_adapt_err =loss_adapt_err + abs(values_increment[i]- orig_values_increment[i])
    print('\n final quantile_err = ',loss_adapt_err)
    #------------Debug ------------------------------------------------
    return quantile_buck_1,quantile_index,values_increment




#量化还原压缩的模型
#输入 :   
#       w        ： 原始的未排序的值
# quantile_buck  ： 边界值
# value_sequence ： 排序后的量化的值
#输出 ： 
# w_rel          ： 还原量化的值，对量化的值value_sequence恢复到原始排序
def value_replace(w, quantile_buck, value_sequence):
    w_rel = copy.deepcopy(w)
    m = 0
    for j in range(len(quantile_buck)-1):
        for i in w.keys():
            arr = w[i].cpu().numpy()
            arr_index = np.where( (arr>quantile_buck[j+1]) & (arr<=quantile_buck[j])) # 
            #print('w[i] = ',w[i])
            #print("\n arr_inde =",arr_index)
            if len(arr_index[0])>0:
                if len(arr_index) == 1: 
                    for k in range(len(arr_index[0])):
                        w_rel[i][arr_index[0][k]] = (quantile_buck[j]+quantile_buck[j+1])/2
                        m += 1
                elif len(arr_index) == 2:
                    for k in range(len(arr_index[0])):
                        w_rel[i][arr_index[0][k]][arr_index[1][k]] = (quantile_buck[j]+quantile_buck[j+1])/2
                        m += 1
    if m != len(value_sequence):
        print('Quantile Error',len(value_sequence),m)
    return w_rel

def value_replace_2(w, value_sequence):  # w模型形式 ,value_sequence 数组形式，
    w_rel = copy.deepcopy(w)
    m =0
    for i in w.keys():
        for index, element in np.ndenumerate(w[i].cpu().numpy()): #顺序获取每一个值
            w_rel[i][index] = value_sequence[m]
            m =m +1
    #print('\n m=',m,'len(value_sequence) = ',len(value_sequence))    
    if m != len(value_sequence):
        print('Quantile Error',len(value_sequence),m)
    return w_rel

def average_weights(w,w_glob):
    w_avg = copy.deepcopy(w[0])
    if isinstance(w[0],np.ndarray) == True:
        for i in range(1, len(w)):
            w_avg += w[i]
        w_avg = w_avg/len(w)
    else:
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    for k in w_avg.keys():
        w_avg[k] += w_glob[k]
    return w_avg





if __name__ == '__main__':
    # return the available GPU
    """
    av_GPU = torch.cuda.is_available()
    if  av_GPU == False:
        exit('No available GPU')
    print("\n GPU is running !!! \n ")
    """

    if not os.path.exists('./experiresult'):
        os.mkdir('./experiresult')


    batch_size = 128
    learning_rate = 0.01
    num_epochs = 10
    epsilon = 10
    set_epsilon = [70,60,40,20,10,5,1]
    zeta = 0.1
    set_zeta = [0,0.1,0.2,0.3,0.4,0.5,0.6]
    num_experi = 1
    use_gpu = torch.cuda.is_available()
    data_set = 'cifar' # 可选择 选项  'cifar'  'mnist'
    set_iid = 'iid'    # 可选择 选项  'iid'  'non-iid'

    num_users =10      #  总的训练用户数
    num_Chosenusers =5 #  随机选中训练的用户数

    #data_set=['mnist']
    #set_sketch_sche = ['bucket_quantile','uniform_quantile','bucket_quantile_opt','orig','uniform_quantile_opt']
    #set_sketch_sche = ['bucket_quantile','uniform_quantile','orig','uniform_quantile_opt']
    #set_sketch_sche = ['quantile_bucket_opt_proposed','uniform_quantile_opt_propose']
    set_sketch_sche = ['quantile_bucket_opt_proposed']
    sketch_sche = 'bucket_quantile'

    dim_in_layer=784
    dim_hidden1_layer=50
    dim_hidden2_layer=50
    dim_out_layer=10
    compress_rate =0.5  #压缩率 20倍速*log2(1000)
    dim12_sr_layer,dim12_sc_layer = found_sc_sr(dim_hidden1_layer,dim_hidden2_layer,compress_rate)
    print('--------------------------------Information print-------------------------------')
    print('\n dim_hidden1_layer = ',dim_hidden1_layer,' dim_hidden2_layer = ',dim_hidden2_layer)
    print('\n 模型压缩率 compress_rate = ',compress_rate,' dim12_sr_layer = ',dim12_sr_layer,' dim12_sr_layer= ',dim12_sc_layer)

    quantile_level = 64  #量化等级(数目)
    print('\n 量化等级 quantile_level = ',quantile_level,'\n 量化压缩率  quantile_compress_rate = ',np.ceil(np.log2(quantile_level))/32 )


    print('\n num_epochs = ',num_epochs,'num_experi = ',num_experi)
    print('\n set_sketch_sche = ',set_sketch_sche)


    #------------------------------------Data set -----------------------------------------------------
    print('\n -------------------------------Data loader set-------------------------------------------------')
    if data_set == 'mnist' :
        print('\n mnist 数据集 set sucessful !')
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(
            root='./dataset/mnist/', train=True, transform=img_transform, download=False)

        test_dataset = datasets.MNIST(
            root='./dataset/mnist/', train=False, transform=img_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    elif data_set == 'cifar':
        print('\n cifar 数据集 set sucessful !')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10('./dataset/cifar/', train=True, transform=transform, target_transform=None, download=True)
        #test_dataset = copy.deepcopy(train_dataset)
        test_dataset = datasets.CIFAR10('./dataset/cifar/', train=False, transform=transform, target_transform=None, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #------------------------------------Data set --------------------------------------------------------------------

    #------------------------------------Nekural Networ model set--------------------------------------------------------
    print('---------------------------Nekural Networ model list ------------------------------\n ')
    img_size = train_dataset[0][0].shape
    len_in = 1
    for x in img_size:
        len_in *= x

    dim_in_layer= len_in
    model = MLP_triple(dim_in=dim_in_layer, dim_hidden1=dim_hidden1_layer, dim_hidden2=dim_hidden2_layer, dim_out=dim_out_layer)
    #model = MLP_triple_SVD(dim_in=dim_in_layer, dim_hidden1=dim_hidden1_layer,dim12_sr=dim12_sr_layer,dim12_sc=dim12_sc_layer, dim_hidden2=dim_hidden2_layer,dim_out=dim_out_layer)
    print(model)

    #------------------------------------Nekural Networ model set--------------------------------------------------------

    #------------------------------iid set or non-iid data set ---------------------------------------
    # sample users
    """
    if set_iid == 'iid':
        dict_users = mnist_iid(args,dataset_train, args.num_users, args.num_items_train)
        # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test) 
        dict_sever = mnist_iid(args,dataset_test, args.num_users, args.num_items_test)
    else:
        dict_users = mnist_noniid(args, dataset_train, args.num_users, args.num_items_train)
        dict_sever = mnist_iid(args,dataset_test, args.num_users, args.num_items_test)     
    """
    #------------------------------iid set or non-iid data set ---------------------------------------



    if use_gpu:
        model = model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  梯度更新算法
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    print('model.parameters() = ',model.parameters())    

    criterion = nn.CrossEntropyLoss()

    # print(w['layer_hidden.weight'])

    p_list = []
    for k in range(len(set_epsilon)):
        epsilon = set_epsilon[k]
        print('\n epsilon = ',epsilon)

        w_ = model.state_dict() #读取模型中的参数给w_   
        #print(w_)  打印模型参数的值  
        w, p = ErdosRenyi_random_graph(epsilon,w_)
        print('\n P =',p)
        model.load_state_dict(w) #载入修改后的参数，导入到模型中
        p_list.append(p)
        
        #--------- final ----------------- 
        final_bucket_quantile_loss_test_list,final_uniform_quantile_loss_test_list,final_bucket_quantile_opt_loss_test_list =[],[],[]
        final_bucket_quantile_acc_test_list,final_uniform_quantile_acc_test_list,final_bucket_quantile_opt_acc_test_list =[],[],[]
        final_bucket_quantile_err_list,final_uniform_quantile_err_listt,final_bucket_quantile_opt_err_list =[],[],[]
        final_orig_loss_test_list,final_orig_acc_test_list,final_orig_err_listt =[],[],[]
        final_uniform_quantile_opt_loss_test_list,final_uniform_quantile_opt_acc_test_list,final_uniform_quantile_opt_err_listt =[],[],[]
        final_uniform_propose_loss_test_list,final_uniform_propose_opt_acc_test_list,final_uniform_propose_opt_err_listt =[],[],[]
        final_bucket_propose_loss_test_list,final_bucket_propose_opt_acc_test_list,final_bucket_propose_opt_err_listt =[],[],[]
        #--------- final--------------------
        for exper in range(num_experi):
            test_acc, test_loss = 0, 0
            #w_locals, w_locals_1ep, loss_locals, acc_locals = [], [], [], []
            loss_avg_list, acc_avg_list, loss_test_list, acc_test_list, commun_bits_list = [], [], [], [], []
            bucket_quantile_loss_test_list,uniform_quantile_loss_test_list,bucket_quantile_opt_loss_test_list =[],[],[]
            bucket_quantile_acc_test_list,uniform_quantile_acc_test_list,bucket_quantile_opt_acc_test_list =[],[],[]
            bucket_quantile_err_list,uniform_quantile_err_listt,bucket_quantile_opt_err_list =[],[],[]
            orig_loss_test_list,orig_acc_test_list,orig_err_listt =[],[],[]
            uniform_quantile_opt_loss_test_list,uniform_quantile_opt_acc_test_list,uniform_quantile_opt_err_listt =[],[],[]
            uniform_propose_loss_test_list,uniform_propose_opt_acc_test_list,uniform_propose_opt_err_listt =[],[],[]
            bucket_propose_loss_test_list,bucket_propose_opt_acc_test_list,bucket_propose_opt_err_listt =[],[],[]
            timeslot = time.time()
            for ways_ in range(len(set_sketch_sche)):
                sketch_sche = set_sketch_sche[ways_]  
                model.load_state_dict(w) #重新载入原始参数，导入到模型中
                for epoch in range(num_epochs):
                    start_time = time.time()   #时间计数器
                    w_glob = model.state_dict()  # 导出更新的参数模型 w_glob
                    print('*' * 15,f'Experiment: {exper+1}, Epoch: {epoch+1}','*' * 15)
                    #----------------移植 FL 做准备 ----------------------
                    # step1 :随机选择FL用户数目
                    if  num_Chosenusers < num_users:
                        chosenUsers = random.sample(range(num_users),num_Chosenusers)
                        chosenUsers.sort()
                    else:
                        chosenUsers = range(num_users)
                    print("\nChosen users:", chosenUsers)
                    w_locals, w_locals_1ep, loss_locals, acc_locals = [], [], [], []
                    values_glob = []
                    # step2 : 导出模型，将模型中的参数变成一维可处理的数组格式 ，可先不用
                    for i in w_glob.keys():
                        values_glob += list(w_glob[i].view(-1).cpu().numpy())
                    
                    # 联邦学习，随机单个用户更新
                    for idx in range(len(chosenUsers)):
                        #  本地跟新 移植这个东西到新的文件中 明后天弄
                            
                    #----------------移植 FL 做准备 -----------------
                        #print(f'epoch {epoch+1}')
                        running_loss = 0.0
                        running_acc = 0.0
                        num_train = 0
                        w_0_ = model.state_dict() #导出更新之前的模型参数 W_0
                        w_0 = copy.deepcopy(w_0_) 
                        
                        for i, data in enumerate(train_loader, batch_size):
                            #C_0 , R_0, U_0 =model_split_martix(w_0) # 获取更新之前的3个矩阵
                            img, label = data
                            
                            num_train += len(img)
                    
                            if use_gpu:
                                img = img.cuda()
                                label = label.cuda()
                    
                            out = model(img)
                            loss = criterion(out, label)
                            running_loss += loss.item()
                            # print('loss:', loss, running_loss)
                            _, pred = torch.max(out, 1)
                            # print('Lab result:', label)
                            running_acc += sum((pred == label).float().cpu())
                    
                            optimizer.zero_grad() #作用是先将梯度归零（optimizer.zero_grad()）
                            loss.backward()   # 反向传播计算得到每个参数的梯度值（loss.backward()）
                            optimizer.step()  # 最后通过梯度下降执行一步参数更新（optimizer.step()）
                                
                        # reset the zero weights #
                        """
                        if p < 1:
                            w_ = model.state_dict() 
                            w_reset = prune_reset(w,w_)
                            model.load_state_dict(w_reset)
                        if i % 200 == 0:
                            print(f'[{epoch+1}/{num_epochs}] Loss: {running_loss/num_train:.6f}, Acc: {running_acc/num_train:.6f}')
                        """
                        print(f'--------------Finish {epoch+1} epoch---------------\n train_Loss: {running_loss/num_train:.6f}, train_Acc: {running_acc/num_train:.6f}')
                                

                        #-----------------------------------------传输量化压缩处理-------------------------------
                        w_1_ = model.state_dict()          # 导出更新的参数模型 w_1
                        w_1 =copy.deepcopy(w_1_)
                        

                        w_increas = copy.deepcopy(w_1_)  # step1 : 对参数做量化

                        #weights_values(w_increas)
                        values_increment = []             # step2: 获取数值，排序
                        #--------test---------
                        #w1_increment = []             
                        #--------test------
                        for i in w_increas.keys():
                            values_increment += list(w_increas[i].view(-1).cpu().numpy())
                            #----------test-------------
                            #w1_increment += list(w_1[i].view(-1).cpu().numpy())
                            #--------------test --------------------
                        #print('model lenth = ',len(values_increment))
                                                                            
                                                                                # step3 : 桶量化，得到量化的边界值数组 quantile_buck
                        #value_sequence_=[abs(i) for i in value_sequence]
                        #print('\n value_sequence =',sum(value_sequence_))
                        if sketch_sche == 'bucket_quantile':
                            print('ways = bucket_quantile')
                            value_sequence = sorted([d for d in values_increment], reverse=True)  # value sequence
                            sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=True)  
                            #参数数目均匀量化
                            quantile_buck, quantile_index, values_increment = quantile_bucket(int(np.ceil(len(value_sequence)/quantile_level)),value_sequence,sorted_id)
                            if len(quantile_buck)!=(quantile_level+1): # 判断量化后的边界值数目和量化数目是否相等
                                print('Quantile error:',len(quantile_buck),quantile_level)
                        elif sketch_sche == 'bucket_quantile_opt':
                            print('ways = bucket_quantile_opt')
                            value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence
                            sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False)  
                            #参数数目均匀量化
                            quantile_buck, quantile_index, values_increment = quantile_bucket_opt(int(np.ceil(len(value_sequence)/quantile_level)),value_sequence,sorted_id)
                            if len(quantile_buck)!=(quantile_level+1): # 判断量化后的边界值数目和量化数目是否相等
                                print('Quantile error:',len(quantile_buck),quantile_level)
                        elif sketch_sche == 'uniform_quantile':
                            print('ways = uniform_quantile')
                            value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence  reverse=False 有小到大排列
                            sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False) #reverse=False 有小到大排列
                            #区间均匀量化
                            quantile_buck, quantile_index, values_increment = quantile_uniform(quantile_level,value_sequence,sorted_id) #values_increment 的量化传输的差值
                        elif sketch_sche == 'uniform_quantile_opt':
                            print('ways = uniform_quantile_opt')
                            value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence  reverse=False 有小到大排列
                            sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False) #reverse=False 有小到大排列
                            #区间均匀量化
                            quantile_buck, quantile_index, values_increment = quantile_uniform_opt(quantile_level,value_sequence,sorted_id) #values_increment 的量化传输的差值                    
                        elif sketch_sche == 'uniform_quantile_opt_propose': 
                            print('ways = uniform_quantile_opt_propose')
                            orig_values_increment = copy.deepcopy(values_increment)
                            value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence  reverse=False 有小到大排列
                            sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False) #reverse=False 有小到大排列
                            #区间均匀量化-边界优化
                            quantile_buck,quantile_index,values_increment = quantile_uniform_opt_proposed(quantile_level,value_sequence,sorted_id,orig_values_increment)
                        elif sketch_sche == 'quantile_bucket_opt_proposed':
                            print('ways = quantile_bucket_opt_proposed')
                            orig_values_increment = copy.deepcopy(values_increment)
                            value_sequence = sorted([d for d in values_increment], reverse=False)  # value sequence  reverse=False 有小到大排列
                            sorted_id = sorted(range(len(values_increment)), key=lambda i: values_increment[i], reverse=False) #reverse=False 有小到大排列
                            #区间桶量化-边界优化
                            quantile_buck,quantile_index,values_increment = quantile_bucket_opt_proposed(quantile_level,value_sequence,sorted_id,orig_values_increment)   
                        elif sketch_sche == 'orig':
                            print('ways = orig')
                            
                    

                        # step4: 获取每个边界值区间的数，求得最精确的量化值编码的 quantile_index（？？？现在需要做的）
                        # step5: 更新模型的差值 w_increas 原始顺序，values_increment 排序量化后的值
                        #        【让】values_increment 【去替换】 w_increas 的差值，并【恢复原始排序】                                
                        
                        #w_rel = value_replace(w_increas,quantile_buck,values_increment) # values_increment 数组形式， w_rel 模型形式
                        if sketch_sche != 'orig':
                            w_rel = value_replace_2(w_increas, values_increment) # w模型形式 ,value_sequence 数组形式
                        else:
                            w_rel = copy.deepcopy(w_increas)
                        # step6: 本地端接收在w_rel ，通过差值更新w_0

                    
                    w_update_ = copy.deepcopy(w_rel) # step7: 得到传输还原后的平均模型
                    quantile_err_ = diff_values(w_update_,w_1)  # step1: 做差值进行 w_1: 实际跟新 w_update_ ： 量化 更新（对差值进行量化）
                    quantile_err = weights_values(quantile_err_)# 模型绝对值求和
                    print("update_quantile_err = ",quantile_err)
                    
                    
                    model.load_state_dict(w_update_)                   # step4: 载入（导入）更新后的参数

                    #-------------------Debug----------------------------
                    """
                    w_rel_weight = weights_values(w_rel)#  量化差值权重
                    w_0_weight = weights_values(w_0)  #  w0模型权重
                    w_1_weight = weights_values(w_1)  #  w1模型权重                               
                    w_update_weight = weights_values(w_update_)#量化更新模型权重

                    print("\n -------------------- weight list : --------------------- ")
                    print("\n w_0_weight = ",w_0_weight)
                    print("\n w_1_weight = ",w_1_weight)
                    print("\n 量化差值权重 w_rel_weight = ",w_rel_weight)
                    print("\n w_update_weight = ",w_update_weight)
                    print("\n 模型量化误差 quantile_err_wight = ",quantile_err)
                    print("\n -------------------- weight list : --------------------- ")
                    """
                    #-----------------Debug --------------------------------

                    #-----------------------------------------传输量化压缩处理-------------------------------
    

                    """
                    #-----------------------------------------模型 SVD分解-------------------------------
                    # SVD(singular value decomposition) #
                    # w_1 = model.state_dict()
                    # w服务器下发的服务器参数 : w_update_
                    
                    C_1 , R_tmp, U_1 =model_split_martix(w_update_) # step1 : 获取下发的参数模型，更新后的3个矩阵
                    R_1 =replace_R(C_0,R_0,U_0,C_1,U_1)             # step2: 计算得到R_1
                    w_update =w_replace(w_update_,R_1)              # step3: 更新模型参数
                    model.load_state_dict(w_update)                 # step4: 载入（导入）更新后的参数
                    #model.load_state_dict(w_tmp)                   # step4: 载入（导入）更新后的参数
                    #-----------------------------------------模型 SVD分解----------------------------------
                    """
                    
                    model.eval()
                    eval_loss = 0.
                    eval_acc = 0.
                    for data in test_loader:
                        img, label = data
                        if use_gpu:
                            img = img.cuda()
                            label = label.cuda()
                        with torch.no_grad():
                            out = model(img)
                            loss = criterion(out, label)
                        eval_loss += loss.item()
                        _, pred = torch.max(out, 1)
                        eval_acc += (pred == label).float().mean()
                    end_time = time.time()   #时间计数器
                    print(f'Test Loss: {eval_loss/len(test_loader):.6f}, Test_Acc: {eval_acc/len(test_loader):.6f}\n')
                    print("run time : %.2f s."%(end_time-start_time))
                    if sketch_sche == 'bucket_quantile':
                        bucket_quantile_loss_test_list.append(eval_loss/len(test_loader))
                        bucket_quantile_acc_test_list.append(eval_acc.cpu().numpy()/len(test_loader))
                        bucket_quantile_err_list.append(quantile_err.cpu().numpy())
                    elif sketch_sche == 'bucket_quantile_opt':
                        bucket_quantile_opt_loss_test_list.append(eval_loss/len(test_loader))
                        bucket_quantile_opt_acc_test_list.append(eval_acc.cpu().numpy()/len(test_loader))
                        bucket_quantile_opt_err_list.append(quantile_err.cpu().numpy())
                    elif sketch_sche == 'uniform_quantile':
                        #----------make lists-----------------------------
                        uniform_quantile_loss_test_list.append(eval_loss/len(test_loader))
                        uniform_quantile_acc_test_list.append(eval_acc.cpu().numpy()/len(test_loader))
                        uniform_quantile_err_listt.append(quantile_err.cpu().numpy())
                    elif sketch_sche == 'uniform_quantile_opt':
                        #----------make lists-----------------------------
                        uniform_quantile_opt_loss_test_list.append(eval_loss/len(test_loader))
                        uniform_quantile_opt_acc_test_list.append(eval_acc.cpu().numpy()/len(test_loader))
                        uniform_quantile_opt_err_listt.append(quantile_err.cpu().numpy())
                    elif sketch_sche == 'uniform_quantile_opt_propose':
                        print('ways = uniform_quantile_opt_propose')
                        #----------make lists-----------------------------
                        uniform_propose_loss_test_list.append(eval_loss/len(test_loader))
                        uniform_propose_opt_acc_test_list.append(eval_acc.cpu().numpy()/len(test_loader))
                        uniform_propose_opt_err_listt.append(quantile_err.cpu().numpy())
                    elif sketch_sche == 'quantile_bucket_opt_proposed':
                        print('ways = quantile_bucket_opt_proposed')
                        #----------make lists-----------------------------
                        bucket_propose_loss_test_list.append(eval_loss/len(test_loader))
                        bucket_propose_opt_acc_test_list.append(eval_acc.cpu().numpy()/len(test_loader))
                        bucket_propose_opt_err_listt.append(quantile_err.cpu().numpy())
                    elif sketch_sche == 'orig':
                        #----------make lists-----------------------------
                        orig_loss_test_list.append(eval_loss/len(test_loader))
                        orig_acc_test_list.append(eval_acc.cpu().numpy()/len(test_loader))
                        orig_err_listt.append(quantile_err.cpu().numpy())
                    
                    #print('\n loss_test_list = ',loss_test_list)
                    #print('\n acc_test_list = ',acc_test_list)
            print('\n----------plot-------------\n')
            # plot test_loss curve
            labels = "quantile level-{}".format(quantile_level)                           
            plt.figure(exper)
            plt.subplot(131)
            plt.plot(range(len(bucket_quantile_loss_test_list)), bucket_quantile_loss_test_list,'r',label=labels)
            plt.plot(range(len(bucket_quantile_opt_loss_test_list)), bucket_quantile_opt_loss_test_list,'g',label=labels)
            plt.plot(range(len(uniform_quantile_loss_test_list)), uniform_quantile_loss_test_list,'b',label=labels)
            plt.plot(range(len(orig_loss_test_list)), orig_loss_test_list,'black',label=labels)
            plt.plot(range(len(uniform_quantile_opt_loss_test_list)), uniform_quantile_opt_loss_test_list,'orange',label=labels)          
            plt.plot(range(len(uniform_propose_loss_test_list)), uniform_propose_loss_test_list,'c',label=labels)  
            plt.plot(range(len(bucket_propose_loss_test_list)), bucket_propose_loss_test_list,'y',label=labels)
            plt.ylabel('test loss')
            plt.xlabel('epoches')
            plt.grid(linestyle = "--")

            # plot test_acc curve
            labels = "256 level-{}".format(quantile_level)                           
            plt.figure(exper)
            plt.subplot(132)
            plt.plot(range(len(bucket_quantile_acc_test_list)), bucket_quantile_acc_test_list,'r',label=labels)
            plt.plot(range(len(bucket_quantile_opt_acc_test_list)), bucket_quantile_opt_acc_test_list,'g',label=labels)
            plt.plot(range(len(uniform_quantile_acc_test_list)), uniform_quantile_acc_test_list,'b',label=labels)
            plt.plot(range(len(orig_acc_test_list)), orig_acc_test_list,'black',label=labels)
            plt.plot(range(len(uniform_quantile_opt_acc_test_list)), uniform_quantile_opt_acc_test_list,'orange',label=labels)
            plt.plot(range(len(uniform_propose_opt_acc_test_list)), uniform_propose_opt_acc_test_list,'c',label=labels)
            plt.plot(range(len(bucket_propose_opt_acc_test_list)), bucket_propose_opt_acc_test_list,'y',label=labels)
            plt.ylabel('test acc')
            plt.xlabel('epoches')
            plt.grid(linestyle = "--")

            # plot test_acc curve
            labels = "quantile level-{}".format(quantile_level)                           
            plt.figure(exper)
            plt.subplot(133)
            plt.plot(range(len(bucket_quantile_err_list)), bucket_quantile_err_list,'r',label=labels)
            plt.plot(range(len(bucket_quantile_opt_err_list)), bucket_quantile_opt_err_list,'g',label=labels)
            plt.plot(range(len(uniform_quantile_err_listt)), uniform_quantile_err_listt,'b',label=labels)
            plt.plot(range(len(orig_err_listt)), orig_err_listt,'black',label=labels)
            plt.plot(range(len(uniform_quantile_opt_err_listt)), uniform_quantile_opt_err_listt,'orange',label=labels)
            plt.plot(range(len(uniform_propose_opt_err_listt)), uniform_propose_opt_err_listt,'c',label=labels)
            plt.plot(range(len(bucket_propose_opt_err_listt)), bucket_propose_opt_err_listt,'y',label=labels)
            plt.ylabel('quantile err')
            plt.xlabel('epoches')
            plt.grid(linestyle = "--")

            plt.savefig('./experiresult/level-{} exper ={} num_epochs={}  times-{}.pdf'.\
                                format(quantile_level,exper,num_epochs,timeslot))

        #----------make lists-----------------------------

            final_bucket_quantile_loss_test_list.append(bucket_quantile_loss_test_list)
            final_bucket_quantile_acc_test_list.append(bucket_quantile_acc_test_list)
            final_bucket_quantile_err_list.append(bucket_quantile_err_list)

            final_bucket_quantile_opt_loss_test_list.append(bucket_quantile_opt_loss_test_list)
            final_bucket_quantile_opt_acc_test_list.append(bucket_quantile_opt_acc_test_list)
            final_bucket_quantile_opt_err_list.append(bucket_quantile_opt_err_list)
        
            final_uniform_quantile_loss_test_list.append(uniform_quantile_loss_test_list)
            final_uniform_quantile_acc_test_list.append(uniform_quantile_acc_test_list)
            final_uniform_quantile_err_listt.append(uniform_quantile_err_listt)


            final_orig_loss_test_list.append(orig_loss_test_list)
            final_orig_acc_test_list.append(orig_acc_test_list)
            final_orig_err_listt.append(orig_err_listt)

            final_uniform_quantile_opt_loss_test_list.append(uniform_quantile_opt_loss_test_list)
            final_uniform_quantile_opt_acc_test_list.append(uniform_quantile_opt_acc_test_list)
            final_uniform_quantile_opt_err_listt.append(uniform_quantile_opt_err_listt)

            final_uniform_propose_loss_test_list.append(uniform_propose_loss_test_list)
            final_uniform_propose_opt_acc_test_list.append(uniform_propose_opt_acc_test_list)
            final_uniform_propose_opt_err_listt.append(uniform_propose_opt_err_listt)

            final_bucket_propose_loss_test_list.append(bucket_propose_loss_test_list)
            final_bucket_propose_opt_acc_test_list.append(bucket_propose_opt_acc_test_list)
            final_bucket_propose_opt_err_listt.append(bucket_propose_opt_err_listt)
        
        
        #-------------------------------plot final average list ------------------------
        print('\n -----------------------  average list ------------------------------------')
        average_bucket_quantile_loss_test = np.sum(final_bucket_quantile_loss_test_list,axis = 0)
        average_bucket_quantile_acc_test_list = np.sum(final_bucket_quantile_acc_test_list,axis = 0)
        average_bucket_quantile_err_list = np.sum(final_bucket_quantile_err_list,axis = 0)
        print('\n average_bucket_quantile_loss_test = ',average_bucket_quantile_loss_test)
        print('\n average_bucket_quantile_acc_test_list = ',average_bucket_quantile_acc_test_list)
        print('\n average_bucket_quantile_err_list = ',average_bucket_quantile_err_list)


        average_bucket_quantile_opt_loss_test_list = np.sum(final_bucket_quantile_opt_loss_test_list,axis = 0)
        average_bucket_quantile_opt_acc_test_list = np.sum(final_bucket_quantile_opt_acc_test_list,axis = 0)
        average_bucket_quantile_opt_err_list = np.sum(final_bucket_quantile_opt_err_list,axis = 0)
        print('\n average_bucket_quantile_opt_loss_test_list = ',average_bucket_quantile_opt_loss_test_list)
        print('\n average_bucket_quantile_opt_acc_test_list = ',average_bucket_quantile_opt_acc_test_list)
        print('\n average_bucket_quantile_opt_err_list = ',average_bucket_quantile_opt_err_list)
        

        average_uniform_quantile_loss_test_list = np.sum(final_uniform_quantile_loss_test_list,axis = 0)
        average_uniform_quantile_acc_test_list = np.sum(final_uniform_quantile_acc_test_list,axis = 0)
        average_uniform_quantile_err_listt = np.sum(final_uniform_quantile_err_listt,axis = 0)
        print('\n average_uniform_quantile_loss_test_list = ',average_uniform_quantile_loss_test_list)
        print('\n average_uniform_quantile_acc_test_list = ',average_uniform_quantile_acc_test_list)
        print('\n average_uniform_quantile_err_listt = ',average_uniform_quantile_err_listt)


        average_orig_loss_test_list = np.sum(final_orig_loss_test_list,axis = 0)
        average_orig_acc_test_list = np.sum(final_orig_acc_test_list,axis = 0)
        average_orig_err_listt = np.sum(final_orig_err_listt,axis = 0)
        print('\n average_orig_loss_test_list = ',average_orig_loss_test_list)
        print('\n average_orig_acc_test_list = ',average_orig_acc_test_list)
        print('\n average_orig_err_listt = ',average_orig_err_listt)



        average_uniform_quantile_opt_loss_test_list = np.sum(final_uniform_quantile_opt_loss_test_list,axis = 0)
        average_uniform_quantile_opt_acc_test_list = np.sum(final_uniform_quantile_opt_acc_test_list,axis = 0)
        average_uniform_quantile_opt_err_listt = np.sum(final_uniform_quantile_opt_err_listt,axis = 0)
        print('\n average_uniform_quantile_opt_loss_test_list = ',average_uniform_quantile_opt_loss_test_list)
        print('\n average_uniform_quantile_opt_acc_test_list = ',average_uniform_quantile_opt_acc_test_list)
        print('\n average_uniform_quantile_opt_err_listt = ',average_uniform_quantile_opt_err_listt)



        average_uniform_propose_loss_test_list = np.sum(final_uniform_propose_loss_test_list,axis = 0)
        average_uniform_propose_opt_acc_test_list = np.sum(final_uniform_propose_opt_acc_test_list,axis = 0)
        average_uniform_propose_opt_err_listt = np.sum(final_uniform_propose_opt_err_listt,axis = 0)
        print('\n average_uniform_propose_loss_test_list = ',average_uniform_propose_loss_test_list)
        print('\n average_uniform_propose_opt_acc_test_list = ',average_uniform_propose_opt_acc_test_list)
        print('\n average_uniform_propose_opt_err_listt = ',average_uniform_propose_opt_err_listt)

        average_bucket_propose_loss_test_list = np.sum(final_bucket_propose_loss_test_list,axis = 0)
        average_bucket_propose_opt_acc_test_list = np.sum(final_bucket_propose_opt_acc_test_list,axis = 0)
        average_bucket_propose_opt_err_listt = np.sum(final_bucket_propose_opt_err_listt,axis = 0)
        print('\n average_bucket_propose_loss_test_list = ',average_bucket_propose_loss_test_list)
        print('\n average_bucket_propose_opt_acc_test_list = ',average_bucket_propose_opt_acc_test_list)
        print('\n average_bucket_propose_opt_err_listt = ',average_bucket_propose_opt_err_listt)

        

        print('\n -----------------average plot --------------------------------')
        # plot test_loss curve               
        labels = "quantile level-{}".format(quantile_level)                           
        plt.figure(num_experi)
        plt.subplot(131)
        plt.plot(range(len(average_bucket_quantile_loss_test)), average_bucket_quantile_loss_test,'r',label=labels)
        plt.plot(range(len(average_bucket_quantile_opt_loss_test_list)), average_bucket_quantile_opt_loss_test_list,'g',label=labels)
        plt.plot(range(len(average_uniform_quantile_loss_test_list)), average_uniform_quantile_loss_test_list,'b',label=labels)
        plt.plot(range(len(average_orig_loss_test_list)), average_orig_loss_test_list,'black',label=labels)
        plt.plot(range(len(average_uniform_quantile_opt_loss_test_list)), average_uniform_quantile_opt_loss_test_list,'orange',label=labels)          
        plt.plot(range(len(average_uniform_propose_loss_test_list)), average_uniform_propose_loss_test_list,'c',label=labels)
        plt.plot(range(len(average_bucket_propose_loss_test_list)), average_bucket_propose_loss_test_list,'y',label=labels)    
        plt.ylabel('test loss')
        plt.xlabel('epoches')
        plt.grid(linestyle = "--")


        # plot test_acc curve
        labels = "256 level-{}".format(quantile_level)                           
        plt.figure(num_experi)
        plt.subplot(132)
        plt.plot(range(len(average_bucket_quantile_acc_test_list)), average_bucket_quantile_acc_test_list,'r',label=labels)
        plt.plot(range(len(average_bucket_quantile_opt_acc_test_list)), average_bucket_quantile_opt_acc_test_list,'g',label=labels)
        plt.plot(range(len(average_uniform_quantile_acc_test_list)), average_uniform_quantile_acc_test_list,'b',label=labels)
        plt.plot(range(len(average_orig_acc_test_list)), average_orig_acc_test_list,'black',label=labels)
        plt.plot(range(len(average_uniform_quantile_opt_acc_test_list)), average_uniform_quantile_opt_acc_test_list,'orange',label=labels)
        plt.plot(range(len(average_uniform_propose_opt_acc_test_list)), average_uniform_propose_opt_acc_test_list,'c',label=labels)
        plt.plot(range(len(average_uniform_propose_opt_acc_test_list)), average_uniform_propose_opt_acc_test_list,'y',label=labels)
        plt.ylabel('test acc')
        plt.xlabel('epoches')
        plt.grid(linestyle = "--")


        # plot test_acc curve
        labels = "quantile level-{}".format(quantile_level)                           
        plt.figure(num_experi)
        plt.subplot(133)
        plt.plot(range(len(average_bucket_quantile_err_list)), average_bucket_quantile_err_list,'r',label=labels)
        plt.plot(range(len(average_bucket_quantile_opt_err_list)), average_bucket_quantile_opt_err_list,'g',label=labels)
        plt.plot(range(len(average_uniform_quantile_err_listt)), average_uniform_quantile_err_listt,'b',label=labels)
        plt.plot(range(len(average_orig_err_listt)), average_orig_err_listt,'black',label=labels)
        plt.plot(range(len(average_uniform_quantile_opt_err_listt)), average_uniform_quantile_opt_err_listt,'orange',label=labels)
        plt.plot(range(len(average_uniform_propose_opt_err_listt)), average_uniform_propose_opt_err_listt,'c',label=labels)
        plt.plot(range(len(average_bucket_propose_opt_err_listt)), average_bucket_propose_opt_err_listt,'y',label=labels)
        plt.ylabel('quantile err')
        plt.xlabel('epoches')
        plt.grid(linestyle = "--")




        plt.savefig('./experiresult/average level-{} exper ={} num_epochs={}  times-{}.pdf'.\
                            format(quantile_level,exper,num_epochs,timeslot))


        exit('\n --------------- Experiment has finished. ----------------------')
    # print(w_glob['layer_hidden.weight']) 













