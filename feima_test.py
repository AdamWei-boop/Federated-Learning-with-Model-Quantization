import math
import random
import matplotlib.pyplot as plt
import numpy as np

#一维多边形费马点求解
# 输入： Xi  : 一维数组的所有点
# 输出： x   ：拟合的最优点
def loss_min(Xi):
      n=len(Xi)
      x=sum(Xi)/n
      loss = 0
      while True:
            xfenzi=0
            xfenmu=0
            for i in range(n):
                  g=math.sqrt((x-Xi[i])**2)
                  xfenzi=xfenzi+Xi[i]/g
                  xfenmu=xfenmu+1/g
            xn=xfenzi/xfenmu
            print('\n x = ',x,)
            if abs(xn-x)<0.001 :
                  break
            else:
                  x=xn
      x=xn
      #----- tset loss -------
      y =np.zeros(n)
      plt.scatter(Xi,y,color='r')
      for i in range(len(Xi)):
            loss += abs(x-Xi[i])
      print('\n x = ',x, ' loss = ',loss)
      plt.scatter(x,0,color='b')
      plt.show()
      return x

a1 = np.array([0,0,0,0,4,35,6,2,50,38,77,129])
a2 = np.array([0,0,0,0,0,0,0,0,0,0,0,1])
x = loss_min(a2)