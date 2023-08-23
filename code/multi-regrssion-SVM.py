##Nonlinear multi-class regression 實作
###這邊我們選legredre-polynomial作為transfrom
###利用softmax實作多分類問題

#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import itertools
import scipy
import math
import time
import torch
import cvxpy as cvx
#sample點用(而且它是保證均勻的取 )
def sampling_circle(sample_size,r_sqaure,x_0,y_0):
    a = np.random.uniform(size=sample_size)*r_sqaure
    b = np.random.uniform(size=sample_size)
    x = np.sqrt(a) * np.cos(2 * np.pi * b)+x_0
    y = np.sqrt(a) * np.sin(2 * np.pi * b)+y_0
    return x, y
###模擬數據數量
mock_data_number_list=[5,5,5,5]
verify_data_number_list=[5,5,5,5]
###種類
k=np.size(mock_data_number_list)
###feature number
n=2
#
N=sum(mock_data_number_list)

#把mock data們整理一下，順便加入零的維度(W參數還要考慮一個常數)
#標籤我們後面在依據數量給
def data_generator(mock_data_number,r,x0,y0,label):
  ones=np.ones([1,mock_data_number])
  data1=np.array(sampling_circle(mock_data_number,r**2,x0,y0))
  label_array=label*np.ones([1,mock_data_number])
  #plt.scatter(data1[0,:],data1[1,:])
  data1=np.concatenate([ones,data1],0)
  data1=np.concatenate([data1,label_array],0)
  return data1

data1=data_generator(mock_data_number_list[0],0.2,0.3,0.7,0)
data2=data_generator(mock_data_number_list[1],0.2,0.7,0.7,1)
data3=data_generator(mock_data_number_list[2],0.2,0.7,0.3,2)
data4=data_generator(mock_data_number_list[3],0.2,0.3,0.3,3)

v_data1=data_generator(verify_data_number_list[0],0.2,0.3,0.7,0)
v_data2=data_generator(verify_data_number_list[1],0.2,0.7,0.7,1)
v_data3=data_generator(verify_data_number_list[2],0.2,0.7,0.3,2)
v_data4=data_generator(verify_data_number_list[3],0.2,0.3,0.3,3)

all_data=np.concatenate([data1,data2,data3,data4],1)
test_data=np.concatenate([v_data1,v_data2,v_data3,v_data4],1)

def one_hot(a,k):
  a=a.astype(int)
  return np.eye(k)[a]


def compare_two_array_count(answer,learn_label):
    error_count=0
    for i in range(len(answer)):
        if answer[i]!=learn_label[i]:
            error_count=error_count+1
    return error_count
#print(cvxpy.installed_solvers())

x=all_data[1:n+1,:]
y=one_hot(all_data[n+1,:],k).T
print(y.shape)
w=cvx.Variable([k,n])
b0=cvx.Variable([k,1])
xi=cvx.Variable([k*N])
for i in range(N):
    if i==0:
     b=b0
    else:
        b=cvx.hstack([b,b0])

C=0

obj=cvx.Minimize(cvx.square(cvx.norm(cvx.vec(w)))+C*cvx.sum(xi))
constraints=[]
constraints.append(cvx.vec(cvx.multiply(y,w@x+b))>=1-xi)
constraints.append(xi>=0)
prob = cvx.Problem(obj,constraints)
prob.solve()
x_test=test_data[1:n+1,:]
print(b.value)
print(w.value@x)
print(cvx.multiply(y,w@x+b).T.value)


#print(np.argmax(softmax(w.value@x_test,axis=0),axis=0))




