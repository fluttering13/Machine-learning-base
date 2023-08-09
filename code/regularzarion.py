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
np.random.seed(50)

#sample點用(而且它是保證均勻的取)
def sampling_circle(sample_size,r_sqaure,x_0,y_0):
    a = np.random.uniform(size=sample_size)*r_sqaure
    b = np.random.uniform(size=sample_size)
    x = np.sqrt(a) * np.cos(2 * np.pi * b)+x_0
    y = np.sqrt(a) * np.sin(2 * np.pi * b)+y_0
    return x, y
###模擬數據數量
mock_data_number_list=[50,50,50,50]
verify_data_number_list=[50,50,50,50]
###種類
k=np.size(mock_data_number_list)

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


def new_legredre_poly_x_generator(input_x,m):
    ###input: feature*N
    ###output: (m*feature)*N
    for i in range(1,m+1):
        if i==1:
            all_array=scipy.special.legendre(i)(input_x)
        else:
            new_array=scipy.special.legendre(i)(input_x)
            all_array=np.concatenate([all_array,new_array],axis=0)
    return all_array

def compare_two_array_count(answer,learn_label):
    error_count=0
    for i in range(len(answer)):
        if answer[i]!=learn_label[i]:
            error_count=error_count+1
    return error_count

n=2
error_CE_list=[]
error_sq_list=[]
error_01_list=[]
validation_01_list=[]
highest_order=100

for m in range(2,highest_order):

    input_x=all_data[1:1+n,:]

    new_x_list=new_legredre_poly_x_generator(all_data[1:1+n,:],m).T
    test_x_list=new_legredre_poly_x_generator(test_data[1:1+n,:],m)

    #new_x_list=torch.from_numpy(new_x_list).float()
    #test_x_list=torch.from_numpy(test_x_list).float()

    w=torch.rand(k,2*m, requires_grad=True)

    #y_prime=torch.from_numpy(one_hot(all_data[n+1,:],k).T).float().t()
    y_prime=one_hot(all_data[n+1,:],k)
    #多強的正則因子，越接近1越強
    lam=10
    x_square=new_x_list.T.dot(new_x_list)
    w=scipy.linalg.inv(x_square+lam*np.identity(x_square.shape[0])).dot(new_x_list.T).dot(y_prime)
    
    w=w.T
    new_x_list=new_x_list.T

    answer=np.int64(all_data[-1,:])
    answer_test=np.int64(test_data[-1,:])
    learn_label=np.argmax(w.dot(new_x_list),axis=0)
    test_label=np.argmax(w.dot(test_x_list),axis=0)
    
    #print('2d norm error',err_his[-1])

    learning_error_count=compare_two_array_count(answer,learn_label)
    print('1/0 learning error',learning_error_count)
    test_error_count=compare_two_array_count(answer_test,test_label)
    print('1/0 test error',test_error_count)

    #error_CE_list.append(err_his[-1])
    error_01_list.append(learning_error_count)
    validation_01_list.append(test_error_count)

    error_sq=np.linalg.norm(w.dot(new_x_list).T.flatten()-y_prime.flatten(),2)/len(w.dot(new_x_list).T.flatten())
    error_sq_list.append(error_sq)
# print(error_CE_list)

# #plt.scatter(range(1,highest_order),error_CE_list,label='error_CE')
plt.scatter(range(2,highest_order),error_01_list,label='train_error_01')
plt.scatter(range(2,highest_order),validation_01_list,label='validation_error_01')

plt.legend(
    loc='upper left',
    fontsize=10,
    shadow=True,
    facecolor='#ccc',
    edgecolor='#000',
    title='m-oder-ploy-regression',
    title_fontsize=10)
plt.show()


plt.scatter(range(2,highest_order),error_sq_list,label='2d_error_01')
plt.show()