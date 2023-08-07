###Nonlinear multi-class regression 實作
###這邊我們選polynomial包含cross terms作為transfrom

###我們來利用softmax實作多分類問題
#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.special import softmax
import math
import time
#sample點用(而且它是保證均勻的取)
def sampling_circle(sample_size,r_sqaure,x_0,y_0):
    a = np.random.uniform(size=sample_size)*r_sqaure
    b = np.random.uniform(size=sample_size)
    x = np.sqrt(a) * np.cos(2 * np.pi * b)+x_0
    y = np.sqrt(a) * np.sin(2 * np.pi * b)+y_0
    return x, y
###模擬數據數量
mock_data_number_list=[50,50,50,50]
verify_data_number_list=[5,5,5,5]
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

n=all_data.shape[0]-2
def sigmoid(x):
  result=1/(1+np.exp(-x))
  return result

def cross_entropy_error(w,x,y):
  N=np.size(y)
  s=w.dot(x)
  error=np.sum(-np.log(sigmoid(y*w.dot(x))))/N
  return error

def norm_2D_error(x,y):
  return np.sum(np.power(x-y,2))

def one_hot(a,k):
  a=a.astype(int)
  return np.eye(k)[a]

def Optimzer_GD_2D_norm(lr,w,x_0,y_prime):
  start = time.time()
  count=0
  error_his=[]
  error_diff=10
  while error_diff>0.00001:
    y=w.dot(x_0)
    y=softmax(y,axis=0)
    gradient=(y_prime-y).dot(x_0.T)
    w=w+gradient*lr
    error=norm_2D_error(y,y_prime)
    error_his.append(error)
    count=count+1
    if count>1:
      error_diff=abs(error_his[-1]-error_his[-2])
  end= time.time()
  print('running_time',end-start)
  return w,y,error,error_his,count



def new_poly_x_generator(all_data,m):
###輸入(1常數+feature+1標籤)*N
###輸出N*(1常數+new_feature)  
  n=all_data.shape[0]-2
  for i in range(n):
    if i==0:
      index_string='0'
    else:
      index_string=index_string+str(i)
  order_list=list(itertools.combinations_with_replacement(index_string,m))
  #print('order_indeies',order_list)
  ###以下要按照polynomial建置新的feature
  new_x_list=[]
  ###tmp_list用來就是新的一列feature
  ###new_x_list就是所有的
  for h in range(all_data.shape[1]):
    tmp_list=[]
    tmp_list.append(1)
    ###取出單個點的feature
    single_vetor_x=all_data[1:n+1,h].flatten()
    #根據index來相乘
    for i in range(len(order_list)):
      tmp_product=1
      for j in range(m):
        tmp_product=tmp_product*single_vetor_x[int(order_list[i][j])]
      tmp_list.append(tmp_product)
    new_x_list.append(np.array(tmp_list))
  new_x_list=np.array(new_x_list).T
  return new_x_list,order_list

m=2
new_x_list,order_list=new_poly_x_generator(all_data,m)
test_x_list,order_list=new_poly_x_generator(test_data,m)

label_array_one_hot=one_hot(all_data[n+1,:],k).T
w=np.random.normal(size=(k, len(order_list)+1))
###簡簡單單的優化
w,y,error,error_his,count=Optimzer_GD_2D_norm(0.01,w,new_x_list,label_array_one_hot)
#print(w)
#print('learning error',error)
answer=np.int64(all_data[-1,:])
answer_test=np.int64(test_data[-1,:])
learn_label=np.argmax(softmax(w.dot(new_x_list),axis=0),axis=0)
test_label=np.argmax(softmax(w.dot(test_x_list),axis=0),axis=0)

def compare_two_array_count(answer,learn_label):
  error_count=0
  for i in range(len(answer)):
    if answer[i]!=learn_label[i]:
      error_count=error_count+1
  return error_count

learning_error_count=compare_two_array_count(answer,learn_label)
print('1/0 learning error',learning_error_count)
test_error_count=compare_two_array_count(answer_test,test_label)
print('1/0 test error',test_error_count)
################################################################
count=0
for num in mock_data_number_list:
  plot_x=all_data[1,count:count+num]
  plot_y=new_x_list[2,count:count+num]
  plt.scatter(plot_x,plot_y)
  count=count+num
plt.show()
###畫這圖真是有夠麻煩，因為是多項式
###演篹法很簡單就是用篩的，接近的點就把它畫出來
# plot_x=np.arange(-1,2,0.01)
# plot_y=np.arange(-1,2,0.01)
# coordinate=[(x,y) for x in plot_x for y in plot_y]
# pass_list_x=[]
# pass_list_y=[]


# coordinate=np.array(coordinate)
# N_c=coordinate.shape[0]
# coordinate_data=np.concatenate([np.ones([N_c,1]),coordinate,np.ones([N_c,1])],axis=1).T
# poly_coordinate_data,order_list=new_poly_x_generator(coordinate_data,m)
# coordinate_poly_values=w.dot(poly_coordinate_data)

# ###對第一條w作畫圖
# for i in range(coordinate_poly_values.shape[1]):
#   if abs(coordinate_poly_values[0,i])<=0.02:
#     pass_list_x.append(coordinate[i,0])
#     pass_list_y.append(coordinate[i,1])



# plt.scatter(pass_list_x,pass_list_y,label='spilt line '+str(0))
# plt.legend(
#     loc='lower right',
#     fontsize=10,
#     shadow=True,
#     facecolor='#ccc',
#     edgecolor='#000',
#     title='polynomial regression plot',
#     title_fontsize=10)
# plt.show()



#     tmp_sum=0
#     for i in range(len(order_list)):
#       tmp_product=1
#       for j in range(m):
#         tmp_product=tmp_product*xy_list[int(order_list[i][j])]
#       tmp_sum=tmp_sum+tmp_product*w[g,i]
#     test=w[g,:].dot(new_x_list[:,g])
#     score_list.append(abs(tmp_sum-test))
#     if abs(tmp_sum-test)D:
#       pass_list_x.append(np.array(xy_list[0]))
#       pass_list_y.append(np.array(xy_list[1]))
#   print(min(score_list))
#   pass_list=np.array(pass_list)
#   plt.scatter(pass_list_x,pass_list_y,label='spilt line '+str(g))

# plt.legend(
#     loc='lower right',
#     fontsize=10,
#     shadow=True,
#     facecolor='#ccc',
#     edgecolor='#000',
#     title='test',
#     title_fontsize=10)
# plt.show()