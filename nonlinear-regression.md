# Nonlinear regression

先前我們處理過的都是利用線性的方式去做分類，

通常我們就是在多維空間裡面去找一條線來區分這些群集，

假如今天是想要用一條曲線來區別這些群集呢？

那我們就要做一件事情就是用一個function把原來的這些data們在空間A轉到另外一個空間B，

我們就在空間B一樣做線性的事情，但是在還原回去A空間，它就是一條曲線。

所以本質上我們還是做線性的事情，只是在另外一個空間裡面做，最後就還原回去就好。

舉例來說，假如我今天有兩個feature $x_1$ $x_2$

原本之前做的事情是我們想找到
${c_0} + {c_1}{x_1} + {c_2}{x_2} = 0$
這個點去區分不同群集

但現在我想要看的是
${c_0} + {c_1}{x_1}^2 + {c_2}{x_2}^2 + {c_3}{x_1}{x_2} = 0$
可能的圓錐曲線有哪一些


```
###Nonlinear multi-class regression 實作

###我們來利用softmax實作多分類問題
#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.special import softmax

#sample點用(而且它是保證均勻的取)
def sampling_circle(sample_size,r_sqaure,x_0,y_0):
    a = np.random.uniform(size=sample_size)*r_sqaure
    b = np.random.uniform(size=sample_size)
    x = np.sqrt(a) * np.cos(2 * np.pi * b)+x_0
    y = np.sqrt(a) * np.sin(2 * np.pi * b)+y_0
    return x, y
###模擬數據數量
mock_data_number_list=[50,50,50,50]
###種類
k=np.size(mock_data_number_list)
###弄一下label，後面可以對答案
for i in range(0,np.size(mock_data_number_list)):
  if i==0:
    real_label=np.zeros(mock_data_number_list[0])
  else:
    real_label=np.concatenate((real_label,np.ones(mock_data_number_list[i])*i))
#把mock data們整理一下，順便加入零的維度(W參數還要考慮一個常數)
#標籤我們後面在依據數量給
def data_generator(mock_data_number,r,x0,y0,label):
  ones=np.ones([1,mock_data_number])
  data1=np.array(sampling_circle(mock_data_number,r**2,x0,y0))
  label_array=label*np.ones([1,mock_data_number])
  plt.scatter(data1[0,:],data1[1,:])
  data1=np.concatenate([ones,data1],0)
  data1=np.concatenate([data1,label_array],0)
  return data1

data1=data_generator(mock_data_number_list[0],0.2,0.3,0.7,0)
data2=data_generator(mock_data_number_list[1],0.2,0.7,0.7,1)
data3=data_generator(mock_data_number_list[2],0.2,0.7,0.3,2)
data4=data_generator(mock_data_number_list[3],0.2,0.3,0.3,3)

all_data=np.concatenate([data1,data2,data3,data4],1)

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
  return w,y,error,error_his,count



###feature number n
###x_1 x_2 ...x_n
n=all_data.shape[0]-2

###the higest order m
###for example n=3,m=2,one of polynomial term: x_1^1 x_2^(m-1) x_3^(0)
m=2

###polynomial terms number
terms_number=int(np.math.factorial(n+m-1)/(np.math.factorial(m)*np.math.factorial(n-1)))

#取一下變數名稱

#3是常數0維+X跟Y兩個featute

for i in range(n):
  if i==0:
    index_string='0'
  else:
    index_string=index_string+str(i)

order_list=list(itertools.combinations_with_replacement(index_string,m))
print('order_indeies',order_list)

new_x_list=[]
for h in range(all_data.shape[1]):
  tmp_list=[]
  tmp_list.append(1)
  single_vetor_x=all_data[0:n+1,h].flatten()
  for i in range(len(order_list)):
    tmp_product=1
    for j in range(m):
      tmp_product=tmp_product*single_vetor_x[1+int(order_list[i][j])]
    tmp_list.append(tmp_product)
  new_x_list.append(np.array(tmp_list))
new_x_list=np.array(new_x_list).T

label_array_one_hot=one_hot(all_data[n+1,:],k).T

w=np.random.normal(size=(k, terms_number+1))

# ###簡簡單單的優化
w,y,error,error_his,count=Optimzer_GD_2D_norm(0.01,w,new_x_list,label_array_one_hot)
print(w)
print(error)
for i in range(0,k):
  print('line shape',w[i][2]**2-4*w[i][1]*w[i][3])

#print('train_round',count)
print('trained_result\n',np.vstack((np.argmax(y,axis=0),all_data[k-1,:].astype(int))))

###畫這圖真是有夠麻煩，因為是多項式
plot_x=np.arange(-1,2,0.01)

plot_y=np.arange(-1,2,0.01)
coordinate=[(x,y) for x in plot_x for y in plot_y]
pass_list=[]
for g in range(k):
  pass_list_x=[]
  pass_list_y=[]
  for h in range(len(coordinate)):
    xy_list=coordinate[h]
    tmp_sum=0
    for i in range(len(order_list)):
      tmp_product=1
      for j in range(m):
        tmp_product=tmp_product*xy_list[int(order_list[i][j])]
      tmp_sum=tmp_sum+tmp_product*w[g,i]
    test=w[g,:].dot(new_x_list[:,g])
    score_list.append(abs(tmp_sum-test))
    if abs(tmp_sum-test)<0.02:
      pass_list_x.append(np.array(xy_list[0]))
      pass_list_y.append(np.array(xy_list[1]))
  print(min(score_list))
  pass_list=np.array(pass_list)
  plt.scatter(pass_list_x,pass_list_y,label='spilt line '+str(g))

plt.legend(
    loc='lower right',
    fontsize=10,
    shadow=True,
    facecolor='#ccc',
    edgecolor='#000',
    title='test',
    title_fontsize=10)
```
