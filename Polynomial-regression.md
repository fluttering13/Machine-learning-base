# Polynomial-regression

先前我們處理過的都是利用線性的方式去做分類，

通常我們就是在多維空間裡面去找一條線來區分這些群集，

假如今天是想要用一條曲線來區別這些群集呢？

那我們就要做一件事情就是用一個function把原來的這些data們在空間A轉到另外一個空間B，

我們就在空間B一樣做線性的事情，但是在還原回去A空間，它就是一條曲線。

所以本質上我們還是做線性的事情，只是在另外一個空間裡面做，最後就還原回去就好。

舉例來說，假如我今天有兩個feature $x_1$, $x_2$ 與四個分類

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/multi-0.png" width="500px"/></div>

原本之前做的事情是我們想找到
${c_0} + {c_1}{x_1} + {c_2}{x_2} = 0$
這個點去區分不同群集

我們今天試著用比較高層次的方式去看它，即
${c_0} + {c_1}{x_1}^2 + {c_2}{x_2}^2 + {c_3}{x_1}{x_2} = 0$

我們用 $x_0$ 跟 $x_0$ $x_1$來做圖來稍微看一下

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/multi-poly-0.png" width="500px"/></div>

```
###這個code可以放在下面的那實作上面，只是專門畫這張圖而已
count=0
for num in mock_data_number_list:
  plot_x=all_data[1,count:count+num]
  plot_y=new_x_list[2,count:count+num]
  plt.scatter(plot_x,plot_y)
  count=count+num
plt.show()
```
在高維的空間投影似乎是線性可分的(注意，我們的mock data其實就是二階函數產生出來的，這不意外)

以上，我們在這個高維空間的二維投影告訴我們似乎是可以用一條線將這些data切一切

所以這裡的概念是我們投影去高維空間後，試著去找到切割這個高維空間的超切面 (hyperplane)

然後再轉回來原本我們看的空間 $\left( {{x_0},{x_1}} \right)$

意思是今天我做分類前，我可以想多一點，考慮多一點的參數，這邊我們要做一個轉換 (transform)

在這邊我們用到的是簡單的polynomial的轉換

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/multi-poly-1.png" width="500px"/></div>

```
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
  plt.scatter(data1[0,:],data1[1,:])
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

###畫這圖真是有夠麻煩，因為是多項式
###演篹法很簡單就是用篩的，接近的點就把它畫出來
plot_x=np.arange(-1,2,0.01)
plot_y=np.arange(-1,2,0.01)
coordinate=[(x,y) for x in plot_x for y in plot_y]
pass_list_x=[]
pass_list_y=[]


coordinate=np.array(coordinate)
N_c=coordinate.shape[0]
coordinate_data=np.concatenate([np.ones([N_c,1]),coordinate,np.ones([N_c,1])],axis=1).T
poly_coordinate_data,order_list=new_poly_x_generator(coordinate_data,m)
coordinate_poly_values=w.dot(poly_coordinate_data)

###對第一條w作畫圖
for i in range(coordinate_poly_values.shape[1]):
  if abs(coordinate_poly_values[0,i])<=0.02:
    pass_list_x.append(coordinate[i,0])
    pass_list_y.append(coordinate[i,1])



plt.scatter(pass_list_x,pass_list_y,label='spilt line '+str(0))
plt.legend(
    loc='lower right',
    fontsize=10,
    shadow=True,
    facecolor='#ccc',
    edgecolor='#000',
    title='polynomial regression plot',
    title_fontsize=10)
plt.show()



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
```

我們可以把學到的參數w，在把它畫回去，它就是一條曲線

我們來看看到m個order擬合的效果怎麼樣

```
###Nonlinear multi-class regression 實作
###這邊我們選polynomial包含cross terms作為transfrom
###利用softmax實作多分類問題

#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.special import softmax
import math
import time
import torch
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

def compare_two_array_count(answer,learn_label):
    error_count=0
    for i in range(len(answer)):
        if answer[i]!=learn_label[i]:
            error_count=error_count+1
    return error_count

n=2
error_CE_list=[]
error_01_list=[]
validation_01_list=[]
highest_order=50
for m in range(1,highest_order):
    new_x_list,order_list=new_poly_x_generator(all_data,m)
    test_x_list,order_list=new_poly_x_generator(test_data,m)

    new_x_list=torch.from_numpy(new_x_list).float()
    test_x_list=torch.from_numpy(test_x_list).float()

    w=torch.rand(k,len(order_list)+1, requires_grad=True)

    y_prime=torch.from_numpy(one_hot(all_data[n+1,:],k).T).float().t()

    #loss=torch.nn.MSELoss()
    loss=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    opt_d=10
    count=0
    err_his=[]
    while opt_d>0.00001:
        y=torch.mm(w,new_x_list).t()
        loss_fn=loss(y,y_prime)

        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss_fn.backward()         # 误差反向传播, 计算参数更新值 backward懶人包
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
        err_his.append(loss_fn.item())
        count=count+1
        if count>1:
            opt_d=abs(err_his[-1]-err_his[-2])

    w=w.detach().numpy()
    answer=np.int64(all_data[-1,:])
    answer_test=np.int64(test_data[-1,:])
    learn_label=np.argmax(softmax(w.dot(new_x_list),axis=0),axis=0)
    test_label=np.argmax(softmax(w.dot(test_x_list),axis=0),axis=0)

    print('CE norm error',err_his[-1])

    learning_error_count=compare_two_array_count(answer,learn_label)
    print('1/0 learning error',learning_error_count)
    test_error_count=compare_two_array_count(answer_test,test_label)
    print('1/0 test error',test_error_count)

    error_CE_list.append(err_his[-1])
    error_01_list.append(learning_error_count)
    validation_01_list.append(test_error_count)

print(error_CE_list)

#plt.scatter(range(1,highest_order),error_CE_list,label='error_CE')
plt.scatter(range(1,highest_order),error_01_list,label='train_error_01')
plt.scatter(range(1,highest_order),validation_01_list,label='validation_error_01')
plt.legend(
    loc='upper left',
    fontsize=10,
    shadow=True,
    facecolor='#ccc',
    edgecolor='#000',
    title='m-oder-ploy-regression',
    title_fontsize=10)
plt.show()
```

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/m-poly.png" width="500px"/></div>

看起來是非常得欠擬合呢，我們要需要對模型做一些修改，

我們重新看，我們寫得是general的多項式，其中包含了cross terms，

也就是說在優化的時候**這些相交的基底會互相影響，導致優化效能不佳**，所以我們要用一些特別的方法讓優化效果更好一些

在做Transform的時候，我們從原本的多項式轉換

$$\phi \left( {{x_1},{x_2}......} \right) = \lbrace {x_1}^n,{x_1}^{n - 1}{x_2}^1...... \rbrace $$

變成以legendre polynomial做為轉換，只要是基底都是正交的都可以

$$\phi \left( {{x_1},{x_2}......} \right) = \lbrace {P_1}\left( {{x_1}} \right),{P_1}\left( {{x_2}} \right),{P_2}\left( {{x_1}} \right)...... \rbrace $$

來看看結果

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/m-legre-poly.png" width="500px"/></div>

Cross-entropy error隨著階數越高，轉成

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/m-legre-poly-CE-error.png" width="500px"/></div>

underfitting欠擬合的危機解除！再來就是要修過擬合的問題啦，下一篇會介紹！

https://github.com/fluttering13/Machine-learning-base/blob/main/Regularization-example.md

```
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
#sample點用(而且它是保證均勻的取)
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
error_01_list=[]
validation_01_list=[]
highest_order=50

for m in range(2,highest_order):

    input_x=all_data[1:1+n,:]

    new_x_list=new_legredre_poly_x_generator(all_data[1:1+n,:],m)
    test_x_list=new_legredre_poly_x_generator(test_data[1:1+n,:],m)

    new_x_list=torch.from_numpy(new_x_list).float()
    test_x_list=torch.from_numpy(test_x_list).float()

    w=torch.rand(k,2*m, requires_grad=True)

    y_prime=torch.from_numpy(one_hot(all_data[n+1,:],k).T).float().t()

    #loss=torch.nn.MSELoss()
    loss=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    opt_d=10
    count=0
    err_his=[]
    while opt_d>0.00001:
        y=torch.mm(w,new_x_list).t()
        loss_fn=loss(y,y_prime)

        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss_fn.backward()         # 误差反向传播, 计算参数更新值 backward懶人包
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
        err_his.append(loss_fn.item())
        count=count+1
        if count>1:
            opt_d=abs(err_his[-1]-err_his[-2])
        
    w=w.detach().numpy()
    answer=np.int64(all_data[-1,:])
    answer_test=np.int64(test_data[-1,:])
    learn_label=np.argmax(softmax(w.dot(new_x_list),axis=0),axis=0)
    test_label=np.argmax(softmax(w.dot(test_x_list),axis=0),axis=0)

    print('CE norm error',err_his[-1])

    learning_error_count=compare_two_array_count(answer,learn_label)
    print('1/0 learning error',learning_error_count)
    test_error_count=compare_two_array_count(answer_test,test_label)
    print('1/0 test error',test_error_count)

    error_CE_list.append(err_his[-1])
    error_01_list.append(learning_error_count)
    validation_01_list.append(test_error_count)


print(error_CE_list)

#plt.scatter(range(1,highest_order),error_CE_list,label='error_CE')
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


plt.scatter(range(2,highest_order),error_CE_list,label='validation_error_01')
plt.show()
```
