# Over-fitting (過擬合)
## 線性模型，限制權重法
先前我們介紹了nonlinear的方式做迴歸分析，我們再次利用這個example

https://github.com/fluttering13/Machine-learning-base/blob/main/polynomial-regression.md

現在我目標是有四個分類並利用上述方式來進行

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/multi-0.png" width="500px"/></div

先前的example遇到了過擬合的問題，也就是model的complexity所造成的問題

因為同時滿足同一個學習目標有無限多種，但是真相只有一個！

但現實上往往不知道我們需要用多少參數，為了讓模型有泛化的能力，我們要做一些修正或限制

所以第一個想法就是，既然參數太多，那如果先禁用一些
$${w_1}^2 + {w_2}^2..... = 0$$

但是我們也不太知道要禁用哪些參數，所以我們可以放寬一下這個限制改寫成
$${E_{2d}} = {\left( {wx - y} \right)^T}\left( {wx - y} \right)\;s.t.\;{w^T}w \le c$$

利用Lagrange multiplier可以改寫一下

$$L = {E_{2d}} + \lambda \left( {{w^T}w - c} \right)$$

$${{\partial L} \over {\partial {w^T}}} = {{\partial {E_{2d}}} \over {\partial w}} + \lambda w = 0$$

在改寫一下令 $\lambda={{2\lambda '} \over N}$

$${2 \over N}\left( {{x^T}xw - {x^T}y} \right) + {{2\lambda '} \over N}w = 0$$

$$w = {\left( {{x^T}x + \lambda I} \right)^{ - 1}}{x^T}y$$

這也就代表其實我們只是加了一個 ${\lambda I}$的修正項

以上是基於線性的模型，我們一樣用上面的多分類的模型來實做看看

當 $\lambda=0$ ，就是不加任何的正則因子時

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/re-example-1.png" width="500px"/></div

從這邊可能不太能夠看出是underfitting的問題還是overfitting的問題，也許兩者都有

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/re-example-2.png" width="500px"/></div

稍微看一下2d norm的loss function，看起來只是有點不穩定，但是loss function應該會隨著模型階數增加而變小

讓我們加了修正項來看看效果

當 $\lambda=1$ 

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/re-example-1-lambda1.png" width="500px"/></div

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/re-example-2-lambda1.png" width="500px"/></div

再拉高一些 當 $\lambda=10$ 

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/re-example-1-lambda10.png" width="500px"/></div

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/re-example-2-lambda10.png" width="500px"/></div

看起來表現的還行！純線性的模型好處就是快

```
##linear multi-class regression 實作
###這邊我們選legredre-polynomial作為transfrom

#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import math
import time
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
    

    learning_error_count=compare_two_array_count(answer,learn_label)
    print('1/0 learning error',learning_error_count)
    test_error_count=compare_two_array_count(answer_test,test_label)
    print('1/0 test error',test_error_count)


    error_01_list.append(learning_error_count)
    validation_01_list.append(test_error_count)

    error_sq=np.linalg.norm(w.dot(new_x_list).T.flatten()-y_prime.flatten(),2)/len(w.dot(new_x_list).T.flatten())
    error_sq_list.append(error_sq)

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
```

## Bayesian regularization

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/re-example-3-0.png" width="500px"/></div
                                                                                                                                                       
承之前介紹過的，一個很方便的正則化技巧，就是我們利用條件機率與對應的機率分佈來進行正則化

實作上就是直接在誤差函數裡面加入正則化的項就好

在這裡選擇l-1 norm 對應到的就是Laplace distribution 因為我們的標籤就是onehot的0,1,2,3
```
loss=torch.nn.CrossEntropyLoss()
loss_fn=loss(y,y_prime)
###regularization L1
lambda_l1=0.0001
loss_fn=loss_fn+torch.sum(torch.abs(torch.flatten(w)))*lambda_l1
```

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/re-example-3-1.png" width="500px"/></div

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/re-example-3-2.png" width="500px"/></div

看起來做得還不錯！我們的目的是要讓validation的loss盡量壓低，並且訓練的loss以能夠隨著模型複雜度增加且下降
