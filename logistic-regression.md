# 邏輯回歸 logistic-regression
先前可以回顧PLA跟linear regression，我們都是對分類問題做線性的優化，在這個條目會有非線性的優化

我們可以藉由簡單的數值求解來找到極值

https://github.com/fluttering13/Machine-learning-base/blob/main/PLA-zh.md

https://github.com/fluttering13/Machine-learning-base/blob/main/Linear-regression.md

有時候我們對於分類問題，需要提供一個分數或是機率的評估

例如說，根據以往病人的病歷去評估一個病人的手術成功率

根據品牌的價格與市占率，去評估該商品的銷售概率或是評比

所以真實數據本身只告訴你在某些情況的有無，如手術成功或失敗，貨品已銷售還是在架上

這個時候我們就需要從眾多DATA $X$ 優化參數 $W$ 去評估機率
$$\alpha (wx)$$
其中 $\alpha $ 一個特殊的函數可以把任意的數值map到區間0到1，這邊所選取的是sigmoid函數
$${1 \over {{e^{ - wx}} + 1}}$$

至於這個函數是怎麼訂出來的，詳見

https://github.com/fluttering13/Machine-learning-base/blob/main/why-and-how-to-use-activation-and-loss.md

激勵函數如果選擇sigmoid，那很自然的，我們就會使用cross-entropy作為誤差函數，來衡量標籤 $y$ 與模型預測 ${w^T}{x_n}$ 之間的對比(Contrasts)

$${E_{ce}} = {\min _w}{1 \over N}\sum\limits_n^N { - \ln \alpha \left( {{y_n}{w^T}{x_n}} \right)} $$

由於這邊我們有非線性函數 $\alpha$ ，所以我們直接用數值求解的方式找到誤差函數極小的值的地方

$${E_{ce}} = {\min _w}{1 \over N}\sum\limits_n^1 { - \ln \alpha \left( {{y_n}{w^T}{x_n}} \right)} $$
接著我們就是在數學上寫下它的梯度，丟給演算法更新

$$\nabla {E_{ce}} = {1 \over N}\sum\limits_n^1 { - \ln \alpha \left( {{y_n}{w^T}{x_n}} \right)} \left( {{y_n}{x_n}} \right)$$

這邊的優化演算法叫做梯度下降GD(gradient descent)，在數值方法叫做尤拉方法，其實就是根據周邊的斜率去找極值的概念
$$w' = w - l\nabla {E_{ce}}\left( w \right)$$

接著我們來實作一下
```
##logistic regression 實作
#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
mock_data_number_each_group=100 #模擬數據數量

#sample點用
def sampling_circle(sample_size,r_sqaure,x_0,y_0):
    a = np.random.uniform(size=sample_size)*r_sqaure
    b = np.random.uniform(size=sample_size)
    x = np.sqrt(a) * np.cos(2 * np.pi * b)+x_0
    y = np.sqrt(a) * np.sin(2 * np.pi * b)+y_0
    return x, y
ones=np.ones([1,mock_data_number_each_group])

data1=np.array(sampling_circle(mock_data_number_each_group,0.4**2,0.25,0.75))
label=np.ones([1,mock_data_number_each_group])
plt.scatter(data1[0,:],data1[1,:])
data1=np.concatenate([ones,data1],0)
data1=np.concatenate([data1,label],0)

data2=np.array(sampling_circle(mock_data_number_each_group,0.4**2,0.75,0.25))
label=-np.ones([1,mock_data_number_each_group])
plt.scatter(data2[0,:],data2[1,:])
data2=np.concatenate([ones,data2],0)
data2=np.concatenate([data2,label],0)
all_data=np.concatenate([data1,data2],1)

def sigmoid(x):
  result=1/(1+np.exp(-x))
  return result 

def cross_entropy_error(w,x,y):
  N=np.size(y)
  s=w.dot(x)
  error=np.sum(-np.log(sigmoid(y*w.dot(x))))/N
  return error

#GD實作
x=all_data[0:3,:]
y=all_data[3,:]
N=np.size(y)

#w初始化
w=np.zeros([1,3])
#設定learning rate
lr=0.1
for i in range(10000):
  ###其中sigmoid(-y*w.dot(x))可理解成是對(-y*x)的線性加權
  gd=np.sum(sigmoid(-y*w.dot(x))*(-y*x),1)/N
  w=w-lr*gd
print('err',cross_entropy_error(w,x,y))
print('w',w)
#算完就把W換算劃線
w=w.flatten()
plt_x=np.linspace(0.1,0.9,mock_data_number_each_group)
plt.plot(plt_x,-w[1]/w[2]*plt_x-w[0]/w[2])

##測試一下未知點得分
x_test1=np.array([1,0,1])
x_test2=np.array([1,1,0])
x_test3=np.array([1,0.5,0.5])
print(sigmoid(w.dot(x_test1)))
print(sigmoid(w.dot(x_test2)))
print(sigmoid(w.dot(x_test3)))
```
這邊為了方便我們省略第0維，但在優化裡面要把它放進去
 
在這邊我們測試一下 $(0,1)$ 跟 $(1,0)$ 這種很邊邊的點

看起來都符合我們的預測 分別都很接近1跟0

如果用 $(0.5,0.5)$這種看起來很中間的點呢？會不會得到0.5這個答案呢？

答案是不會，因為中性的表達在cross entropy 跟一般直覺的1D或是2D norm是不同的

```
0.9999757066132966
3.196385210777573e-05
0.5342474349324107
```

在來我們來看一下這個方式能不能延伸到多分類問題

假如今天有A、B、C、D，四類標籤，前面我們用logistic處理二分類能得到一個機率，也可以把它當作一個分數

1. 第一種處理方式叫做 One-versus-all decomposition：

今天可以用一對多的方式去拿到不一樣的分數

A可以對BCD，B可以對ACD，C可以對ABD，D可以對ABC

一個點就會拿到四個分數，我們取最大的那個當作是我們的分類

```
###logistic regression 實作
###我們來利用logistic實作多分類問題
#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt


#sample點用(而且它是保證均勻的取)
def sampling_circle(sample_size,r_sqaure,x_0,y_0):
    a = np.random.uniform(size=sample_size)*r_sqaure
    b = np.random.uniform(size=sample_size)
    x = np.sqrt(a) * np.cos(2 * np.pi * b)+x_0
    y = np.sqrt(a) * np.sin(2 * np.pi * b)+y_0
    return x, y
###模擬數據數量
mock_data_number_list=[10,30,30,30]
###弄一下label，後面可以對答案
for i in range(0,np.size(mock_data_number_list)):
  if i==0:
    real_label=np.zeros(mock_data_number_list[0])
  else:
    real_label=np.concatenate((real_label,np.ones(mock_data_number_list[i])*i))
#把mock data們整理一下，順便加入零的維度(W參數還要考慮一個常數)
#標籤我們後面在依據數量給
def data_generator(mock_data_number,r,x0,y0):
  ones=np.ones([1,mock_data_number])
  data1=np.array(sampling_circle(mock_data_number,r**2,x0,y0))
  label=-np.ones([1,mock_data_number])
  plt.scatter(data1[0,:],data1[1,:])
  data1=np.concatenate([ones,data1],0)
  data1=np.concatenate([data1,label],0)
  return data1

data1=data_generator(mock_data_number_list[0],0.2,0.3,0.7)
data2=data_generator(mock_data_number_list[1],0.2,0.7,0.7)
data3=data_generator(mock_data_number_list[2],0.2,0.7,0.3)
data4=data_generator(mock_data_number_list[3],0.2,0.3,0.3)

all_data=np.concatenate([data1,data2,data3,data4],1)

def sigmoid(x):
  result=1/(1+np.exp(-x))
  return result 

def cross_entropy_error(w,x,y):
  N=np.size(y)
  s=w.dot(x)
  error=np.sum(-np.log(sigmoid(y*w.dot(x))))/N
  return error

#GD實作
x=all_data[0:3,:]
N=np.sum(mock_data_number_list)
lr=0.1

count=0
for group_number in mock_data_number_list:
  count=count+group_number
  y=-np.ones(N)
  y[count-group_number:count]=1
  err_difference=10
  while_count=0
  ###w初始化
  w=np.zeros([1,3])
  ###GD 實作
  ###這邊寫一個當收斂就結束演算法的迴圈
  while err_difference>0.00001:
    ###其中sigmoid(-y*w.dot(x))可理解成是對(-y*x)的線性加權
    gd=np.sum(sigmoid(-y*w.dot(x))*(-y*x),1)/N
    w=w-lr*gd
    if while_count==0:
      while_count=while_count+1
    else:
      err_difference=err-cross_entropy_error(w,x,y)
    err=cross_entropy_error(w,x,y)
  print('err',cross_entropy_error(w,x,y))
  print('w',w)
  score=sigmoid(w.dot(x)).reshape(N,1)
  #print(sigmoid(w.dot(x)))
  ###算完就把W換算劃線
  w=w.flatten()
  plt_x=np.linspace(0.1,0.9,N)
  plt.plot(plt_x,-w[1]/w[2]*plt_x-w[0]/w[2])
  if count==mock_data_number_list[0]:
    score_array=score
  else:
    score_array=np.concatenate([score_array,score],1)
#print(score_array)
max_list=np.argmax(score_array,axis=1).reshape(N,1)
print(np.concatenate([max_list,real_label.reshape(N,1)],1))
```

然而這個演算法有一個缺陷就是當DATA數相較其他不足的時候容易被挖過去
**DATA不平衡的時候，預測會不準**
```
mock_data_number_list=[10,30,30,30]
```

```
[[3. 0.]
 [0. 0.]
 [3. 0.]
 [1. 0.]
 [0. 0.]
 [1. 0.]
 [1. 0.]
 [3. 0.]
 [1. 0.]
 [1. 0.]
 [1. 1.]
 [1. 1.]
......

```
