# 多分類問題(multi-class regression)
我們從之前的條目了解到如何處理二元分類問題

PLA:

https://github.com/fluttering13/Machine-learning-base/blob/main/PLA-zh.md

Linear-regression:

https://github.com/fluttering13/Machine-learning-base/blob/main/Linear-regression.md

logistic-regression:

https://github.com/fluttering13/Machine-learning-base/blob/main/logistic-regression.md

現在我們想要推廣到多分類問題

假如今天有A、B、C、D，四類標籤，前面我們用logistic處理二分類能得到一個機率，也可以把它當作一個分數

1. 第一種處理方式叫做 One-versus-all decomposition：

今天可以用一對多的方式去拿到不一樣的分數

A可以對BCD，B可以對ACD，C可以對ABD，D可以對ABC

一個點就會拿到四個分數，我們取最大的那個當作是我們的分類

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/log2.png" width="500px"/></div>

```
###logistic regression OVA演算法實作
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

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/log3.png" width="500px"/></div>

```
mock_data_number_list=[10,30,30,30]
```
左行是分類器的結果，右行是原本的標籤

可以看到在只有10個點的分類預測一個歪掉
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
2. 接下來我們來介紹另外一個方式，叫做one-versus-one，來解決上面的問題

剛剛我們用一對多的方式，這會造成小群體的數據容易被忽略

這次我們用一對一的方式來評估，例如取AB,AC,AD,BC,BD,CD，共 $C_2^4$ 種

有點像是今天舉辦循環賽，最後積分高的人贏

在經過六場比賽後，如果是A的累加積分最高，則該點被視為是A分類

而這邊的累加積分是用投票機制，大於0.5就+1分，才能解決少分類點會被蓋過去的問題

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/log4.png" width="500px"/></div>

```
###logistic regression 實作
###OVO演算法
###我們來利用logistic實作多分類問題
#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
import itertools


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
N=np.sum(mock_data_number_list)
lr=0.1

#分組名單
combination_list=list(itertools.combinations(range(np.size(mock_data_number_list)), 2))
print('combination',combination_list)
score_all=np.zeros([N,np.size(mock_data_number_list)])
for i,j in combination_list:
  ###把特定label的根據分組名單挑出來
  index0=int(np.sum(mock_data_number_list[0:i]))
  index1=int(np.sum(mock_data_number_list[0:j]))
  x0=all_data[0:3,index0:index0+mock_data_number_list[i]]
  x1=all_data[0:3,index1:index1+mock_data_number_list[j]]
  y0=np.ones(mock_data_number_list[i])
  y1=-np.ones(mock_data_number_list[j])
  ###分類演算法用的x跟y是根據分組名單
  x=np.concatenate([x0,x1],1)
  y=np.concatenate([y0,y1])
  count=0
  err_difference=10
  while_count=0
  ###w初始化
  w=np.zeros([1,3])
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
  ###用算完的w去評估所有的點的分數
  x_all=all_data[0:3,:]
  score=sigmoid(w.dot(x_all)).reshape(N)
  ###這邊特別用for迴圈強調這邊是個投票機制
  ###如果直接把評分放上去會跟OVA有一樣的問題
  for k in range(0,N):
    if score[k]>0.5:
      score_all[k,i]=score_all[k,i]+1
    else:
      score_all[k,j]=score_all[k,j]+1
  ###算完就把W換算劃線
  w=w.flatten()
  plt_x=np.linspace(0.1,0.9,N)
  plt.plot(plt_x,-w[1]/w[2]*plt_x-w[0]/w[2])
max_list=np.argmax(score_all,axis=1).reshape(N,1)
print(np.concatenate([max_list,real_label.reshape(N,1)],1))
###畫太多線要限制一下範圍
plt.xlim(0,1)
plt.ylim(0,1)
```

以下的結果就看起來好很多了
```
[[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [1. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
```

# softmax來做多回歸問題
如果嫌二元性的分類太麻煩，也可以直接回到類似傳統做多元分類的方式，

在這邊可以使用softmax這個特殊函數來map到我們要分類到的那些標籤的機率

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/multi-1.png" width="500px"/></div>

```
###multi-class regression 實作

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
mock_data_number_list=[10,50,50,50]
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


x=all_data[0:k-1,:]
label_array_one_hot=one_hot(all_data[k-1,:],k).T


w=np.random.normal(size=(k, 3))
#3是常數0維+X跟Y兩個featute

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

###簡簡單單的優化
w,y,error,error_his,count=Optimzer_GD_2D_norm(0.01,w,x,label_array_one_hot)
print('train_round',count)
print('trained_result\n',np.vstack((np.argmax(y,axis=0),all_data[k-1,:].astype(int))))



###畫圖用
plt_x=np.linspace(0.1,0.9,100)
for i in range(0,k):
  plt.plot(plt_x,-w[i,1]/w[i,2]*plt_x-w[i,0]/w[i,2],label=str(i))
plt.legend(
    loc='lower right',
    fontsize=10,
    shadow=True,
    facecolor='#ccc',
    edgecolor='#000',
    title='test',
    title_fontsize=10)
```

其中，利用了$w$矩陣，去做參數化，第一列的 $w_1i$ 代表著對第一個one-hot向量的衡量

**也就代表著對第一個分類所進行的衡量**，我們可以稍微看一下藍色的點跟藍色的線能不能很好的隔開去知道現在分類了怎麼樣

