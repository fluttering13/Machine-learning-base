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
其中$\alpha $一個特殊的函數可以把任意的數值map到區間0到1，這邊所選取的是sigmoid函數
$${1 \over {{e^{ - wx}} + 1}}$$
至於這個函數是怎麼訂出來的，詳見

https://github.com/fluttering13/Machine-learning-base/blob/main/why-and-how-to-use-activation-and-loss.md

激勵函數如果選擇sigmoid，那很自然的，我們就會使用cross-entropy作為誤差函數，衡量標籤 $y$ 與模型預測 ${w^T}{x_n}$之間的對比(Contrasts)

$${E_{ce}} = {\min _w}{1 \over N}\sum\limits_n^N { - \ln \alpha \left( {{y_n}{w^T}{x_n}} \right)} $$

由於這邊我們有非線性函數，所以我們直接用數值求解的方式找到誤差函數極小的值的地方
$${E_{ce}} = {\min _w}{1 \over N}\sum\limits_n^1 { - \ln \alpha \left( {{y_n}{w^T}{x_n}} \right)} $$
接著我們就是在數學上寫下它的梯度，丟給演算法更新

$$\nabla {E_{ce}} = {1 \over N}\sum\limits_n^1 { - \ln \alpha \left( {{y_n}{w^T}{x_n}} \right)} \left( {{y_n}{x_n}} \right)$$

這邊的優化演算法叫做梯度下降(gradient descent)，在數值方法叫做尤拉方法，其實就是根據周邊的斜率去找極值的概念
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
#GD 實作
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

