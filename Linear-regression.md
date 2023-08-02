# 線性回歸 (Linear-regression)

線性回歸常用於各式各樣預測性的問題：

在統計上，可以看這篇，它可以推廣到多變數多分類的問題

https://github.com/fluttering13/Causal-models/blob/main/SEM-model.md

在之前的PLA實作

https://github.com/fluttering13/Machine-learning-base/blob/main/PLA-zh.md

對於PLA實作中，我們常常遇到實作上去找到停止的點，是要跑幾個迴圈，等等之類的問題？

也許你就說，那我就等它收斂到一定的值就停下來。

但是往往它收斂的值都不太行，所以我們要多跑幾個不同的初始點，

對於這個問題在資訊科學是一個NP hard的問題。

那我們能不能有比較好的做法呢？但相對就會犧牲掉一些東西

相較於之前的0/1 error，在實務上很常使用方差(square error)
$${E_{sq}} = {1 \over N}{\sum\limits_{n = 1}^N {\left( {{W^T}{X_n} - {y_n}} \right)} ^2}$$
簡單推導一下梯度，就發現
$$\nabla {E_{sq}} = {1 \over N}\left( {2{X^T}XW - 2{X^T}y} \right)$$
方差有最小在
$${W_{lin}} = {\left( {{X^T}X} \right)^{ - 1}}{X^T}y$$
意思就是說，我們今天把這個問題變成了一個求解反矩陣的問題

那我們來看看代價是什麼？

由於Squre error恆大於0/1 error，我們可以拿到比較寬鬆的VC bound

意思是說相對於0/1 error，你可能會花費比較多的資源，如樣本數去找到比較好的bound

**但好處就是這是一個global的解，只要求個反矩陣就出來，當然如果遇到singular value那就沒辦法**

 如果關心singular value的問題也可以直接用梯度下降的方式進行數值求解，詳見

 https://github.com/fluttering13/Machine-learning-base/blob/main/multi-class-regression.md

壞處就是有時候方差不一定是最好的衡量 (因為方差不一定是量化"差距"概念最好的方式)

或是在實作上，方差優化的過程也可能較其他函數比較漫長


```
###linear regression 實作
#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
#用scipy的反矩陣的數值誤差比較小
import scipy
mock_data_number_each_group=100 #模擬數據數量

#sample點用(而且它是保證均勻的取)
def sampling_circle(sample_size,r_sqaure,x_0,y_0):
    a = np.random.uniform(size=sample_size)*r_sqaure
    b = np.random.uniform(size=sample_size)
    x = np.sqrt(a) * np.cos(2 * np.pi * b)+x_0
    y = np.sqrt(a) * np.sin(2 * np.pi * b)+y_0
    return x, y
ones=np.ones([1,mock_data_number_each_group])

#把mock data們整理一下，順便加入零的維度(W參數還要考慮一個常數)跟標籤
data1=np.array(sampling_circle(mock_data_number_each_group,0.2**2,0.3,0.7))
label=np.ones([1,mock_data_number_each_group])
plt.scatter(data1[0,:],data1[1,:])
data1=np.concatenate([ones,data1],0)
data1=np.concatenate([data1,label],0)

data2=np.array(sampling_circle(mock_data_number_each_group,0.2**2,0.7,0.3))
label=-np.ones([1,mock_data_number_each_group])
plt.scatter(data2[0,:],data2[1,:])
data2=np.concatenate([ones,data2],0)
data2=np.concatenate([data2,label],0)
all_data=np.concatenate([data1,data2],1)

###Linear Regression algorithm　（use for binary classification)
#def lin_reg(data):
x=all_data[0:3,:]
y=all_data[3,:]
x=x.T
y=y.T
x_psuedo_inverse=scipy.linalg.inv(x.T.dot(x)).dot(x.T)
w=x_psuedo_inverse.dot(y)

#算完就把W換算劃線
plt_x=np.linspace(0.1,0.9,mock_data_number_each_group)
plt.plot(plt_x,-w[1]/w[2]*plt_x-w[0]/w[2])
```
