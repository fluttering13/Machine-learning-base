# 感知器學習演算法(Perceptron learning algorithm)
這篇是 台大開放式課程Hsuan-Tien Lin Machine Learning Foundations (機器學習基石) 的學習筆記

在早期的很多演算法的東西都是從生物的機制獲得靈感而來的，其中這個感知器
便是從神經元的概念所啟發。神經元中的突觸，需要生物電壓超過一定的閥值才能啟動，所以是二元性的

<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/450px-Complete_neuron_cell_diagram_zh.png" width="500px"/></div>
(圖來源自wikipedia)

日常也有很多二元性質的問題，例如：

這家餐廳值不值得去吃？

這招衣服是不是現下流行的？

現在的房地產是不是合投資？

這個客戶是否能夠承擔風險？
......

很多時候我們需要去針對一個問題去做一個評估，為此我們需要做迴歸分析或是建立起二元性的模型來進行預測
$${\rm{WX = Y}}$$
在這簡單的線性模型中，其中 向量 $W$ 是我們需要優化的參數，並透過輸入向量 $X$ 去預測向量標籤 $Y$
$${{sign(WX) = Y}}$$
此處感知器意味著二元性分類

在此再引入取正負進行二元性分類的線性方程，當然以上的雛形還能夠進行推廣，我們從向量的角度來切入要如何實現優化

首先我們想做的事情是想要讓 $X$ 透過一個線性函數變成 $Y$，其中有一個方式就是去做內積

內積的就是 $X$ 在 $Y$ 或是 $Y$ 在 $X$ 上的投影，意思是說我想要知道 $X$ 與 $Y$ 他們的多相關，而且單位向量做內積最多就是1，是有邊界的

所以我只要想辦法讓 $W$ 的更新是與 $X$ 和 $Y$ 的內積有關就好

$$W' = W + XY$$

好的演算法會是可以收斂的，一定是在有限步數內可以得到最優解，

我們先假設這個演算法後來會收斂我們會得到一個向量 $W_f$，跟再某個步驟 $t$ 所得到的 $W_t$

我們內積一下來看看他們的相關性
$${W_f}{W_{t + 1}} = {W_f}\left( {{W_t} + YX} \right) = {W_f}{W_t} + {W_f}YX$$
從第二個項 $Y{W_f}X$，它的意義是最終的 $Y$ 標籤與 $W_fX$ 的投影，如果 $Y$ 與 $W_fX$ 很像，這個內積值一定會大於等於0

也就是說，當如果整個演算法做完，所有的分類都處理好分成兩塊，即

$${Y_n} = sign({W_f}{X_n})$$
那我們一定會有

$${\min_n}Y{W_f}X \ge 0$$
注意，如果後者成立，但前者不一定成立，但我們會說這是一個線性可分(Linear Separable)的條件

而從內積完的結果，我們可以知道隨著演算法更新， $W_{t+1}$ 逐漸往我們想要的 $W_{f}$ 邁進

但這僅僅只是從內積 ${W_f}{W_{t + 1}}$

造成內積值增加有兩個可能：兩個向度長度增加或者是兩個向量夾角變小

(通常在演算法前者的影響其實還好，因為隨時都可以重整化，但這個簡單的算法還是可以看一下它的長度變化是怎麼樣)
$${\rm{|}}{W_{t + 1}}{|^2} = |{W_t} + YX{|^2} = |{W_t}{|^2} + 2{W_t}YX + |YX{|^2}$$

唯一會產生負值的地方在 $2{W_t}{Y}{X}$，也就是還沒分類完的時候

從這個式子可以看出 ${|W_{t + 1}|}{^2}$ 在增長的速度沒這麼快，所以引導著內積增長的是兩者的角度逐漸縮小

意思是愈來越逼近我們想要的 $W_f$，同時上式也告訴我們 $W_{t + 1}$ 和 $W_{t}$的回歸關係

剛剛我們看的是兩者的內積，再讓我們排除掉長度所造成的影響，我們來看看單位向量之間的內積

$${{{W_T}{W_f}} \over {|{W_T}||{W_f}|}} = {{\left( {{W_0} + T\min YX} \right){W_f}} \over {\sqrt {|{W_0}{|^2} + T\max YX} |{W_f}|}}$$

分式上面內積可從 $W_T=W_0+TYX$ 代入定義所給出，就像前面我們做過的內積計算。

分式下方可由上述的回歸關係給出，註：也如果從 ${\rm{|}}{W_T}{{\rm{|}}^2} = {W_0}^2 + {T^2}YX + 2T......$ 給出的是 $T^2$ 不是最靠近的bound，要從回歸關係找

我們令 $W_0=0$ 再令 $|Y{|^2} = 1$ 標籤化簡一下我們可以得到 

$${{\sqrt T \min \left( {Y{W_f}X} \right)} \over {\max X}}$$

我們得到了一個 $\sqrt T$，再稍微移項整理一下

$${{\max X} \over {\min Y{{{W_f}} \over {{\rm{|}}{W_f}{\rm{|}}}}X}} \ge T$$
意思是更新步數 其實在這個演算法下其實是有一個上界在，也就是說最多最多你需要這麼多步就可以找到最佳值

但實際上你不知道 $W_f$ 是什麼，只是在理論上有個上界

當然上述我們必須在線性可分的前提下才能找到最佳解

```
###naive PLA 實作
import numpy as np
import matplotlib.pyplot as plt

#可以線性分割的dataset
dataset = np.array([
((1, -0.4, 0.3), -1),
((1, -0.3, -0.1), -1),
((1, -0.2, 0.4), -1),
((1, -0.1, 0.1), -1),
((1, 0.9, -0.5), 1),
((1, 0.7, -0.9), 1),
((1, 0.8, 0.2), 1),
((1, 0.4, -0.6), 1)])

#初始化
w=np.zeros(3)

for i in range(0,8):
  #條件判斷，sign相反就執行演算法
  while np.sign(sum(dataset[i,0]*w))!=dataset[i,1]:
    w=w+np.array(dataset[i,0])*dataset[i,1]

#這邊只是畫圖用
coordinate=np.array(dataset[:,0])
x1=[]
x2=[]
y1=[]
y2=[]
count=0
for i,j,k in coordinate:
  count=count+1
  if count>4:
    x1.append(j)
    y1.append(k)
  else:
    x2.append(j)
    y2.append(k)  

line = np.linspace(-2,2)
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.plot(line,line*-w[1]/w[2]+-w[0]/w[2])
```
<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/PLA1.png" width="500px"/></div>

但是native PLA在以下的例子會壞掉
<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/PLA2.png" width="500px"/></div>

```
###non native PLA 實作
#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
lambda_noise=0.1 #生成模擬數據用的雜訊參數
initial_m_noise=0.01 #在演算法中挑初始點的雜訊參數
mock_data_number=10 #模擬數據數量

#生成模擬數據用的東東
bias_1=0.3 
bias_2=-0.3
solpe_1=0.3
solpe_2=0.5
X=np.arange(1,mock_data_number+1,1)
x1=solpe_1*X+np.random.random(mock_data_number)*lambda_noise*mock_data_number+bias_1*mock_data_number
x1=x1/max(x1) #讓模擬數據的Y軸限制在0到1
x2=solpe_2*X+np.random.random(mock_data_number)*lambda_noise*mock_data_number+bias_2*mock_data_number

x2=x2/max(x2) #讓模擬數據的Y軸限制在0到1
plt.scatter(X,x1)
plt.scatter(X,x2)
x0=np.zeros(mock_data_number)

#整理一下這些模擬用數據
all_data=np.concatenate([np.array([X,x1,np.ones(mock_data_number)]),np.array([X,x2,-np.ones(mock_data_number)])],1)
#初始化PLA內要訓練的參數w
w=np.zeros([2])

#讓模擬數據隨機排列一下
all_data=np.random.permutation(all_data.T).T

goal_count=mock_data_number*2

error_count=0
#pocket演算法用的記憶空間
pocket=[]
#mock_data_number*2

#找演算法初始點很好的方式是根據平均值來取
mu1=np.average(all_data[0])
mu2=np.average(all_data[1])
for i in range(0,mock_data_number*2):
  #把平均值跟隨機一點取斜率，並允許雜訊，轉換成m的參數
  random_initial_compared_point_index=np.random.randint(low=1,high=mock_data_number*2)
  delta_1=all_data[0,random_initial_compared_point_index]-mu1
  delta_2=all_data[1,random_initial_compared_point_index]-mu2
  w=np.array([-delta_2*(1+initial_m_noise*(np.random.random())),delta_1*(1+initial_m_noise*(np.random.random()))])
  error_count=0
  #PLA算法
  while np.sign(all_data[:-1,i].dot(w))!=all_data[2,i]:
    w=w+all_data[2,i]*all_data[:-1,i]
  #pocket算法，做完一輪PLA看看這個W是不是適用其他的m
  for i in range(0,mock_data_number*2):
    if np.sign(all_data[:-1,i].dot(w))!=all_data[2,i]:
      error_count=error_count+1
  #pocket算法，把結果存起來
  pocket.append(np.append(w,error_count))
  print(np.append(w,error_count))
  index=np.argmin(pocket,axis=0)[2]
  #pocket算法，取最好的m當作下一輪新的w
  w=pocket[index][0:2]

print(w)
plt.plot(X,-w[0]/w[1]*X)
```

調整完了演算法 以下這個看起來還可以接受了！
<div align=center><img src="https://raw.githubusercontent.com/fluttering13/Machine-learning-base/master/pic/PLA3.png" width="500px"/></div>








