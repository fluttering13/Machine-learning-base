# 函數們

我們常常在ML裡面使用許許多多的函數，這個條目主要用來整理如何定義與使用這些不同的函數

1. 激勵函數 (Activation function)：針對目標(標籤)的特化，我們要把眾多資料集經過成一個特別的function，map到指定的區間以符合標籤 $y$ 特性
2. 誤差函數 (Error function)：去衡量模型所預估的結果與目標的差距，我們有很多定義任意兩個set他們之間差距的方式
3. 正則化函數 (Regularzation function)：通常是在模型具有高參數，避免過擬合(overfitting)，而對誤差函數進行的修正。

在這個條目裡面，我最終會來談談那些函數式怎麼被定義出來的。

誤差函數：
https://github.com/fluttering13/Machine-learning-base/blob/main/information-and-entropy.md

如何正則化可以看這篇
https://github.com/fluttering13/Machine-learning-base/blob/main/Regularization.md

# 激勵函數 (Activation function)
在機器學習中，我們時常為了符合標籤的性質

如，二元分類中如果我們需要評分，我們會使用sigmoid來幫助我們map到0到1之間的機率

如果是多分類問題，那我們就會使用softmax，來給予多個分類的機率值

那可能就會自然的問說，有很多function都可以做到同樣的事情，那這些function它只是人為的被選出來？就這樣而已嗎？

背後式有些故事可以幫助我們了解，我們心中可能有什麼前提，才會去做這些選擇，而不僅僅是為了要符合標籤這個原因。

故事是這樣的，常常我們就會見到人們在談論著各式各樣的機率分佈：Gaussian分布、Bernoulli分布、Laplace分布、Lorentz分布

那有沒有一個方式可以系統化的告訴我這些分布的故事呢？

## 指數族分布(Exponential Family)
如其名，這是一個包山包海，可以涵蓋我們上述的那些分布們，這邊我們把它當作一套系統來幫助我們得到想要的資訊
$$p\left( {y|\eta } \right) = h\left( y \right)\exp \left( {{\eta ^T}T\left( y \right) - A\left( \eta  \right)} \right)$$

1. $\eta$ 就是那些雜七雜八的參數 (Natural Parameters)

2. $A\left( \eta  \right)$ 配分函數 (Log Partition Function) 通常是用在理上連結巨觀和微觀物理量的數學工具，主要就是對它進行微分，可以得到不同的物理量定義。

3. ${T\left( y \right)}$ 充分統計量 (Sufficient statistics) 簡單來說就是把那些關於y的訊息，把它打包在矩陣裡面。因為我們很常對分布進行微分，那些跟y無關的就是常數，不太重要。

4. $h\left( y \right)$ 基底量測 (base measure) 它就是我們把分布寫出來最後剩下的那些係數

先來看一下配分函數長什麼鬼樣
$$A\left( \eta  \right) = ln\int {h\left( y \right)exp\left( {{\eta ^T}T\left( y \right)} \right)} dy$$
在來看它微分之後的定義，它連結到分布裡面重要的兩個統計量，期望值跟變異數
$$E\left( {T\left( y \right)|\eta } \right) = {{\partial A\left( \eta  \right)} \over {\partial \eta }}$$

$$Var\left( {T\left( y \right)|\eta } \right) = {{{\partial ^2}A\left( \eta  \right)} \over {\partial {\eta ^2}}}$$

### 常態分布
$$P\left( y \right) = {1 \over {\sqrt {2\pi \sigma } }}\exp \left( { - {1 \over {2{\sigma ^2}}}{{\left( {y - \mu } \right)}^2}} \right)$$
我們把它整理一下
$${1 \over {\sqrt {2\pi } }}\exp \left( {\ln {1 \over \sigma } + {\mu  \over {{\sigma ^2}}}y - {1 \over {2{\sigma ^2}}}{y^2} - {1 \over {2\sigma }}{\mu ^2}} \right)$$
對應到的基底就是
$$h\left( y \right){\rm{ = }}{1 \over {\sqrt {2\pi } }}$$
參數是一個向量

$$
\eta=\left[\begin{array}{l}
\frac{\mu}{\sigma^2} \\
-\frac{1}{2 \sigma^2}
\end{array}\right]=\left[\begin{array}{l}
\eta_1 \\
\eta_2
\end{array}\right]
$$

充分統計量也是一個向量

$$T\left( y \right) = \left[ \matrix{
  y \hfill \cr 
  {y^2} \hfill \cr}  \right]$$

配分函數就有點複雜了，要稍微代數轉換一下，等等微分比較方便

$$A\left( \eta  \right) = \ln \sigma  + {1 \over {2{\sigma ^2}}}{\mu ^2} =  - {1 \over 2}\ln \left( { - 2{\eta _2}} \right) - {{{\eta _1}^2} \over {4{\eta _2}}}$$

那就開始對配分函數做微分

$$
\frac{\partial A(\eta)}{\partial \eta}=\frac{\partial A(\eta)}{\partial \eta_1}\left[\begin{array}{l}
1 \\
0
\end{array}\right]+\frac{\partial A(\eta)}{\partial \eta_2}\left[\begin{array}{l}
0 \\
1
\end{array}\right]=\left[\begin{array}{l}
-\frac{\eta_1}{2 \eta_2} \\
-\frac{1}{2 \eta_2}+\frac{\eta_1^2}{4 \eta_2^2}
\end{array}\right]=\left[\begin{array}{l}
\mu \\
\mu^2+\sigma^2
\end{array}\right]
$$

它告訴我們，要擬合一個Normal Distribution需要兩個統計量，如果有學過動差(momentum)，就是那個用來檢測兩個隨機變數的東東，這邊對應到的分別是一階動差跟二階動差

如果我今天知道這些參數了，又告訴你分布是什麼，那能不能反推回去說平均是什麼？變異數是什麼？

在機器學習裡面常常使用一些特殊的函數，會選用這些函數其實都來自於，我們心中可能有某個分布，我今天只是把這些參數去map到這些分布的平均上面

這又叫作反向參數映射 Inverse parameter mapping

# 激勵函數的懶人包
這邊根據wiki整理了一下
https://en.wikipedia.org/wiki/Exponential_family

二元分類用sigmoid

多元分類用softmax

短時間的隨機事件(Poisson distribution) 用 ${e^\eta }$

常態分布(normal distribution) 可以用剛剛導過的

$$
\left[\begin{array}{l}
-\frac{\eta_1}{2 \eta_2} \\
-\frac{1}{2 \eta_2}+\frac{\eta_1^2}{4 \eta_2^2}
\end{array}\right]
$$

可靠性分析和壽命檢驗 (Weibull distribution) 用 ${\left( { - \eta } \right)^{ - {1 \over k}}}$  
k=1，它是指數分布； k=2時，是Rayleigh distribution (k也可以當作參數喔喔喔)

布朗運動可以用 $-1 \over \eta$

## 這邊應該沒什麼差


以下是就算你沒用function應該也沒差，什麼都不加的泛用性其實也很高

社會上的財富分配(Pareto distribution) 用 $-1- \eta$，但就是差個常數應該沒差

(chi-squared distribution) 用 $2\eta+2$
