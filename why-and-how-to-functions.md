# 激勵函數與誤差函數

我們常常在ML裡面使用的兩個函數

1. 激勵函數：針對目標(標籤)的特化，我們要把眾多資料集寫成一個function，map到指定的區間與符合標籤 $y$ 的特性
2. 誤差函數：去衡量模型所預估的結果與目標的差距，我們有很多定義任意兩個set他們之間差距的方式

在這個條目裡面，我最終會來談談那些我們常見的激勵函數或誤差函數他們是怎麼被定義出來的

誤差函數還可以詳見這篇
https://github.com/fluttering13/Machine-learning-base/blob/main/information-and-entropy.md

# 正規化 (Regularization)
我們時常在做的任務就是老師去教學生，

要如何的去教：在這個條目，我們會先介紹 最大概似估計 (Maximum Likelihood Estimation) 我們想要去學習的目標是甚麼要怎麼定義下來

學壞了怎麼辦：再來是如何用運用貝式理論去做後驗機率的修正，我們稱之為正規化

## 概似估計 (Likelihood function)
Likelihood function的意思是估計統計模型中的參數的函數，

假如今天我有骰骰子的結果D，我今天想要去建構一個函數，由不同的角度或是高度 (parameter) $\theta$，藉由快門$m$去紀錄這些過程 ( $m$ 叫做hyper parameter與這些參數獨立無關，只是扮演更新算法的角色)
$$p\left( {D|m,\theta } \right)$$

## 最大概似估計 (Maximum Likelihood Estimation)
以下介紹一下常見使用的概念，最大概似估計 (Maximum Likelihood Estimation)，它跟上面那篇文章內的推導非常的像，只是加入了一些哲學思考

以預測一個骰子出來的面來舉例，我就需要讓這個函數盡量的逼近我真實的資料 $D$

由不同的角度或是高度 $\theta$ 我可以組出一組輸入的資料 $x_i$，輸出資料標記為 $y_i$

這個過程我們叫作 最大概似估計 (Maximum Likelihood Estimation)

最佳化的參數可以寫成，意味者我們想找到最大參數化的那組參數
$${\theta _{MLE}} = {1 \over N}\arg {\max _\theta }\sum\limits_i {p\left( {{y_i}|{x_i},\theta ,m} \right)} $$

我們來對這些機率分佈編碼一下，同時也方便加總，也就是取log

$${\theta _{LMLE}} =  - \arg {\min _\theta }\sum\limits_i {{1 \over N}\ln p\left( {{y_i}|{x_i},\theta ,m} \right)} $$

所以我們就可以寫成期望值，**只是全部條件機率的權重都是相同的**，在這邊就可以當作是CROSS ENTROPY的定義
$${\theta _{LMLE}} =  - \arg {\min _\theta }E\left( {\ln p\left( {{y_i}|{x_i},\theta ,m} \right)} \right) = \arg {\min _\theta }H\left( {p,q} \right)$$

這邊是我們做最大概似估計最重要的asummption，假如這個世界就是穩定運轉，那我們從母群體抽樣參數 $\theta$ 的過程也是穩定的

### Bernoulli distribution
我們舉簡單的二元性分佈為例子，連續投擲硬幣的結果為正面 $N_ +$ 與 反面 $N_-$ 的次數，所給出的機率分佈為

$$f\left( {\theta ,{N_ + },{N_ - }} \right) = {\theta ^{{N_ + }}}{\left( {1 - \theta } \right)^{{N_ - }}}$$

它的最大概似估計地方在

$${{df} \over {d\theta }} = 0 \Rightarrow \theta_{MLE}={N_+ \over (N_+ + N_-)}$$

所以直覺上我們對二元性分類問題，例如說三個正面與兩個反面出現最大的概率，就是在正面所佔的比重，也就是0.6的地方，很符合直覺

### Normal distribution
那常見的正態分佈呢？
$$f\left( {D|x} \right) = {1 \over {\sqrt {2\pi {\sigma ^2}} }}\exp \left( { - {{\left( {x - \mu } \right)}^2}/{{\left( {2\sigma } \right)}^2}} \right)$$
微分一下就會發現，就是很直觀的在平均數上
$${{df} \over {d\theta }} = 0 \Rightarrow {\theta _{M{\rm{LE}}}}{\rm{ = }}\mu $$

## 最大後驗估計 Maximum A Posterior (MAP)
在前面MLE提到了，所有從母群體抽樣參數 $\theta$ 都是相同的，


但仔細想想這個應該是還蠻奇怪的一件事情，極端狀況的參數 $\theta_e$ 跟普通狀況下的參數 $\theta_n$ 應該是不同的。

在實際分類問題或是運用機器學習可能會遇到一個問題就是，當模型的參數越來越大的時候，預測的狀況反而不準的。這個名詞就叫作過擬合overfitting。

感覺上就是你教一個小孩教太久，導致他無法很好的梳理與理解，知識，而在它上面發生衝突導致發瘋。

這是因為模型增長參數，相對的也會引進complexity造成的noise。

換句話說，就是當模型增長之後，允許更多的function 或是 probility distribution可以對應到同一個要預測的data

因為我們無法做到以下的事情

1. 我們無法做到無窮取樣，完美的詮釋data

2. 我們可能對於構成data的條件一無所知

也就是說，我們要想辦法去作出限制，並給出參數的權重呢？

在這裡我們運用到貝式學派的精神，詳見Bayes詮釋
https://github.com/fluttering13/Causal-models/blob/main/Introduction-Zh.md

**也就是我們去假設一下參數的分佈，給出這些參數的限制**

我們習以為之的normal distribution, Lorenz distribution, Bolzman distribution，當我說們它是特定distribution的時候

其實就已經把先驗的概念，過往經驗發生了甚麼，給考慮了進去

我們重寫一下條件機率，利用鍊式法則可以整理一下，在這裡可以看到不同版本的貝氏公式
$$p\left( {\theta |D,m} \right) = {{p\left( {D,m|\theta } \right)p\left( \theta  \right)} \over {p\left( {D,m} \right)}} = {{p\left( {D|m,\theta } \right)p\left( {m|\theta } \right)p\left( \theta  \right)} \over {p\left( {D|m} \right)p\left( m \right)}} = {{p\left( {D|m,\theta } \right)} \over {p\left( {D|m} \right)}}p\left( {\theta |m} \right)$$

1. ${p\left( {D|m,\theta } \right)}$ 就是前面說的likehood
2. $p\left( {\theta |m} \right)$ 稱作先驗機率，也就是我們透過以往的經驗，或者是觀察而得到的機率
3. 左式的　$p\left( {\theta |D,m} \right)$　就是後驗機率，經由先驗機率，和其他的觀察所結合後更新得到的新的機率

所以我們要對後驗機率找到最佳的參數，
$${\theta _{MAP}} = \arg {\max _\theta }p\left( {\theta |D,m} \right) = \arg {\max _\theta }{{p\left( {D|m,\theta } \right)} \over {p\left( {D|m} \right)}}p\left( {\theta |m} \right)$$
老樣子取個log
$${\theta _{MAP}} = \ln \sum\limits_i {p\left( {{y_i}|{x_i},m,\theta } \right) + } \ln p\left( {\theta |m} \right) - \sum\limits_i {\ln p\left( {{y_i}|{x_i},m} \right)} $$
超參數應該跟最後找到最好的參數是無相關的獨立事件，所以刪掉 $\sum\limits_i {\ln p\left( {{y_i}|{x_i},m} \right)} $ 。換句話說就是怎麼拍照，你都不會影響到真實物體的美麗。
$${\theta _{MAP}} = \ln \sum\limits_i {p\left( {{y_i}|{x_i},m,\theta } \right) + } \ln p\left( {\theta |m} \right)$$
跟MAE相比就只是多了一項修正項，在機器學習裡面這叫作Regularization

### uniform distribution
$$\ln p\left( {\theta |m} \right)=c$$
差一個常數在微分的時候就會被微掉了，就會回到MAE的例子

### Normal distribution
$$p\left( {\theta |m} \right) = {1 \over {\sqrt {2\pi {\sigma ^2}} }}\exp \left( { - {{\left( {\theta  - \mu } \right)}^2}/{{\left( {2\sigma } \right)}^2}} \right)$$
MAE寫一下
$${\theta _{MAP}} = \ln \sum\limits_i {p\left( {{y_i}|{x_i},m,\theta } \right) + } {1 \over {2{\sigma ^2}}}{\theta ^2}$$
新增的修正項有一個新的名字，它就叫作L2 Regularization，所以它**有用到Normal distribution的假設**

### Laplace distribution
$$p\left( {\theta |m} \right) = {1 \over {2b}}\exp \left( { - \left| \theta  \right|/b} \right)$$
MAE寫一下
$${\theta _{MAP}} = \ln \sum\limits_i {p\left( {{y_i}|{x_i},m,\theta } \right)}  - \left| \theta  \right|/b$$
新增的修正項有一個新的名字，它就叫作L1 Regularization，所以它**有用到Laplace distribution的假設**
