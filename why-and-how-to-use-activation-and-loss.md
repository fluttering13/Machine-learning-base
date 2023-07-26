# 激勵函數與誤差函數

我們常常在ML裡面使用的兩個函數

1. 激勵函數：針對目標(標籤)的特化，我們要把眾多資料集寫成一個function，map到指定的區間與符合標籤 $y$ 的特性
2. 誤差函數：去衡量模型所預估的結果與目標的差距，我們有很多定義任意兩個set他們之間差距的方式

在這個條目裡面，我最終會來談談那些我們常見的激勵函數或誤差函數他們是怎麼被定義出來的

誤差函數還可以詳見這篇
https://github.com/fluttering13/Machine-learning-base/blob/main/information-and-entropy.md

# 概似估計 (Likelihood function)
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

這邊是我們做最大概似估計最重要的asummption，假如這個世界就是穩定運轉，那我們從母群體抽樣的過程也是穩定的

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
