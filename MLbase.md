# Machine-learning
>哈利，你居然用我的魔法來對付我的魔法

來說說一些 從數據科學角度出發去看ML 去看這些oracle 不能僅僅只是煉丹爐或者魔法

## 機器學習(What is machine learning?)
在切入主題之前，我們來定義一下什麼是學習？

是把所有問題的答案都背下來？ 是學習嗎？

很明顯，從小到大如果你有朋友是這樣，也許它會過得很辛苦。

上述的過程也只能說是一個資料庫，什麼是學習呢，在於使用這些資料的能力。

你會看到這些資料是怎麼彼此建立起他們的連結，

例如說我現在資料裡面有 大象、猴子、企鵝

我們試著去建立起他們的屬性，透過觀察，我們可以知道這些都是哺乳類動物，他們都在動物園裡面

這些資料跟資料之間可以透過函數的連結，幫助我們去找到他們之間的關係

如果重建這些關係，我們就說是學習的一個過程

但往往我們可能不太清楚現實世界的關係是長什麼樣子，所以我們會有一個hypothesis set

例如說我們把動物這樣的概念當成一個hypothesis set

數據上你今天知道了 大象、猴子、企鵝

所以你試圖去刻劃動物這樣的集合，

`舉例來說，你可以去看他們共有的器官`

`他們有 眼睛、鼻子、嘴巴......等等。`

`得知現有的數據 (大象、猴子、企鵝)`

`在有目標函數的情況 (共有的器官)`

`試圖去找到一個最佳化的參數 (眼睛、鼻子、嘴巴)`

`來描述你有興趣的模型 (動物)`


所以很自然的，我們會問說，要怎麼去建立起一個模式去找到這樣的函數關係？
$$f(X) = Y$$

>Laplace 準確定性概念(Laplace's quasi-deterministic conception)
>>所有的隨機模型(古典)，都可以用函數模型來進行表示

在現今常用的方式我們試圖去寫下一個類線性的方程
也就是所謂的類神經元網路
$$\alpha W{\rm{X + b = Y}}$$

$$
\alpha 
\left[
\begin{matrix}
    w_{11} & ... & w_{1n} \\\\
    ... & ... & ... \\\\
    w_{m1} & ... & w_{mn}
\end{matrix}
\right]
\left[
\begin{matrix}
x_{1}\\\\
...\\\\
x_{m}
\end{matrix}
\right]
+
\left[
\begin{matrix}
b_{1}\\\\
...\\\\
b_{m}
\end{matrix}
\right]
\leq
\left[
\begin{matrix}
y_{1}\\\\
...\\\\
y_{m}
\end{matrix}
\right]
$$

其中 $\alpha$ 可以是一個非線性的方程
在這邊我們只是寫下他們的形式，其中會遇到各式各樣的問題
如：
優化怎麼做？
目標方程要怎麼選？
要選什麼樣的 $\alpha$ ?

再來我們從一些數據科學的角度來進行切入

# 機器學習能不能學
從上述的雛形，我們可以知道我們是想要找到一個方程或是特定的機率分布來了解我們想知道的hypothesis

以下我們漸漸去放入一些根據統計假設，逐漸得出我們需要多少的數據

## 有限數據 (Finite statistics)
往往我們是想要追求理想上的模型

但我們今天只能夠從真實數據拿到的是有限的數量

這邊會衍生一些問題

我要拿多少的數據才能能夠很好的去回答一個問題？

此處我們來引入一個Hoeffding不等式來說明

假設我們從一個母體作抽樣，抽樣過程都是獨立同分布(i.i.d) 總共抽N筆資料，

把母體的結果，所佔的比例為 $\mu$，

把取樣集的結果，所佔的比例為 $\nu$

我們令這兩的比例的誤差為 $\epsilon$

$$P[{\rm{|}}\mu {\rm{ - }}\nu {\rm{| < }}\varepsilon ] \ge 2\exp \left( { - 2{\varepsilon ^2}N} \right)$$

假如今天取了1000筆的資料，我們允許這個誤差可以到7%

我們會得到機率為0.00011090319886435389 約為0.01%

我們取了10000筆，這個機率可以降到10^-43

**只要我們採樣(sample)了足夠多的數量N，一定是更有信心去做預測的**

上述是我們只對一個hypothesis做思考，但假如我們今天有M個hypothesis呢？從union bound可以給出
$$P[{\rm{|}}\mu {\rm{ - }}\nu {\rm{| < }}\varepsilon ] \ge 2 M \exp \left( { - 2{\varepsilon ^2}N} \right)$$
這是因為我們今天對 $M$ 個hypothesis之間沒有做限制，而這會導致這個bound在N很大的時候會發散掉，

例如說在PLA裡面，$M$ 就可以有無窮多線去拆分不同多個標籤，也就是 $m$ 有無窮大的選擇

**小的 $M$，可以讓我們做到 母體錯誤率 $\mu$ 跟 取樣集錯誤率 $\nu$ 很接近，但太少的選擇 $M$ 不一定找到很小的錯誤率**

**大的 $M$，看起來做不到 母體錯誤率 $\mu$ 跟 取樣集錯誤率 $\nu$ 很接近，於是我們要引入限制，讓 $P$ 還是可以收斂**

### 修正後的公式 Vapnik-Chervonenkis bound
$$P[{\rm{|}}\mu {\rm{ - }}\nu {\rm{| > }}\varepsilon ] \le 4{M_H}\left( {2N} \right)\exp \left( { - {1 \over 8}{\varepsilon ^2}N} \right)$$
我們先給出這個結論，或是先給出一個對於去修正公式的直覺
1. 我們希望 $N$ 是隨著自然指數遞減
2. 我們希望引入成長函數 $M_H$ ，且它是多項式成長的
3. 我們上述都是母體與取樣集的比較，實際上我們是測試集與取樣集的比較

後續我們會花很大的篇幅在講成長函數 $M_H$ ，

最後得出 VC bound (Vapnik-Chervonenkis bound)
## Generation theory

上述我們想要加一些限制，這些限制從那裡來呢？

我們先回到 $M$ 的意義是什麼，它代表，這個參數空間要區別的可能性有多少


### 1D positive ray
假設我們今天做PLA，但是是在一維，現在遊戲規則是從某點開始向右取正，向左取負
$$h\left( x \right) = sign\left( {x - a} \right)$$

那現在有多少可能性呢？

一個點

$$\left( 0 \right),\left( 1 \right)$$

兩個點
$$\left( {0,0} \right),\left( {0,1} \right),\left( {1,1} \right)$$
三個點
$$\left( {0,0,0} \right),\left( {0,0,1} \right),\left( {0,1,1} \right),\left( {1,1,1} \right)$$

$N$ 個點共 (從N+1個區間選點)
$$M\left( N \right) = C_1^{N + 1} $$

### 1D positive interval
再來我們來看看另外一個遊戲規則是在區間內是正，其他區間為負
$$h\left( x \right) =  + 1\ \quad if \qquad x \in \left[ {l,r} \right]$$


一個點
$$\left( 0 \right),\left( 1 \right)$$
兩個點
$$\left( {0,0} \right),\left( {0,1} \right),\left( {1,0} \right),\left( {1,1} \right)$$
三個點
$$\left( {0,0,0} \right),\left( {0,0,1} \right),\left( {0,1,0} \right),\left( {1,0,0} \right),\left( {0,1,1} \right),\left( {1,1,0} \right),\left( {1,1,1} \right)$$


$N$ 個點共 (從N+1個區間選兩個，並加上都是零的這種組合)
$$M\left( N \right) = C_2^{N + 1} + 1 = {N^2}/2 + N/2 + 1$$

### 2D convex
遊戲規則是PLA的參數集合為平面凸集合，而所有的資料群在平面上也是個凸集
$$x,w \in Convex \quad set$$

$N$ 個點共 (我們可以找到凸多邊形，把這些資料包起來)
$$M\left( N \right) = {2^N}$$

### 小總結
$M(N)$ 我們把它稱作是成長函數Growth function，也就是我們想知道的限制是什麼

第一個限制來自於二元性資訊量上的侷限
$$M\left( N \right) \le {2^N}$$
第二個限制來自於遊戲規則上造成的侷限，這邊我們可以討論一下

在positive ray $N=1$ 和 positive interval $N=1,2$ 它們都滿足 $M\left( N \right) = {2^N}$，達成二元資訊上最大的滿足

**我們就會說在這個狀況下被shatter**

在positive ray $N=2$ 和 positive interval $N=3$，從這邊開始不滿足

**N=k時，是個break point**

在這邊我們可以有一個觀察或者是猜想，稍微觀察一下

當點越多時，越容易受到遊戲規則所影響

試想，如果有越難受到影響的， $k$ 越大的，代表成長函數成長的越快越接近 $2^N$

我們發現break point的地方 $k$ 從直覺上，跟成長函數是呈正相關的，

而對於這種效應，我們會期許成長的速度是以多項式的方式

$$M\left( N \right) = O\left( {{N^{k - 1}}} \right)$$

### Bounding function

從以上，我們可以反過來看，也是我們有興趣的，當給予 $k$ 跟 $N$ 的時候， $M$ 最多最多可以是多少？

所以我們可以定義一個 Bounding function

$$B\left( {N,k} \right)$$

聽起來要反過來看，好像是一件很困難的事情，因為我們好像要去猜它對應到的遊戲規則是什麼？

但其實k是一個足夠的資訊，讓你知道限制是 當有N個點的時候，裡面任意k個得不能夠完全shatter，

也就是說任意裡面k個不能出現 $2^k$ 排列的情況

聽起來很抽象我們來舉一個例子，來找找看在k=2時，M(N)最大可以到多少

當N=1時，因為只有一個點 < $k$，肯定shatter

$$M(1)=2,B(1,2)=2$$

當N=2時，這時候有兩個點，因為k=2時，要找到不shatter的例子，就像是上面我們做過的positive ray，直接4-1就行

$$M(2)=3,B(2,2)=3$$

當N=3時，這時候有三個點，所以我們要討論看看

首先，一定可以有三個

$$\eqalign{
  & \left( {0,0,0} \right)  \cr 
  & \left( {0,0,1} \right)  \cr 
  & \left( {0,1,0} \right) \cr} $$

我們再加上第四個來看 $1,0,0$

$$\eqalign{
  & \left( {0,0,0} \right)  \cr 
  & \left( {0,0,1} \right)  \cr 
  & \left( {0,1,0} \right)  \cr 
  & \left( {1,0,0} \right) \cr} $$

這個時候檢查任意兩行，都不會發生shatter，所以可以

但加到第五個的時候，無論如何都會shatter，這個時候我們可以確定
$$M(3)=4,B(2,3)=4$$

**這是因為追加新的列的時候，假如新的列是其他列的線性組合，這個時候就會shatter**
$${x_{k + 1}} = {c_k}{x_k} + {c_{k - 1}}{x_{k - 1}}......{c_1}{x_1}$$
以上我們可以把它當作是定義，但其實bounding function是有公式的


### 快速總結公式
上述我們已經演示完了bounding function，這邊我們直接跳過繁雜的證明

這些不同的bounding function有以下回歸關係

$$B\left( {N,k} \right) \le B\left( {N - 1,k} \right) + B\left( {N - 1,k - 1} \right)$$

總結一下

$$B\left( {N,k} \right) = \sum\limits_{i = 0}^{k - 1} {C_i^N} $$
這個bounding function其實非常的符合直覺，也就是把 $N$ 個點的取樣到 $k$ 再加總起來，當滿足二項是定理 $(1+1)^(N)$時就是shatter！

**有了這個我們就可以知道，生長函數有一個上界，而且還是多項式的**
$${M_H}\left( N \right) \le B\left( {N,K} \right) = \sum\limits_{i = 0}^{k - 1} {C_i^N}  \le {N^{k - 1}}$$

### Vapnik-Chervonenkis bound
上述我們已經考慮完了生長函數的修正，現在我們還要把母體 $\mu$ 改為測試集 $\mu'$，再讓總數變成 $2N$

意思是我們分N個去測試集，另外N個去取樣集，所以Hoffdining不等式左邊改成
$$P[{\rm{|}}\mu {\rm{ - }}\nu '{\rm{| > }}{\varepsilon  \over 2}]$$
假如我們今天在意的是取樣集跟 取樣集與測試集的平均，也就是抽出來資料後不放回。
$$P[{\rm{|}}\mu {\rm{ - }}{{\nu  + \mu '} \over 2}{\rm{| > }}{\varepsilon  \over 4}]$$
右式
$$2{M_H}\left( {2N} \right)2\exp \left( { - 2{{\left( {{\varepsilon  \over 4}} \right)}^2}N} \right)$$
我們整理一下可以拿到
$$P[{\rm{|}}\mu {\rm{ - }}{{\nu  + \mu '} \over 2}{\rm{| > }}\varepsilon ] \le 4{M_H}\left( {2N} \right)\exp \left( { - {1 \over 8}{\varepsilon ^2}N} \right)$$
就是VC bound！
### VC dimension
我們把VC dimsension定義成
$$k - 1$$

再重新回顧一下之前的例子

positive ray的時候 $d_{vc}=1$ 因為我們的遊戲規則裡 可以選 $a$ 的位置

positive interval的時候 $d_{vc}=2$ 因為我們的遊戲規則裡 可以選 $l,r$ 兩個位置

這告訴我們說，**VC dimensio的概念其實就是在這些參數空間中有多少的自由度**

再來我們回顧一下　VC bound
$$P[{\rm{|}}\mu {\rm{ - }}{{\nu  + \mu '} \over 2}{\rm{| > }}\varepsilon ] \le 4{M_H}\left( {2N} \right)\exp \left( { - {1 \over 8}{\varepsilon ^2}N} \right)$$
我們把 ${M_H}\left( {2N} \right)$ 直接用 ${2N}^d_{VC}$ 取代它的上限，
$$\sqrt {{8 \over N}\ln {{4{{\left( {2N} \right)}^{{d_{VC}}}}} \over \delta P}}  \le \varepsilon $$
我們會說這時一個權衡模型複雜度的一個方式 (the penalty for model complexity)

同樣我們也可以寫下泛化偏差 (generalization error)
$$\nu ' - \varepsilon  \le \nu  \le \nu ' + \varepsilon $$
在這邊我們簡化一下代號讓 $nu '$ 代表那些非取樣集(或是訓練集)

**隨著模型複雜度增加，直覺上我們會認為表現會越好，但實際上從這個penalty就告訴你其中同時也會造成錯誤率上升**
這就告訴我們，在實際操作上我們要選適合複雜度模型去處理我們所遇到的問題

我們來代個數字
$$\varepsilon  = 0.1, \qquad \delta  = 0.1, \qquad {d_{VC}} = 3$$
$$N = {10^5}, \qquad \delta P \le 1.65 \times {10^{ - 38}}$$
看起來 $d_{vc}=3$ 就需要10000筆資料才能做到，似乎是非常多，但注意，這是我們放寬了很多限制才拿到的bound
1. 我們使用Hoffdining 不等式：無論如何的機率分布(distribution) 或是 目標函數(target function)，我們都能考慮進去
2. 再來任意的data我們也可以考慮進去
3. 我們做了一個多項式的近似，不是直接用 ${M_H}$ 來看
**所以好處就是因為這個bound很寬鬆，所以它很好進行估計**
### 總結
所以我們要進行機器學習，需要有以下流程

1. 有足夠的數據量 $N$
2. 建立起訓練模型，其中， $d_{VC}$ 必須要是有限的，而且要適中
3. 評估好對Error的側量
4. 利用演算法最佳化，找到最佳參數，讓Error足夠低
