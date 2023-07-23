# 激勵函數與誤差函數

我們常常在ML裡面使用的兩個函數

1. 激勵函數：針對目標(標籤)的特化，我們要把眾多資料集寫成一個function，map到指定的區間與符合標籤 $y$ 的特性
2. 誤差函數：去衡量模型所預估的結果與目標的差距，我們有很多定義任意兩個set他們之間差距的方式

在這個條目裡面，我最終會來談談那些我們常見的激勵函數或誤差函數他們是怎麼被定義出來的

# 資訊(information)
首先，先來定義一下甚麼叫做資訊

是我今天講了這句話有幾個字叫做資訊嗎？

>>在非洲每過一個小時，就有3600秒不見了

那如果我今天說的這些話都是廢話，那可以叫作資訊嗎？

>>在非洲每過一個小時，就有一對黑人夫婦產下了一個白人嬰孩

像上面的資訊量好像就多了一些

**資訊的定義就是我們對於一個系統的可能性，或是無知做衡量**

## self-information
簡單的定義是我們對事件發生的機率取log，會取負號是因為機率都小於1
$$I\left( x \right) = -{\log _2}p\left( x \right)$$
因為我們會希望兩個獨立事件 $x_1$ 跟 $x_2$ 所定義的一起發生的訊息量是可以被相加，而且可以很直觀的連結到兩個獨立事件機率的相乘
$$I\left( {{x_1}} \right) + I\left( {{x_2}} \right) = {\log _2}{p_1}\left( {{x_1}} \right){p_2}\left( {{x_2}} \right)$$
當我們取log的時候，也相當於我們在對事件，對事情的可能性做編碼。

在熱力學內，我們想要衡量這個系統有多平衡，或者多秩序，我們就在乘上個波茲曼常數

$$S = {k_{\rm{B}}}\ln P$$

但都是同一個意思，我們對於這個系統有多無知，需要花費多少的力氣才能夠了解它
## Shannon Entropy
Shannon是一個在二戰時期有名的密碼學家，建構起基礎的資訊理論，並說明資訊基礎理論能夠應用於自然語言和計算機語言，做出了巨大的貢獻

人家的碩士論文就是現在教科書常見的布林代數應用於電子領域，是有跨時代的價值，在21歲就展現了才華

Shannon Entropy就是對self-information的期望值，也就是要達成這個平均上來說，系統最低的要求是多少

$$H\left( p \right) = E[I\left( p \right)] = \sum\limits_i {{p_i}{{\log }_2}{p_i}} $$

舉例來說，我今天來擲三個銅板，我來算一下出現正的時候的資訊量有多少

$$ - \left( {{1 \over 2}{{\log }_2}{1 \over 2} + {1 \over 4}{{\log }_2}{1 \over 4} + {1 \over 8}{{\log }_2}{1 \over 8}} \right) = {3 \over 2}$$

也就是說，平均我需要1.5個bits，我就可以傳達出三個銅板出現正的資訊

我們現在回過頭來看Shannon Entropy，改取個自然對數方便等等作分析(正統推導要用Lagrange Multiplier，這邊只是簡單感受一下)
$$H =  - \sum\limits_i {{p_i}\ln {p_i}} $$
$${{dH} \over {d{p_i}}} = \sum\limits_i {\ln {p_i} + } \sum\limits_i 1  = 0$$
$$N + \log \left( {{p_1}...{p_n}} \right) = 0$$
從這邊我們可以簡單的看到有一個可能性是Shannon Entropy有最大極值的時候，就是當這些p是常數，也就是uniform distribution

換句話說就是當Shannon Entropy最大的時候，就是系統變成最均勻的時候，也就是最無聊，死氣沉沉的時候。

## Cross entropy
既然我們有了衡量資訊的手段了，那麼我去衡量更多的系統的時候，例如說兩個set內有個別事件發生的機率: $p_i$ 跟 $q_i$ 我們要怎麼去衡量他們的差別呢？
$${H_{q,p}}\left[ { - \ln q\left( x \right)} \right] = -\sum\limits_i {{p_i}\ln {q_i}} $$

第一個概念，我們要介紹的是參照的想法

已實務上來說 $p_i$ 是目標分佈，而 $q_i$是模型的分佈，套入前面編碼的概念，我們今天把 $q_i$ 編碼後再參考 $p_i$ 作為加權

稍微提一下，Shannon Entropy一定是任何其他衡量資訊的下界
$${H_{q,p}}\left[ { - \ln q\left( x \right)} \right] \ge {E_q}\left[ { - \ln q\left( x \right)} \right]$$

也就是說，直到 $q_i$ 成為 $p_i$ ， ${H_{q,p}}=H_p$，這個時候就會有最小的Cross entropy

用Cross entropy還有一個特性，就是 $q,p$ 是可以彼此交換的，也就是說編碼 $q_i$ 後再參考 $p_i$ 跟 編碼 $p_i$ 再參考 $q_i$ 是等價的

概念上可以把這樣的差距當成是一種距離。

## Kullback-Leibler Divergence
剛剛我們提到了我們可以運用參照的手段來衡量兩個系統之間的差距，那能不能有更直接的方式，就是直接用減的
$$H\left( {q|p} \right) = {E_p}\left[ {\ln q\left( x \right) - \ln p\left( x \right)} \right]$$

在這裡可以看作以 $p_i$ 目標分佈為基準， $q_i$ 是模型的分佈，試圖去訓練 $q_i$ 成為 $p_i$

所有實務上有一點要注意的是， $H\left( {p|q} \right)$ 跟 $H\left( {q|p} \right)$ 是不同的，q跟p搞錯，意思會完全不同，所以在這裡我們會說這個差距是一種對比 (contrast)

再來KL Divergence也可以寫成這樣
$$H\left( {q,p} \right) - H\left( p \right)$$

就是Cross entropy跟Shannon Entropy的差距，所以這裡又串起來了！

## f-divergence
上述介紹了很多種衡量的方式，我們來寫一下更general的版本

$${D_f}\left( {q|p} \right) = {E_p}\left[ {f\left( {{{q\left( x \right)} \over {p\left( x \right)}}} \right)} \right] \qquad s.t \qquad f\left( 1 \right) = 0,\quad f \qquad is \qquad convex$$

令 ${q/p} = u$ 當 $f\left( u \right) = u\ln u$ 就是KL divergence啦

$$ - \sum\limits_i {{p_i}{{{q_i}} \over {{p_i}}}\left( { - \ln {{{q_i}} \over {{p_i}}}} \right)}  = \sum\limits_i {{p_i}\ln {{{p_i}} \over {{q_i}}}}  = H\left( {p|q} \right)$$

當 $f\left( u \right) = -\ln u$ 就是反過來的KL divergence

$$ - \sum\limits_i {{p_i}\left( { - \ln {{{q_i}} \over {{p_i}}}} \right)}  = \sum\limits_i {{p_i}\ln {{{q_i}} \over {{p_i}}}}  = H\left( {q|p} \right)$$

如果是 ${\left( {u - 1} \right)^2}$ 就叫做 $\chi$ Square Divergence，比較像是KL的開方版本

$$ - {\sum\limits_i {{p_i}\left( {{{{q_i}} \over {{p_i}}} - 1} \right)} ^2} =  - \sum\limits_i {{{{{\left( {{q_i} - {p_i}} \right)}^2}} \over {{p_i}}}} $$

如果是 $f\left( u \right) =  - \left( {u + 1} \right)\ln {{u + 1} \over 2} + u\ln u$ 叫做Jensen–Shannon divergence

開個坑，以後再介紹它





