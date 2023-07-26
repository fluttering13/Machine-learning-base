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

## 足夠統計量 (Sufficient statistic)
在這邊先引入一個名詞
