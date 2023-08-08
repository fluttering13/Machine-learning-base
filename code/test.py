import torch
import numpy as np

if torch.cuda.is_available():
   cuda0 = torch.device('cuda:0')
   t1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64, device=cuda0)
# 也可以寫成
cuda0 = torch.device('cuda', 0)
# 使用第二個 GPU
cuda1 = torch.device('cuda', 1)
# 使用 CPU
#cpu = torch.device('cpu')

# 建立隨機數值的 Tensor 並設定 requires_grad=True
x = torch.randn(2, 3, requires_grad=True)
y = torch.randn(2, 3, requires_grad=True)
z = torch.randn(2, 3, requires_grad=True)
# 計算式子
a = x * y
b = a + z
c = torch.sum(b)
# 計算梯度
c.backward()
# 查看 x 的梯度值
print(x.grad)