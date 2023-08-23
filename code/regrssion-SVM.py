###SVM原型是針對二分類問題
###註解部分拿掉是soft margin

#首先我們需要一些模擬數據
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
#sample點用(而且它是保證均勻的取 )
def sampling_circle(sample_size,r_sqaure,x_0,y_0):
    a = np.random.uniform(size=sample_size)*r_sqaure
    b = np.random.uniform(size=sample_size)
    x = np.sqrt(a) * np.cos(2 * np.pi * b)+x_0
    y = np.sqrt(a) * np.sin(2 * np.pi * b)+y_0
    return x, y
###模擬數據數量
mock_data_number_list=[50,50]
verify_data_number_list=[50,50]
###種類
k=np.size(mock_data_number_list)
###feature number
n=2
#
N=sum(mock_data_number_list)

#標籤我們後面在依據數量給
def data_generator(mock_data_number,r,x0,y0,label):
  ones=np.ones([1,mock_data_number])
  data1=np.array(sampling_circle(mock_data_number,r**2,x0,y0))
  label_array=label*np.ones([1,mock_data_number])
  plt.scatter(data1[0,:],data1[1,:])
  data1=np.concatenate([ones,data1],0)
  data1=np.concatenate([data1,label_array],0)
  return data1

data1=data_generator(mock_data_number_list[0],0.2,0.3,0.7,1)
data2=data_generator(mock_data_number_list[1],0.2,0.7,0.7,-1)

all_data=np.concatenate([data1,data2],1)


def one_hot(a,k):
  a=a.astype(int)
  return np.eye(k)[a]


def compare_two_array_count(answer,learn_label):
    error_count=0
    for i in range(len(answer)):
        if answer[i]!=learn_label[i]:
            error_count=error_count+1
    return error_count
###print(cvxpy.installed_solvers())

x=all_data[1:n+1,:]
y=all_data[-1,:].reshape([1,N])
w=cvx.Variable([1,n])
b=cvx.Variable()
#xi=cvx.Variable([1,N])
obj=cvx.Minimize(cvx.square(cvx.norm(cvx.vec(w))))
#obj=cvx.Minimize(cvx.square(cvx.norm(cvx.vec(w)))+C*cvx.sum(xi))
constraints=[]
constraints.append(cvx.vec(cvx.multiply(y,w@x+b))>=1)
# constraints.append(xi>=0)
prob = cvx.Problem(obj,constraints)
prob.solve()
print('sum over w',prob.value)
print('w',w.value)


###plt
plt_x=np.linspace(0.1,0.9,100)
plt.plot(plt_x,-w.value[0,0]/w.value[0,1]*plt_x-b.value/w.value[0,1])
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()


