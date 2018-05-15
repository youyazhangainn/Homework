import os
import numpy as np
import matplotlib.pyplot as plt


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array


def cal(array):
    c = []
    for item in array:
        a = item.split(',')
        b = []
        for i in range(len(a)-1):
            b.append(float(a[i]))
        c.append(b)
    return c


def test(file_path):
    array = read_lines(file_path)
    return cal(array)


filepath = 'magic04.txt'
c = test(filepath)
Y=np.array(c)
Y = np.mat(Y)
X=np.mean(Y,axis=0)
print(X)


a=np.ones((19020,1))
D=a*X
K=Y-D
Z1=(K.T*K)/19020
print(Z1)

sum=K[:,0]*K[:,0].T
for i in range(1,9):
    sum=sum+(K[:,i]*K[:,i].T)
Z2=sum/19020
print(Z2)

C=np.cos(Y[:,0],Y[:,1])
print(C)
N=range(0,C.shape[0])
f1=plt.figure(1)
p1=plt.scatter(N,C)
plt.xticks(N)
plt.show()


