from pulp import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Create the 'prob' variable to contain the problem data
import time

from sklearn.datasets import load_iris
data = load_iris()
# # print(list(data.data))

from sklearn import datasets
# swissX, swissY = datasets.make_swiss_roll(n_samples=400, noise=0.0, random_state=None)

# print(xx, yy)
data = np.loadtxt('2classtrain.txt', usecols=range(3))
np.random.shuffle(data)

# data = data[]
rate = 1
               
total = data.shape[0]
X_train, _, y_train, _ = data[:int(rate*total),:-1], data[int(rate*total):,:-1], data[:int(rate*total),-1].astype(np.int32), data[int(rate*total):,-1].astype(np.int32)
data = np.loadtxt('2classtest.txt', usecols=range(3))
np.random.shuffle(data)

# data = data[]
rate = 0      
               
total = data.shape[0]
_, X_test, _, y_test = data[:int(rate*total),:-1], data[int(rate*total):,:-1], data[:int(rate*total),-1].astype(np.int32), data[int(rate*total):,-1].astype(np.int32)


# import matplotlib.pyplot as plt
# # print("data is", X_train[:,1])      
# # 9/0  
# plt.plot(X_train[:,0], X_train[:,1], 'o')
# plt.show()
# plt.plot(ys, 'o', linewidth=3)
# print(data)
# 9/0
# X_train, X_test, y_train, y_test = train_test_split(
#      swissX, swissY, test_size=0.33, random_state=42)
# # print("a",X_train, X_test, y_train, y_test)


# from keras.datasets import cifar10

# (X_train, y_train), (x_test, y_test) = cifar10.load_data()
# ys = x**2
print(y_train)
x = X_train
ys = y_train #[-1 if yi == 0 else 1 for yi in y_train]

# print(ys)
# 8/0
centers_count = len(x)
v = [LpVariable("Weight" + str(i), cat='Continuous') for i in range(centers_count)]
# centers = np.random.choice(x,centers_count, replace = False)
centers = x
from math import exp
# def g(xl, center):
#     lamda = .5
#     return exp(-lamda*(xl - center)*(xl - center))
lamda = .1
start_time = time.time()
model = LpProblem("The Miracle Worker", LpMinimize)
# g = [[exp(-lamda*sum((xl - center)*(xl - center))) for center in centers]  for xl in x]
g = [[exp(-lamda*(sum((xl - center)*(xl - center))**.5)) for center in centers]  for xl in x]
# ep = [LpVariable("ep" + str(i), cat='Continuous') for i in range(len(x))]

# y = [LpVariable("y" + str(i), cat='Continuous') for i in range(len(x))]

# g = exp
# print("y is", y)
# # print("gg is:", gg)
#
# print("v is :", v)
# print("ez",[g(1,cent) for cent in centers] * v)

# print(prob)
# for xl in x:
#     model +=
# print("ys+ep:", ys+ep)

v1 = [LpVariable("v1s" + str(i),lowBound = 0, cat='Continuous') for i in range(len(x))]
v2 = [LpVariable("v2s" + str(i),lowBound = 0, cat='Continuous') for i in range(len(x))]
ep1= [LpVariable("ep1s" + str(i),lowBound = 0,upBound=10, cat='Continuous') for i in range(len(x))]
# ep2= [LpVariable("ep2s" + str(i),lowBound = 0, cat='Continuous') for i in range(len(x))]
# print("before v")
v = [v1[i] - v2[i]  for i in range(len(v1))]
# y = np.dot(g,v1) - np.dot(g,v2) #g.dot(v)
# y = np.dot(g, v) 
#optimizeeeeeeD :)
y = [pulp.lpSum(g[di][i] * v[i] for i in range(len(x)))  for di in range(len(x))]
start2_time = time.time()
# print("Y is", pulp.lpSum([v1[i] + v2[i] - ep1[i] for i in range(len(x))] ))
model +=  260*pulp.lpSum(v1+v2) + pulp.lpSum(ep1) #pulp.lpSum([v1[i] + v2[i] - ep1[i] for i in range(len(x))] )
for i in range(len(x)):
    model += ys[i] * y[i] == 1 + ep1[i]
# model += pulp.lpSum(v1 + v2) == 1
# The problem is solved using PuLP's choice of Solver
# print("before solve")
solve_time = time.time()
model.solve()
end_solve_time = time.time()
# The status of the solution is printed to the screen
print("Status:", LpStatus[model.status])


# Each of the variables is printed with it's resolved optimum value
w = np.zeros(len(x))
for v in model.variables():
    print(v.name, "=", v.varValue)
    if (v.name[:3] == "v1s"):
        w[int(v.name[3:])] += v.varValue
    if (v.name[:3] == "v2s"):
        w[int(v.name[3:])] -= v.varValue
# print("w is", w)

# print("g",g)
x = X_test
y_preds = []
non_zero_nodes = [i for i in range(len(g))]
err_nodes = []
g = [[exp(-lamda*(sum((xl - center)*(xl - center))**.5)) for center in centers]  for xl in X_test]
yst = y_test #[-1 if yi == 0 else 1 for yi in y_test]
sign = lambda a: 1 if a>0 else -1 if a<0 else 0
for i in range(len(g)):
    y_prob = np.dot(g[i],w)
    result_y = sign(y_prob)
    y_preds.append(result_y)
    if (result_y != yst[i]):
            err_nodes.append(i)
#     print("x:",x[i], "y prob:", y_prob,"y:",result_y , "y real", yst[i], result_y == yst[i])

print("w is", w)

print("Status:", LpStatus[model.status])
from sklearn.metrics import confusion_matrix
print("conf matrix:", confusion_matrix(yst, y_preds))
from sklearn.metrics import accuracy_score
print("accuracy:", accuracy_score(yst, y_preds))

zero_w = 0
for i in range(len(w)):
    if w[i] ==0:
        zero_w += 1
        non_zero_nodes.remove(i)
print("zero count is:", zero_w, "total count:", len(w), "non zero g:", len(non_zero_nodes))
print("Status:", LpStatus[model.status])
print("solve time:",end_solve_time - solve_time, "total time :", end_solve_time - start_time, "total time variable:", end_solve_time - start2_time, )
print("pulp time:", model.solutionTime)
import matplotlib.pyplot as plt
# print("data is", X_train[:,1])      
# 9/0  
# plt.plot(zero_nodes, 'o')
# plt.plot(err_nodes, 'o')
# plt.plot(X_train[:,0], X_train[:,1], 'o')
plt.plot(x[:,0], x[:,1], 'o')
plt.plot(X_train[non_zero_nodes,0], X_train[non_zero_nodes,1],'*')
plt.plot(x[err_nodes,0], x[err_nodes,1],'^')
plt.show()