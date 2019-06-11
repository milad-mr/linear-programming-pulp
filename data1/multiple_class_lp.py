from pulp import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Create the 'prob' variable to contain the problem data

from sklearn.datasets import load_iris
data = load_iris()
# data.target[[10, 25, 50]]

# print(list(data.data))

X_train, X_test, y_train, y_test = train_test_split(
     data.data, data.target, test_size=0.4, random_state=42)
# # print("a",X_train, X_test, y_train, y_test)

# #swiss roll
# data = np.loadtxt('SwissRoll.txt', usecols=range(4))
# np.random.shuffle(data)
# data = data[:1000]
# rate = .8                                     
# total = data.shape[0]
# X_train, X_test, y_train, y_test = data[:int(rate*total),:-1], data[int(rate*total):,:-1], data[:int(rate*total),-1].astype(np.int32), data[int(rate*total):,-1].astype(np.int32)

data = np.loadtxt('data_multi_train.txt', usecols=range(3))
np.random.shuffle(data)

# data = data[]
rate = .1    
               
total = data.shape[0]
X_train, _, y_train, _ = data[:int(rate*total),:-1], data[int(rate*total):,:-1], data[:int(rate*total),-1].astype(np.int32), data[int(rate*total):,-1].astype(np.int32)
data = np.loadtxt('data_multi_test.txt', usecols=range(3))
np.random.shuffle(data)

# data = data[]
rate = 0      
               
total = data.shape[0]
_, X_test, _, y_test = data[:int(rate*total),:-1], data[int(rate*total):,:-1], data[:int(rate*total),-1].astype(np.int32), data[int(rate*total):,-1].astype(np.int32)

y_train -= 1
y_test -= 1
# ys = x**2
x = X_train
ys = y_train 
print("x is", ys)
# 9/0

centers_count = len(x)
v = [LpVariable("Weight" + str(i), cat='Continuous') for i in range(centers_count)]
# centers = np.random.choice(x,centers_count, replace = False)
centers = x
from math import exp
# def g(xl, center):
#     lamda = .5
#     return exp(-lamda*(xl - center)*(xl - center))
lamda = 1
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
class_count = 5
# for ysi in range(len(ys)):
#     print ("hi")
u = [[1 if ys[i] == c  else 0 for c in range(class_count)] for i in range(len(ys))]  #this line is depend on problem
one_minus_u = [[0 if ys[i] == c else 1 for c in range(class_count)] for i in range(len(ys))]
print("u is", u)

v1 = [[LpVariable("v1c" + str(c) + "l" + str(l), lowBound = 0, cat='Continuous') for c in range(class_count)] for l in range(len(x))]
v2 = [[LpVariable("v2c" + str(c) + "l" + str(l), lowBound= 0, cat='Continuous') for c in range(class_count)] for l in range(len(x))]
ep1= [LpVariable("ep1l" + str(l),lowBound = 0, upBound=10, cat='Continuous') for l in range(len(x))]
# ep2= [LpVariable("ep2s" + str(i),lowBound = 0, cat='Continuous') for i in range(len(x))]
print("v is", pulp.lpSum(pulp.lpSum(v1) + pulp.lpSum(v2) + ep1))
v = [[v1[i][j] - v2[i][j] for j in range(len(v1[0])) ] for i in range(len(v1))]
print("v is:", v)
y = np.dot(g,v) #g.dot(v)
print("Y is", y)
# model += pulp.lpSum(pulp.lpSum(v1) + pulp.lpSum(v2) + ep1)
model += pulp.lpSum([v1[i][j] + v2[i][j] for j in range(class_count) for i in range(len(x))]) - pulp.lpSum( [ep1[i] for i in range(len(x))] ), "objective"
# print("u[0]:", (one_minus_u[0] * y[0]) )
for i in range(len(x)):
    model += pulp.lpSum(u[i] * y[i]) - pulp.lpSum(one_minus_u[i] * y[i]) == ep1[i]
model += pulp.lpSum([v1[i][j] - v2[i][j] for i in range(len(x)) for j in range(class_count)]) == 1
# for j in range(class_count):
#      
# The problem is solved using PuLP's choice of Solver
model.solve()
# The status of the solution is printed to the screen
print("Status:", LpStatus[model.status])

# print("model is", model)

# Each of the variables is printed with it's resolved optimum value
w = np.zeros((len(x),class_count))
# print("w is", w)
for v in model.variables():
    print(v.name, "=", v.varValue)
    
    if (v.name[:2] == "v1"):
     #    print("name is ", v.name[v.name.index('c') + 1: v.name.index('l')])
        w[int(v.name[v.name.index('l')+1:])][int(v.name[v.name.index('c') + 1: v.name.index('l')])] += v.varValue
    if (v.name[:2] == "v2"):

        w[int(v.name[v.name.index('l')+1:])][int(v.name[v.name.index('c') + 1: v.name.index('l')])] -= v.varValue
print("w is", w)

# print("g",g)
x = X_test
g = [[exp(-lamda*(sum((xl - center)*(xl - center))**.5)) for center in centers]  for xl in X_test]
yst = y_test
mse = 0
y_preds = []
for i in range(len(yst)):
    result_prob = np.dot(g[i],w)
    result_y = result_prob.argmax() 
    y_preds.append(result_y)
    print("x:",x[i], "probs::", result_prob,"y predict",result_y, "y real", yst[i],result_y == yst[i])
# print("g is :", g)
print("Status:", LpStatus[model.status])
from sklearn.metrics import confusion_matrix
print("conf matrix:", confusion_matrix(yst, y_preds))
from sklearn.metrics import accuracy_score
print("accuracy:", accuracy_score(yst, y_preds))

zero_w = 0
for wi in w:
    for wj in wi:
          if wj ==0:
               zero_w += 1
print("zero count is:", zero_w, "total count:", len(w) * class_count)
print("Status:", LpStatus[model.status])