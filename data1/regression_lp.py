from pulp import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Create the 'prob' variable to contain the problem data

from sklearn.datasets import load_iris
data = load_iris()
data.target[[10, 25, 50]]

# print(list(data.data))


data = np.loadtxt('data_regression.txt', usecols=range(4))
X_train, _, y_train, _ = train_test_split(data[:,:-1], data[:,-1], test_size=0.8, random_state=42)
# print("xtrain old is:", y_train)

_, X_test, _, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=.99, random_state=42, shuffle=False)

# np.random.shuffle(data)

# # data = data[]
# X_train, X_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4, random_state=42, shuffle=False)
# data = np.loadtxt('data_regression.txt', usecols=range(4)
# rate = .2  
# print("xtrain", X_train, y_train)
# 9/0
# total = data.shape[0]
# data_train = data[:int(rate*total),:-1]
# X_train, X_test, y_train, y_test = data[:int(rate*total),:-1], data[int(rate*total):,:-1], data[:int(rate*total),-1].astype(np.int32), data[int(rate*total):,-1].astype(np.int32)
# y_test  = y_train
# X_test = X_train
# print("a",X_train, X_test, y_train, y_test)

# from keras.datasets import boston_housing

# (X_train, y_train), (X_test, y_test) = boston_housing.load_data()

#1d regression

# data =  np.loadtxt('shuffled_data.txt',delimiter=',')
# Xx = np.reshape(data[:,0], (len(data), 1))
# Yy =data[:,1]
# X_train, X_test, y_train, y_test = train_test_split(
#      Xx, Yy, test_size=0.33, random_state=42)
# print("xtrain is:", y_train)
# a=9 / 0
# x_train = X[0:143,:]
# x_validation = X[143:191,:]
# x_test = X[192:239,:]
# y_train = Y[0:143]
# #validation = evaluation
# y_validation = Y[143:191]
# y_test = Y[192:239]
# x = np.linspace(0,20, 100)
# ys = x**2
x = X_train
ys = y_train

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
g = [[1.0] + [exp(-lamda*(sum((xl - center)*(xl - center))**.5)) for center in centers]  for xl in x]
print("g is:", g[1])
# 8/0
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
v1 = [LpVariable("v1s" + str(i),lowBound = 0, cat='Continuous') for i in range(len(x) + 1)]
v2 = [LpVariable("v2s" + str(i),lowBound = 0, cat='Continuous') for i in range(len(x) + 1)]
ep1= [LpVariable("ep1s" + str(i),lowBound = 0, cat='Continuous') for i in range(len(x))]
ep2= [LpVariable("ep2s" + str(i),lowBound = 0, cat='Continuous') for i in range(len(x))]
# print("-1ep", ep)
# v = v1 - v2
# y = np.dot(g,v1) - np.dot(g,v2) #g.dot(v)
v = [v1[i] - v2[i]  for i in range(len(v1))]
# y = np.dot(g,v1) - np.dot(g,v2) #g.dot(v)
y = np.dot(g, v)
print("Y is", y)
model += pulp.lpSum(v1 + v2 + ep1 + ep2), "Optimization Function"
# model += pulp.lpSum(v1 + v2) == 1
for i in range(len(x)):
    model += y[i] + ep1[i] - ep2[i] == ys[i]

# model.solve()
# model.
# def u(pred,real):
#     if pred == real :
#         return 1
#     else:
#         return 0;
#
# # Create problem variables
# x=LpVariable("Medicine_1_units",0,None,LpInteger)
# y=LpVariable("Medicine_2_units",0, None, LpInteger)
#
# # The objective function is added to 'prob' first
# prob += 25*x + 20*y, "Health restored; to be maximized"
# # The two constraints are entered
# prob += 3*x + 4*y <= 25, "Herb A constraint"
# prob += 2*x + y <= 10, "Herb B constraint"
#
# # The problem data is written to an .lp file
# prob.writeLP("MiracleWorker.lp")
#
# # The problem is solved using PuLP's choice of Solver
model.solve()
# The status of the solution is printed to the screen
print("Status:", LpStatus[model.status])
# Output=
# Status: Optimal

# Each of the variables is printed with it's resolved optimum value
w = np.zeros(len(v1))
for v in model.variables():
    print(v.name, "=", v.varValue)
    if (v.name[:3] == "v1s"):
        w[int(v.name[3:])] += v.varValue
    if (v.name[:3] == "v2s"):
        w[int(v.name[3:])] -= v.varValue
# print("w is", w)

x = X_test
g = [[1.0] + [exp(-lamda*sum((xl - center)*(xl - center))) for center in centers]  for xl in X_test]
yst = y_test
mse = 0
y_preds = []
# print("g",g)
for i in range(len(yst)):
    y_pred = np.dot(g[i],w)
    # "x:",x[i],
    print( y_pred - yst[i], "y predict:", y_pred, "y real", yst[i],)
    y_preds.append(y_pred)
    mse += (y_pred - yst[i]) * (y_pred - yst[i])
# # Output=
# # Medicine_1_units = 3.0
# # Medicine_2_units = 4.0

print("mse is:", mse / len(yst))
import matplotlib.pyplot as plt
plt.plot(yst, 'o', linewidth=3)
plt.plot(y_preds, 'o')

zero_w = 0
for wi in w:
    if round(wi) ==0:
        zero_w += 1
print("zero count is:", zero_w, "total count:", len(w))
print("Status:", LpStatus[model.status])
plt.show()