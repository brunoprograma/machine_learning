import numpy as np
import sys

import matplotlib.pyplot as plt

def f(X,T):
    return X.dot(T.T)

## returns the cost (residual error) for predicted y
def J(X,y,T):
  m=X.shape[0]
  y_hat=X.dot(T.T)
  cost=np.sum((y_hat-y)**2)/(2*m)
  return cost

## finds the "good" values for the weigths (Theta)
## Gradient descent (optmization function)
def GD (X,y,T,alpha,niters):
  m=X.shape[0]
  cost=np.zeros((niters,1))
  for i in range(0,niters):
    y_hat=X.dot(T.T)
    error=((y_hat-y)*X).sum(0)/m
    T=T-(alpha*(error))
    cost[i]=J(X,y,T)
  return T,cost
  
def featureScaling(X):
  X=X-np.min(X,0)
  den=np.max(X,0)-np.min(X,0)
  return X/den

def featureNormal(X):
	return (X-np.mean(X,0))/np.std(X,0)  
 
def printY(y1,y2,r):
	for i in range(len(y1)):
		print(y1[i],y2[i],r[i])
 
#### Main program
if len(sys.argv) < 3:
	print('Usage: %s <dataset> <delimiter>'%sys.argv[0])
	exit(0)
fname=sys.argv[1]
delim=sys.argv[2]
print(fname, delim)

try:
	data=np.genfromtxt(fname,delimiter=delim)
except:
	print('Error opening',fname)
	exit(0)
##
X=data[:,:-1]
X=featureScaling(X)
X=np.insert(X,0,1,axis=1) ## insert 1 column as the first column into X
y=np.array([data[:,-1]]).T
d=X.shape[1]
###
tr=int(X.shape[0]*.75) # 75% for training and 25% for testing
Xtr=X[:tr,:]
ytr=y[:tr,:]
Xte=X[tr:,:]
yte=y[tr:,:]
#Theta=np.array([np.random.rand(X.shape[1])])
Theta=np.zeros((1,d))
Theta,cost=GD(Xtr,ytr,Theta,0.1,100000)
y_hat=f(Xte[:10,:],Theta)
printY(y_hat,yte[:10],abs(y_hat-yte[:10]))
print('-----')
from sklearn import linear_model
r=linear_model.LinearRegression()
r.fit(Xtr,ytr)
y_hat=r.predict(Xte[:10,:])
printY(y_hat,yte[:10],abs(y_hat-yte[:10]))

plt.plot(cost)
plt.show()
