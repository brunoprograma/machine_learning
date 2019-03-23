import numpy as np
import matplotlib.pyplot as plt
import sys

def h(X, theta):
	return 1 / (1 + np.e ** -(X.dot(theta.T)))


def J(X, y, theta):
	m = X.shape[0]
	y_hat = h(X, theta)
	erro = (-y * np.log(y_hat) - (1-y) * np.log(1-y_hat)).sum(0)
	return erro / m


def GD(X, y, theta, alpha, niters):
	m = X.shape[0]
	cost = np.zeros((niters,1))
	print('iteração:')
	for k in range(0, niters):
		print(' ',k,end='')
		y_hat = h(X, theta)
		erro = ((y_hat-y) *X).sum(0)/m
		theta -= (alpha * (erro))
		cost[k] = J(X, y, theta)
		print('\r\r\r\r\r\r',end='')
	return (cost, theta)

def featureScaling(X):
  X=X-np.min(X,0)
  den=np.max(X,0)-np.min(X,0)
  return X/den

if len(sys.argv) < 5:
	print('Usage %s <dataset> <# of iterations> <alpha> <delimiter>'%sys.argv[0])

f=sys.argv[1]
niters=int(sys.argv[2])
alpha=float(sys.argv[3])
delim=sys.argv[4]
## delete \n at the end of the string
data=np.genfromtxt(f,delimiter=delim)
## split the string into the values
# now data is a list of lists
X=data[:,:-1]
X=np.array(X,dtype=float)
#### Inicio da área que é permita alguma mudança
X=featureScaling(X)
X=np.insert(X,0,1,axis=1)
#### Fim da área que é permitida alguma mudança
y=data[:,-1]
y=np.reshape(np.array(y,dtype=int),(len(y),1))

Theta=np.zeros((1,X.shape[1]))

nsize=int(X.shape[0]*.7)

Xtr=X[:nsize,:]
Xte=X[nsize:,:]
ytr=y[:nsize]
yte=y[nsize:]

c,t=GD(Xtr,ytr,Theta,alpha,niters)

y_hat=np.round(h(Xtr,t))
print('Home made learner:')
print('   Taxa de acerto (treino):', np.mean(y_hat==ytr))
y_hat=np.round(h(Xte,t))
print('   Taxa de acerto (teste):', np.mean(y_hat==yte))
plt.plot(c)
plt.show()
#------------
from sklearn import linear_model
r=linear_model.LogisticRegression()
ytr=np.ravel(ytr)
r.fit(Xtr,ytr)
yte=np.ravel(yte)
y_hat=r.predict(Xtr)
print('Home made learner:')
print('   Taxa de acerto (treino):', np.mean(y_hat==ytr))
y_hat=r.predict(Xte)
print('   Taxa de acerto (teste):', np.mean(y_hat==yte))

