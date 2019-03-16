import numpy as np
wine=np.loadtxt("wine.data",delimiter=",",dtype='float')
X=wine[:,0:13]
##################
X = np.insert(X, X.shape[1], X[:,1]**2, axis=1)
X = np.insert(X, X.shape[1], X[:,2]**3 + X[:,2]**2, axis=1)
X = np.insert(X, X.shape[1], X[:,7]**2, axis=1)
X = np.insert(X, X.shape[1], np.e ** X[:,5], axis=1)
X = np.insert(X, X.shape[1], 1.3 ** abs(X[:,8]), axis=1)
X = np.insert(X, X.shape[1], abs(X[:,2]) / abs(X[:,3]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,2]) / abs(X[:,3])) % abs(X[:,3]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,2]) % abs(X[:,3])) / abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,4]) / abs(X[:,2])) % abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,4]) % abs(X[:,2])) / abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,4]) % abs(X[:,2])) % abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,7]) % abs(X[:,2])) / abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,7]) % abs(X[:,2])) % abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,7]) % abs(X[:,2])) / abs(X[:,7]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,7]) % abs(X[:,1])) % abs(X[:,6]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,10]) % abs(X[:,2])) / abs(X[:,6]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,2]) % abs(X[:,1])) % abs(X[:,6]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,12]) % abs(X[:,2])) % abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,12]) % abs(X[:,1])) % abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,15]) % abs(X[:,2])) % abs(X[:,1]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,15]) / abs(X[:,1])) % abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,18]) / abs(X[:,1])) % abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,19]) % abs(X[:,1])) % abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,21]) % abs(X[:,1])) % abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,20]) % abs(X[:,1])) / abs(X[:,2]), axis=1)
X = np.insert(X, X.shape[1], (abs(X[:,20]) % abs(X[:,1])) % abs(X[:,1]), axis=1)
##################
X-=(np.mean(X,axis=0))
X/=(np.std(X,axis=0))
X=np.insert(X,0,1,axis=1)
y=wine[:,13]
d=X.shape[1]
THETA=np.random.uniform(low=-np.e,high=np.e,size=(d,))
#%X[:,1:3]-=(np.mean(X[:,1:3],axis=0))
#X[:,1:3]/=(np.std(X[:,1:3],axis=0))
def fTheta(X,theta):
    return (X.dot(theta.T))
###
def computeCost(X,y,theta):
   m=X.shape[0]
   newy=fTheta(X,theta)
   erro=((newy-y)**2).sum()
   return (erro/(2*m))
###
def Gradiente(X, y, theta, alpha, niters):
    m=X.shape[0]
    cost=np.zeros(niters).astype(float)
    for k in range (0,niters):
      newy=fTheta(X,theta)
      erro=(newy-y).T.dot(X)
      theta-=(alpha*(erro/m))
      cost[k]=computeCost(X,y,theta)
    return (cost,theta)
