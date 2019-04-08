import numpy as np
from sklearn import linear_model as lm

mpg = np.genfromtxt('auto-mpg.data',delimiter=';',dtype=str)
mpg = mpg[:,:-1]

## Remover ?
mpg = np.char.replace(mpg, '?', '-200')

## Separação de rótulos e dados
x = mpg[:,1:]
y = mpg[:,0]

## Transformar em float
x = x.astype(np.float)
y = y.astype(np.float)

## Altera -200 da coluna para a sua média
avg = np.mean(x[np.nonzero(x[:,2]!=-200),2])
x[np.nonzero(x[:,2] == -200),2] = avg

## Divide o conjunto de dado em 70% para treinamento e 30% para teste
tr=int(mpg.shape[0]*.70)
xtr=x[:tr,:]
ytr=y[:tr]
xte=x[tr:,:]
yte=y[tr:]


r = lm.LogisticRegression()
r.fit(xtr,ytr)
y_hat=r.predict(xte)
print('   Taxa de acerto (treino):', np.mean(y_hat==ytr))

