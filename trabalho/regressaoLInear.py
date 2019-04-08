import numpy as np
from sklearn import linear_model

# carrega o arquivo
data = np.genfromtxt('auto-mpg.data', delimiter=';', dtype=str)

# limpa os missing values '?'
data = np.char.replace(data, '?', '-99999.')

# elimina coluna com os nomes dos carros
data = data[:, :-1]

# separa a matriz X (atributos) do vetor y (rótulos) e transforma em dados numéricos
X = data[:, 1:].astype(np.float)
y = data[:, 0].astype(np.float)

## replace missing values (-99999.) by the average of the feature 2
avg = np.mean(X[np.nonzero(X[:, 2] != -99999), 2])
X[np.nonzero(X[:, 2] == -99999), 2] = avg

# quantidade de registros para treinamento
training_qty = int(data.shape[0] * 0.7)

# grupos de treinamentos e testes
Xtr = X[:training_qty,:]
ytr = y[:training_qty]
Xte = X[training_qty:,:]
yte = y[training_qty:]

# regressão logistica
regr = linear_model.LinearRegression(normalize=True, fit_intercept=False)
regr.fit(Xtr, ytr)

# erro do conjunto de teste
y_pred = regr.predict(Xte)
print('Erro residual (teste):', abs(y_pred - yte).sum() / len(yte))

