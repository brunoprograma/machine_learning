import numpy as np
from sklearn import linear_model


# carrega o arquivo
data = np.genfromtxt('iris.data', delimiter=',', dtype=str)

# sustitui os rótulos
data = np.char.replace(data, 'Iris-setosa', '0')
data = np.char.replace(data, 'Iris-virginica', '1')
data = np.char.replace(data, 'Iris-versicolor', '1')

# embaralha a matriz pois está organizada
np.random.seed(137)
np.random.shuffle(data)

# separa a matriz X (atributos) do vetor y (rótulos) e transforma em dados numéricos
X = data[:,:-1].astype(np.float)
y = data[:,-1].astype(np.int)

# quantidade de registros para treinamento
training_qty = int(data.shape[0] * 0.75)

# grupos de treinamentos e testes
Xtr = X[:training_qty,:]
ytr = y[:training_qty]
Xte = X[training_qty:,:]
yte = y[training_qty:]

# regressão logistica
regr = linear_model.LogisticRegression()
regr.fit(Xtr, ytr)

# resultados
y_pred = regr.predict(Xtr)
print('Taxa de acerto (treino):', np.mean(y_pred == ytr))
y_pred = regr.predict(Xte)
print('Taxa de acerto (teste):', np.mean(y_pred == yte))

