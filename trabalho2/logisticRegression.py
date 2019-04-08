import numpy as np
from sklearn import linear_model

# valor mais frequente no array
def mfv(ar):
	v, c = np.unique(ar, return_counts=True)
	return v[np.argmax(c)]

# leitura do arquivo e elimina espaços extras
data = np.char.strip(np.genfromtxt('adult.data', delimiter=',', dtype=str))

# separa atributos X dos rótulos y
X = data[:, :-1]
# rótulos classificados em 1 ou 2
y = np.unique(data[:, -1], return_inverse=True)[1] + 1

# subtituir valores faltantes em X pelo de maior ocorrência na coluna
for i in range(X.shape[1]):
	mf = mfv(X[:, i])
	X[np.nonzero(X[:, i] == '?'), i] = mf
	# coluna com valores classificados numericamente
	if not X[0, i].isdigit():
		X[:, i] = np.unique(X[:, i], return_inverse=True)[1] + 1

# todos os valores de X para int
X = X.astype(np.int)

# quantidade de registros para treinamento
training_qty = int(data.shape[0] * 0.7)

# grupos de treinamentos e testes
Xtr = X[:training_qty,:]
ytr = y[:training_qty]
Xte = X[training_qty:,:]
yte = y[training_qty:]

# regressão logistica
regr = linear_model.LogisticRegression(penalty='l1', random_state=108, solver='liblinear')
regr.fit(Xtr, ytr)

# resultados
y_pred = regr.predict(Xte)
print('Acurácia (teste):', np.mean(y_pred == yte))
