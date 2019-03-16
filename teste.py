import numpy as np
import pandas as pd

aq = np.genfromtxt('AirQualityUCI.csv', delimiter=';', dtype=str, skip_header=1)[:,2:-2]  # matriz com todos o conjunto de dados tirando as duas colunas iniciais e duas finais
aq = aq[:np.nonzero(aq=='')[0][0], :]  # apagamos as linhas que não possuem dados no final do arquivo
aq = np.char.replace(aq, ',', '.').astype(np.float)  # substituimos as virgulas por pontos depois transformamos os valores em float

y = aq[:, 3]  # guardamos a quarta coluna num novo array
aq = np.delete(aq, 3, axis=1)  # apagamos a quarta coluna da matriz axis: 0 = linha 1 = coluna 

# tratamento de valores inválidos (missing values) no caso do nosso conjunto de dados é -200
# substitui os valores inválidos por uma interpolação entre os valores válidos do conjunto
#xpd = pd.DataFrame(aq)
#xpd = xpd.interpolate(aq)
#xpd = nparray(xpd)

d = X.shape[1]
for i in range(d):
	avg = np.mean(X[np.nonzero(X[:,i] != -200), i])
	X[np.nonzero(X[:,i] == -200)] = avg
