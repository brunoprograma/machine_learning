import numpy as np
# import pandas as pd

aq = np.genfromtxt('AirQualityUCI.csv', delimiter=';', dtype=str, skip_header=1)[:,2:-2]  # matriz com todos o conjunto de dados tirando as duas colunas iniciais e duas finais
aq = aq[:np.nonzero(aq=='')[0][0], :]  # apagamos as linhas que não possuem dados no final do arquivo
aq = np.char.replace(aq, ',', '.').astype(np.float)  # substituimos as virgulas por pontos depois transformamos os valores em float

y = aq[:, 3]  # guardamos a quarta coluna num novo array
aq = np.delete(aq, 3, axis=1)  # apagamos a quarta coluna da matriz axis: 0 = linha 1 = coluna 

# eliminar colunas com porcentagem > X de valores inválidos
m = aq.shape[0]  # numero de linhas
d = aq.shape[1]  # numero de colunas
ths = 0.1  # 10%
cut = int(m * ths)  # número máximo de colunas com valor inválido

for i in range(d-1, -1, -1):
	if np.count_nonzero(aq[:, i] == -200) > cut:
		aq = np.delete(aq, i, axis=1)  # exclui a coluna i

# tratamento de valores inválidos (missing values) no caso do nosso conjunto de dados é -200
# substitui os valores inválidos por uma interpolação entre os valores válidos do conjunto
# xpd = pd.DataFrame(aq)
# xpd = xpd.interpolate()
# aq = np.array(xpd)

d = aq.shape[1]  # numero de colunas

# substitui missing pela média dos valores da coluna
for i in range(d):
	avg = np.mean(aq[np.nonzero(aq[:,i] != -200), i])
	aq[np.nonzero(aq[:,i] == -200)] = avg
	
finaldata = np.insert(aq, d, y, axis=1)  # junta aq com y (na ultima coluna)
np.savetxt('AirQFinal.csv', finaldata, delimiter=',', fmt='%f')  # salva a nova matriz num csv	
