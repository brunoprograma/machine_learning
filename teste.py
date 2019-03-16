import numpy as np

aq = np.genfromtxt('AirQualityUCI.csv', delimiter=';', dtype=str, skip_header=1)[:,2:-2]
aq = aq[:np.nonzero(aq=='')[0][0], :]
