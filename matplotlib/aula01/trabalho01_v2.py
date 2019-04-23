import matplotlib.pyplot as plt
from sklearn import datasets as ds

iris = ds.load_iris()

for i in range(len(iris.feature_names)):
	for j in range(i+1, len(iris.feature_names)):
		if i == j:
			continue

		plt.scatter(iris.data[iris.target==0, i], iris.data[iris.target==0, j], color='r', label=iris.target_names[0])
		plt.scatter(iris.data[iris.target==1, i], iris.data[iris.target==1, j], color='y', label=iris.target_names[1])
		plt.scatter(iris.data[iris.target==2, i], iris.data[iris.target==2, j], color='b', label=iris.target_names[2])
		plt.xlabel(iris.feature_names[i])
		plt.ylabel(iris.feature_names[j])
		plt.legend()
		plt.show()

