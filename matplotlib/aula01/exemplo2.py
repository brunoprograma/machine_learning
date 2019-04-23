import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets as ds

grades = np.array([89, 90, 70, 89, 100, 80, 90, 100, 80, 34,30, 29, 49, 48, 100, 48, 38, 45, 20, 30, 83,67,78])
gender_grades = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0])
plt.scatter(np.linspace(0,100,grades[gender_grades==0].shape[0]),grades[gender_grades==0], color='r',label='girls')
plt.scatter(np.linspace(0,100,grades[gender_grades==1].shape[0]),grades[gender_grades==1], color='g',label='boys')
plt.xlabel('Grades Range')
plt.ylabel('Grades Scored')
plt.legend()
plt.show()
