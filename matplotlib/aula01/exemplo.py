import matplotlib.pyplot as plt
import numpy as np

x = np.array([2,4,6,8,10,12])
plt.plot(x, x, 'rx',x,x**2,'bx')
l=np.array(['linear','quadratic'])
plt.legend(l)
plt.show()
####
t = np.arange(0, 5,0.1)
plt.plot(t, np.sqrt(t), 'r--',t,t**2,'bs',t,t**3,'g^')
plt.legend(['Square root','quadratic','cubic'])
plt.show()
##
points = np.arange(-5, 5, 0.01)
# Make a meshgrid
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs ** 2 + ys ** 2)
# Display the image on the axes
plt.imshow(z, cmap=plt.cm.gray)
# Draw a color bar
plt.colorbar()
# Show the plot
plt.show()
plt.imshow(z, cmap=plt.cm.Greens)
####
from sklearn import datasets as ds
dg=ds.load_digits()
img=dg.data[0,:]
img=np.reshape(img,(8,8))
plt.imshow(img, cmap=plt.cm.gray)
plt.colorbar()
plt.show()
####
img=dg.data[0:10,:]
img=np.reshape(img,(80,8))
plt.imshow(img, cmap=plt.cm.gray)
plt.colorbar()
plt.show()
####
grades = np.array([89, 90, 70, 89, 100, 80, 90, 100, 80, 34,30, 29, 49, 48, 100, 48, 38, 45, 20, 30, 83,67,78])
gender_grades = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0])
plt.scatter(np.linspace(0,100,grades[gender_grades==0].shape[0]),grades[gender_grades==0], color='r',label='girls')
plt.scatter(np.linspace(0,100,grades[gender_grades==1].shape[0]),grades[gender_grades==1], color='g',label='boys')
plt.xlabel('Grades Range')
plt.ylabel('Grades Scored')
plt.legend()
plt.show()
####

