import numpy as np

fname='AirQualityUCI.csv'
#############try:
##################fcsv=open(fname)
############except:
##################print('Error opening:',fname)
#
# Loading the dataset into memory
#data=np.array([row.split(';') for row in fcsv.readlines()])
try:
	data=np.genfromtxt(fname,delimiter=';',skip_header=1,dtype=str)
except:
	print('Error opening:',fname)
	exit(0)
#### AirQualityUCI.csv description:
## There is header
## number of examples (lines):9471
## number of attributes (+label): 17
## Some attributes are empty
## Some attributes store -200 (missing value)
###########################
## delete the two last columns (empty and new line)
data=data[:,:-2]  ### all rows and columns from 0 to the end - 2
## date and time may be useless for our model
data=data[:,2:]  ## all rows and all columns but the first and second
### the end of file is full of empty lines, i.e., ;;;;;
### look for empty lines
## data=='' returns the same matrix but the cells contain True or False
## return the position where the condition is True
## we take the first one, i.e., [0][0]
lastpos=np.nonzero(data=='')[0][0]
data=data[:lastpos,:] ## all rows but the lastpos ones
## copy the 3rd attribute to y
## But before commas must be replaced by points 
## in the csv, the float attribute values are separated by commas
y=[float(label.replace(',','.')) for label in data[:,3]]
## Start creating X
X=np.delete(data,3,axis=1) ## all data matrix but the 3rd column (it is the label)
m=X.shape[0] ## number of lines
d=X.shape[1] ## number of attributes
## convert all string values into float and integer
## first replace , by .
for i in range(d):
  if not X[0,i].isdigit(): ## there is a comma in the i-th column?
    X[:,i]=[v.replace(',','.') for v in X[:,i]] ## all values must be fixed
### convert X to float
X=X.astype(float)
### Let's get rid off some missing values (-200)
ths=0.1 ## corresponds to 10% (you can increase or decrease this value)
cut=int(m*ths) ## correspond to the maximum number of -200 values in a column
## all feature with more than "ths" of missing value will be deleted
for i in range(d-1,-1,-1): ## reverse loop
  if np.count_nonzero(X[:,i]==-200) > cut:
    X=np.delete(X,i,axis=1) ## delete i-th column

## we had 12 attributes, now we have 9
## replace missing values (-200) by the average of the feature
d=X.shape[1] ## d must be updated (some features were deleted)
for i in range(d):
  avg=np.mean(X[np.nonzero(X[:,i]!=-200),i])
  X[np.nonzero(X[:,i]==-200),i]=avg

finaldata=np.insert(X,d,y,axis=1)
np.savetxt('AirQFinal.csv',finaldata,delimiter=',')
