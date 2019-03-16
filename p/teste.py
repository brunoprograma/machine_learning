import numpy as np
###################
###################
######## M A T R I Z E S #############
A=np.array([[10,11],[20,22],[30,33]])
A.shape
## (3,2)
A.shape[0]
# 3
A.shape[1]
# 2
B=np.array([[1,1],[2,2],[3,3]])
## apenas a primeira coluna de todas a linhas
A[:,0]
A[0,:] ## todos os valores da primeira linha (0)
#
z=np.zeros((5,3)) # matriz 5 x 3 preenchida com 0's
o=np.ones((5,3)) # matriz 5 x 3 preenchida com 1's
## A3x2 B3x2
A.dot(B)  ## .dot -> multiplicação de matrizes (A e B são incompatíveis)
B.do(A)   ## não
A*B ## OK multiplicação das células (matrizes de mesma dimensão)
A.T  ## transposta
B.T  ## transposta
A.dot(B.T)
B.dot(A.T)
A.T.dot(B)
B.T.dot(A)
#insert values into an array
#np.insert(array,index,values,axis) # is axis is ommited, the array is flattened first
a=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
np.insert(a,0,10) # insert 10 into the first position and flatten the array, so 4x14 become 16 + the first new position
##  ---> 10,1,2,3,4,1,2,3,4...
np.insert(a,0,10,axis=1) # insert a new column of 10 at position 0
## [[10, 1, 2 ,3 4],...,[10, 1, 2 ,3 4]]
np.insert(a,0,10,axis=0) # insert a new row of 10 at the first position
##[[10,10,10,10],[1,2,3,4],...,[1,2,3,4]]
###
### listas. Parecidas com matrizes mas não possuem as operações das matrizes
l=[] 
l.append([1,2,3,4])
l[0] ## --> [1,2,3,4]
l[0][0] ## --> 1
for v in l[0]:
	print(v)

l.append([5,6,7,8])
l[0] ## --> [5,6,7,8]
l[0][1] ## --> 6
l.append(100)
l.append(200)
for v in l:
	print(v)
## resultado [[1, 2, 3, 4], [5, 6, 7, 8], 100, 200]
# listas aceitam valores de vários tipos
lm=[]
lm.append(100) # inteiro
lm.append('Cem') # alfanumérico
lm.append(4.5) # float
lm.append([1,2,3]) # lista
####################################
#####################################
## importar dados de arquivos textos (sem formatação)
## loadtxt carrega arquivo para memória em uma matriz mas sem muitas opções de carga
## genfromtxt (equivalente ao loadtxt) porém é possível indicar o que é um valor faltante e como substituí-lo
## comando mais básico
np.loadtxt('iris.csv',delimiter=',',dtype='float')
np.loadtxt('iris.csv',delimiter=',',dtype='float',usecols=(0,2))
np.loadtxt('iris.csv',delimiter=',',dtype='float',skiprows=130)
## comando mais completo
np.genfromtxt('iris.csv',delimiter=',',dtype='float')
np.genfromtxt('iris.csv',delimiter=',',dtype='float',usecols=(0,2))
np.genfromtxt('iris.csv',delimiter=',',dtype='float',skip_header=130) ## skiprows
np.genfromtxt('iris.csv',delimiter=',',dtype='float',skip_footer=30)
## os arquivo irismissing.csv possui valores faltantes ou seja tuplas 1.2,0.5,,4.5,0
## se executar o loadtxt, o programa vai retornar um erro
## neste caso, genfromtxt é obrigatório e vai substituir o valor faltante por NaN (not a number)
iris=np.genfromtxt('irismissing.csv',delimiter=',',dtype='float')
## os valores faltantes podem ser subsituídos por um novo valor
iris=np.genfromtxt('irismissing.csv',delimiter=',',dtype='float',filling_values='-1') ## ou filling_values=-1
## retorna um vetor de True ou False que corresponde às celulas que são iguais a -1
nZero=iris==-1
## conta quantos valores não são 0 ou seja, os verdadeiros
print(np.count_nonzero(nZero))
# ou
print(np.count_nonzero(iris==-1))
# pode-se fazer o mesmo com os valores nan
iris=np.genfromtxt('irismissing.csv',delimiter=',',dtype='float')
print(np.count_nonzero(np.isnan(iris)))
##
# carregar a matriz com os nomes dos campos
##
iris=np.loadtxt('iris.csv',delimiter=',',dtype='float')
X=iris[:,0:4]
y=iris[:,4].astype(int)
X.shape
y.shape
## *, /, +, - , ** são elementwise
A=np.array([[1,1],[2,2],[3,3]])
B=np.array([[1,1],[2,2],[3,3]])
## A3x2 B3x2
A.dot(B)  ## não
B.do(A)   ## não
A*B ##OK
A.T  ## transposta
B.T  ## transposta
A.dot(B.T)
B.dot(A.T)
A.T.dot(B)
B.T.dot(A)
#insert values into an array
#np.insert(array,index,values,axis) # is axis is ommited, the array is flattened first
a=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
np.insert(a,0,10) # insert 10 into the first position and flatten the array, so 4x14 become 16 + the first new position
10,1,2,3,4,1,2,3,4...
np.insert(a,0,10,axis=1) # insert a new column of 10 at position 0
[[10, 1, 2 ,3 4],...,[10, 1, 2 ,3 4]]
np.insert(a,0,10,axis=0) # insert a new row of 10 at the first position
[[10,10,10,10],[1,2,3,4],...,[1,2,3,4]]
###
np.append([1,2,3,4],5)
[1,2,3,4,5]
np.append([1,2,3,4],[5,6,7])
[1,2,3,4,5,6,7]
np.append([[1,2],[3,4]],[5,6,7])
[1,2,3,4,5,6,7]
## must inform the axis
np.append([[1,2],[3,4]],[5,6,7],axis=0)
### erro
np.append([[1,2],[3,4]],[[5,6]],axis=0)
## erro
np.append([[1,2],[3,4]],[[5,6],[7,8]],axis=0)
[[1,2],..[5,6],[7,8]]
np.append([[1,2],[3,4]],[[5,6],[7,8]],axis=1)
[[1, 2, 5, 6],[3, 4, 7, 8]])
####
A=np.random.rand(3,3) ## random numbers from uniform distribuiton [0,1] with dimension n,m,k...
A=np.random.uniform(10)
A=np.random.uniform(low=1,high=100,size=(4,4))
A=np.random.uniform(low=1,high=100,size=4)
np.random.seed(seed) ## initialize random seed (track repetition)
A[:,0] ## first column, all rows
A[0,:] ## first row, all columns
A[0,0:2] ## first row, columns 0 and 1
## pay attention the upper limit is not taking into account
M=np.ndarray(shape=(3,),dtype=object) ## empty matrix with 3 rows for any kind of object
M[0]=np.array([4,5,6])
## construção de uma classe
class nome (object):
	def __init__(  ## construtor
	    self,
	    arg1,
	    arg2=None,
	    ...,
	    argN
	
	
	):
		código
		if arg2 is None:
			blá

    def funcao (self,arg1, ..., argN):
		corpo
		return valor;
		
import sys ## para receber parâmetros de chamada
class draw (object):
	 def __init__(x1=0,y1=0, x2=10, y2=10,cor=0):
		 self.X1=X1;
		 self.Y1=Y1;
		 self.X2=X2;
		 self.Y2=Y2;
		 self.cor=cor;

	def setColor(self,cor):
		self.cor=cor;
#		
	def getColor(self):
		return self.cor;
#
#	:
#	:
# para fazer a classe executável
if __name__ == "__main__":
	# sys.argv tem a relação dos parâmetros passados
	# sendo que o 0 é o nome da função
	# por exemplo, se são necessários 5 parâmetros
	if len(sys.argv<5):
		print(' Usage: python %s <par1> <par2> <par3> <par4> <par5>' % sys.argv[0])
		exit(0)
	p1=sys.argv[1]
	:
	p4=sys.argv[4]
	d=draw(p1,p2,p3,p4)
	d.setColor(5)
#
#		
for var in range (inicio, fim): ## vai até fim-1

for var in range (fim, inicio, passo): ##vai até inicio+1
