from Basic_functions.main_func import *
from ESakCipher import *
from Basic_functions.Gauss_Jordan import Gauss_Jordan
from timeit import default_timer as timer
from datetime import timedelta
import gc

W = np.matrix([[1,2],[3,4]])
p = np.int32(7)
X = np.matrix([[1,2],[3,4]])

#print(matrix_exp_left(W,X,p))
#print(matrix_exp_right(W,X,p))


p = 563
q = (p-1)//2
m = 4
G = GenG(p)
X = np.random.randint(0,q,(m,m))
Y = np.random.randint(1,q,(m,m))
while(np.mod(np.linalg.det(Y),q) == 0):
    Y = np.random.randint(1,q,(m,m))

M = [[2,2,3,3],[2,2,4,4],[2,2,3,3],[2,2,4,4]]
"""X = np.matrix([
       [235,   107,   157,    55],
    [96,   275,   134,    59],
   [168,   255,   214,    28],
   [123,   102,   208,   222]])


Y = np.matrix([[215,     56,   121,   250],
   [4,   230,   194,    49],
   [149,    229,   195,    178],
   [108,   9,   8,    181]])"""

"""
Fmap = Gmap(G,X)
print(Fmap)
print(G)
Rmap = imap(G,Fmap)
print(Rmap)
"""
def FMPF(X,W,Y,i,j,p):
   el = 1
   for k in range(m):
      for l in range(m):
         pr = np.mod(X[i,k]*Y[l,j],p-1)
         el = np.mod(el*mod_exp(W[k,l],pr,p),p)
   return el



p = 563
m = 4
X = np.random.randint(0,p,(m,m),dtype='int32')
W = np.random.randint(1,p,(m,m),dtype='int32')
start = timer()
C1 = matrix_exp_left(W,X,p)
C2 = matrix_exp_right(C1,X,p)
#print(C2)
end = timer()
print(timedelta(seconds=end-start))
gc.collect()

D = np.zeros([m,m],dtype='int32')
start = timer()
for i in range(m):
   for j in range(m):
      D[i,j] = FMPF(X,W,X,i,j,p)
end = timer()
print(timedelta(seconds=end-start))
#print(D)

#ats = ESakCipher(G,X,M,Y,p, q, Fmap)
#print(ats)

#YY = Gauss_Jordan(Y,q)

#print(ESakDecryption(G, YY, X, ats,p))

#Test M16