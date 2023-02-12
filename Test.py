from main_func import *
from ESakCipher import *
from Gauss_Jordan import Gauss_Jordan


W = np.matrix([[1,2],[3,4]])
p = np.int32(7)
X = np.matrix([[1,2],[3,4]])

#print(matrix_exp_left(W,X,p))
#print(matrix_exp_right(W,X,p))


p = 563
q = (p-1)//2
m = 4
ind, G = GenG(p)
X = np.random.randint(0,q,(m,m))
Y = np.random.randint(1,q,(m,m))
while(np.mod(np.linalg.det(Y),q) == 0):
    Y = np.random.randint(1,q,(m,m))

M = [[2,2,3,3],[2,2,4,4],[2,2,3,3],[2,2,4,4]]
X = np.matrix([
       [235,   107,   157,    55],
    [96,   275,   134,    59],
   [168,   255,   214,    28],
   [123,   102,   208,   222]])


Y = np.matrix([[215,     56,   121,   250],
   [4,   230,   194,    49],
   [149,    229,   195,    178],
   [108,   9,   8,    181]])
ats = ESakCipher(G,X,M,Y,p)
print(ats)

YY = Gauss_Jordan(Y,q)

print(ESakDecryption(G,ind, YY, X, ats,p))

#Test M16