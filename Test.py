from main_func import *
from ESakCipher import *

W = np.matrix([[1,2],[3,4]])
p = np.int32(7)
X = np.matrix([[1,2],[3,4]])

print(matrix_exp_left(W,X,p))
print(matrix_exp_right(W,X,p))


p = 11
q = (p-1)//2
m = 4
ind, G = GenG(p)
X = np.random.randint(0,q,(m,m))
Y = np.random.randint(1,q,(m,m))
M = [[2,2,3,3],[2,2,4,4],[2,2,3,3],[2,2,4,4]]

ats = ESakCipher(G,X,M,Y,p)
print(ats)

#To do:
#Test mul_inv, hadamard_inv and ESakDecryption