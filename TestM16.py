from M16 import *
from Basic_functions.Gauss_Jordan import Gauss_Jordan
from timeit import default_timer as timer
from datetime import timedelta

t = 8
Ma = np.zeros([256,2])
for a in range(256):
    Ma[a,0] = math.floor(a/np.power(2,t-1))
    Ma[a,1] = np.mod(a,np.power(2,t-1))

nbits = 2
m = 4
M1 = np.random.randint(0,np.power(2,t-1),size=(m,m)).astype('int32')
M = FormM(M1, m, Ma, 1)

print("#",M.shape)
delta = np.zeros([m,m,2]).astype('int32')
delta[:,:,0] = np.random.randint(0,2,size=(m,m))
Y2 = perm_matrix(m,t)
YY = Gauss_Jordan(Y2,np.power(2,(t-1)))
X = np.random.randint(0,np.power(2,t-1),size=(m,m)).astype('int32')

start = timer()
C = M16_Enc(M[:,:,0,:],delta, X,Y2,t, nbits)
end = timer()


de = M16_Dec(C,delta, X,YY.astype('int32'),t, nbits)
print(M[:,:,0,0])
print(de[:,:,0])
print(M[:,:,0,1])
print(de[:,:,1])
M = M.reshape(m,m,2)
print((M == de).all())
print(timedelta(seconds=end-start))




