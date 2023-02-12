from M16 import *
from Gauss_Jordan import Gauss_Jordan

t = 8
Ma = np.zeros([256,2])
for a in range(256):
    Ma[a,0] = math.floor(a/np.power(2,t-1))
    Ma[a,1] = np.mod(a,np.power(2,t-1))

nbits = 2
m = 4
M1 = np.matrix([
        [60,    34,    35,    63],
    [30,     8,    59,     3],
    [53,    60,     9,    10],
    [51,    40,    10,    38]
])
M = FormM(M1, m, Ma, 1)
delta = np.zeros([m,m,2]).astype('int32')
delta[:,:,0] = np.random.randint(0,2,size=(m,m))
Y2 = perm_matrix(m,t)

Y2 = np.matrix([
        [76,   124,    81,   116],
    [42,   105,    96,   104],
     [2,   122,    60,    61],
    [61,    52,    82,   112]
])
X = np.random.randint(0,np.power(2,t-1),size=(m,m)).astype('int32')

C = M16_Enc(M[:,:,0,:],delta, X,Y2,t, nbits)
YY = Gauss_Jordan(Y2,np.power(2,(t-1)))

de = M16_Dec(C,delta, X,YY.astype('int32'),t, nbits)

