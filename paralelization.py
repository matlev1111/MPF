from joblib import Parallel, delayed
from joblib import parallel_backend
import numpy as np
import tracemalloc
import timeit
from ESakCipher import *
import matplotlib.pyplot as plt
from PIL import Image
from modes import E_sak_CBC as Forward, Form_pic_blocks, Form_text_blocks

def mod_exp(a,b,prime):
    v = a
    vv = a
    if(b == 0 ):
        return 1
    for i in range(b-1):
        v  = (v*vv)%prime
        
    return v


def mod_exp(a,b,prime):
    """
    Modular exponenet, computation of operation:
    a^b mod p
    Input: base a, power b, modulus prime;
    Output: returns the result of operation above.
    """
    v = a
    vv = a
    if(b == 0 ):
        return 1
    for i in range(b-1):
        v  = (v*vv)%prime
        
    return v

def f2(w,xi,yi,prime,q, Fm, NX, ii,jj, G):
    """
    Matrix power function for one element.
    Input: platform matrix w, vectors of left and right matrices xi, yi, and prime number prime;
    Output: Returns single element of result matrix.
    """
    rez = 1
    x = xi
    y = yi
    for i in range(len(x)):
        for j in range(len(x)):
            rez = (rez * mod_exp(w[i][j], x[i]*y[j]%(prime-1),prime)) % prime
    return (np.mod(imap(G,np.mod(rez*Fm[ii,jj],prime))+NX[ii,jj],q))

num_cores = 9
T = []
T2 = []
FT = []
FT2 = []
FMemB = []
FMemB2 = []
k=0
pp2 = 563
q = 281
m = 4
T = []
T2 = []
MemB = []
MemB2 = []
im = Image.open("images/coloredChips.png").convert('L')#Image.open("coloredChips.png")
X1 = np.asarray(im)

_, G, X, Y, Z = Gen_parameters(m,pp2)
#W = Gmap(G,np.mod(np.random.randint(0,q,size=(m,m), dtype='int32')+X,q))
Y = Y.astype('int32')
Fm = Gmap(G,X)
for m in range(4,8):
    s1, s2, M1 = Form_pic_blocks(m, X1,1)
    IV = np.zeros([m,m], dtype=int)
    Nblocks = M1.shape[2]
    print(Nblocks)
    CF = np.zeros([m,m,Nblocks+1], dtype='int32')
    CF[:,:,0] = IV
    for k in range(Nblocks):   
        print(f"m = {m}, ",round(k/Nblocks*100,2)) 
        W = Gmap(G,np.mod(CF[:,:,k]+M1[:,:,k],q))
        tracemalloc.start()
        start_time = timeit.default_timer()
        with parallel_backend('loky', n_jobs=num_cores):
            bufferT = Parallel()(delayed(f2)(W,p,pp,pp2,q, Fm, X, ind1, ind2, G) for ind1, p in enumerate(Y) for ind2, pp in enumerate(Y.transpose()))
        tracemalloc.stop()
        MemB.append(tracemalloc.get_tracemalloc_memory())
        T.append(timeit.default_timer()-start_time)
        tracemalloc.start()
        start_time = timeit.default_timer()
        N = [f2(W,p,pp,pp2,q, Fm, X, ind1, ind2, G) for ind1, p in enumerate(Y) for ind2, pp in enumerate(Y.transpose())]
        tracemalloc.stop()
        MemB2.append(tracemalloc.get_tracemalloc_memory())
        T2.append(timeit.default_timer()-start_time)
        CF[:,:,k+1] = np.array(bufferT).reshape([m,m])
    FT.append(np.sum(T))
    FT2.append(np.sum(T2))
    FMemB.append(np.mean(MemB))
    FMemB2.append(np.mean(MemB2))

print(FT)
print(FT2)

"""plt.plot(range(2,12),FT)
plt.plot(range(2,12),FT2)
plt.show()"""