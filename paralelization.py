from joblib import Parallel, delayed
from joblib import parallel_backend
import numpy as np
import tracemalloc
import timeit
from ESakCipher import *
import matplotlib.pyplot as plt
import pickle
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

def matrix_exp_right(W,X,p):
    """
    Function allows to raise matrix W by matrix power X from the right side modulo p
    """
    _, n2 = W.shape
    m1, _ = X.shape
    M = np.zeros([n2,m1])
    for i in range(n2):
        for j in range(m1):
            M[i,j] = 1
            for k in range(n2):
                tmp = mod_exp(W[i,k],X[k,j],p)
                M[i,j] = M[i,j]*tmp % p
    return M.astype('int32')

def f_s(W,X,p):
    m2 = len(X)
    M = 1
    for k in range(m2):
        tmp = mod_exp(W[k],X[k],p)
        M = M*tmp % p
    return M



num_cores = 2
T = []
T2 = []
FT = []
FT2 = []
FT3 = []
FMemB = []
FMemB2 = []
k=0
pp2 = 563
q = 281
m = 16
T = []
T2 = []
T3 = []
MemB = []
MemB2 = []
im = Image.open("images/coloredChips.png").convert('L')#Image.open("coloredChips.png")
X1 = np.asarray(im)


B = []
_, G, X, Y, Z = Gen_parameters(m,pp2)
#W = Gmap(G,np.mod(np.random.randint(0,q,size=(m,m), dtype='int32')+X,q))
Y = Y.astype('int32')
Fm = Gmap(G,X)
s1, s2, M1 = Form_pic_blocks(m, X1,1)
IV = np.zeros([m,m], dtype=int)
Nblocks = 1#M1.shape[2]
CF = np.zeros([m,m,Nblocks+1], dtype='int32')
CF[:,:,0] = IV
B.append(Nblocks)
for k in range(Nblocks):   
    print(f"m = {m}, ",round(k/Nblocks*100,2)) 
    W = Gmap(G,np.mod(CF[:,:,k]+M1[:,:,k],q))
    start_time = timeit.default_timer()
    with parallel_backend('threading', n_jobs=num_cores):
        bufferT = Parallel(verbose = 10)(delayed(f_s)(W[:,w],Y[x,:],pp2) for x in range(m) for w in range(m))
    W2= np.array(bufferT).reshape([m,m])
    with parallel_backend('threading', n_jobs=num_cores):
        bufferT = Parallel()(delayed(f_s)(W2[w,:],Y[:,x],pp2) for x in range(m) for w in range(m))
    W2 = np.array(bufferT).reshape([m,m])
    C2 = hadamard_prod(Fm, W2,pp2)
    T.append(timeit.default_timer()-start_time)
    CF[:,:,k+1] = np.mod(imap(G, C2)+X,q).astype('int32')
    start_time = timeit.default_timer()
    N = Parallel(n_jobs=1)(delayed(f_s)(W[w,:],Y[:,x],pp2) for x in range(m) for w in range(m))
    W = np.array(bufferT).reshape([m,m])
    N2 = Parallel(n_jobs=1)(delayed(f_s)(W[w,:],Y[:,x],pp2) for x in range(m) for w in range(m))
    W = np.array(bufferT).reshape([m,m])
    C2 = hadamard_prod(Fm, W,pp2)
    T2.append(timeit.default_timer()-start_time)
    start_time = timeit.default_timer()
    N = [f_s(W[w,:],Y[:,x],pp2) for x in range(m) for w in range(m)]
    W = np.array(bufferT).reshape([m,m])
    N2 = [f_s(W[w,:],Y[:,x],pp2) for x in range(m) for w in range(m)]
    W = np.array(bufferT).reshape([m,m])
    C2 = hadamard_prod(Fm, W,pp2)
    T3.append(timeit.default_timer()-start_time)
FT.append(np.sum(T))
FT2.append(np.sum(T2))
FT3.append(np.sum(T3))

print(B)
print(FT)
print(FT2)
print(FT3)
"""for m in range(4,5):
    _, G, X, Y, Z = Gen_parameters(m,pp2)
    #W = Gmap(G,np.mod(np.random.randint(0,q,size=(m,m), dtype='int32')+X,q))
    Y = Y.astype('int32')
    Fm = Gmap(G,X)
    s1, s2, M1 = Form_pic_blocks(m, X1,1)
    IV = np.zeros([m,m], dtype=int)
    Nblocks = M1.shape[2]
    CF = np.zeros([m,m,Nblocks+1], dtype='int32')
    CF[:,:,0] = IV
    B.append(Nblocks)
    for k in range(Nblocks):   
        print(f"m = {m}, ",round(k/Nblocks*100,2)) 
        W = Gmap(G,np.mod(CF[:,:,k]+M1[:,:,k],q))
        start_time = timeit.default_timer()
        with parallel_backend('loky', n_jobs=num_cores):
            #bufferT = Parallel()(delayed(f2)(W,p,pp,pp2,q, Fm, X, ind1, ind2, G) for ind1, p in enumerate(Y) for ind2, pp in enumerate(Y.transpose()))
            bufferT = Parallel(delayed(f_left(W,X,pp2)))
        T.append(timeit.default_timer()-start_time)
        start_time = timeit.default_timer()
        N = [f2(W,p,pp,pp2,q, Fm, X, ind1, ind2, G) for ind1, p in enumerate(Y) for ind2, pp in enumerate(Y.transpose())]
        T2.append(timeit.default_timer()-start_time)
        CF[:,:,k+1] = np.array(bufferT).reshape([m,m])
    FT.append(np.sum(T))
    FT2.append(np.sum(T2))

print(B)
print(FT)
print(FT2)

file = open(f'res/CBC_ptimes_12c', 'wb')
pickle.dump([B, FT, FT2], file)
file.close()
"""
"""plt.plot(range(2,12),FT)
plt.plot(range(2,12),FT2)
plt.show()"""

"""
4x4: paralel - 0.063332299843 s/block (806.8535001650453 total)
4x4: normal - 0.069820216491365 s/block (889.5095581016503 total)

"""

""""
_, G, X, Y, Z = Gen_parameters(m,pp2)
    #W = Gmap(G,np.mod(np.random.randint(0,q,size=(m,m), dtype='int32')+X,q))
    Y = Y.astype('int32')
    Fm = Gmap(G,X)
    s1, s2, M1 = Form_pic_blocks(m, X1,1)
    IV = np.zeros([m,m], dtype=int)
    Nblocks = M1.shape[2]
    CF = np.zeros([m,m,Nblocks+1], dtype='int32')
    CF[:,:,0] = IV
    B.append(Nblocks)
    for k in range(Nblocks):   
        print(f"m = {m}, ",round(k/Nblocks*100,2)) 
        W = Gmap(G,np.mod(CF[:,:,k]+M1[:,:,k],q))
        start_time = timeit.default_timer()
        with parallel_backend('loky', n_jobs=num_cores):
            bufferT = Parallel()(delayed(f2)(W,p,pp,pp2,q, Fm, X, ind1, ind2, G) for ind1, p in enumerate(Y) for ind2, pp in enumerate(Y.transpose()))
        T.append(timeit.default_timer()-start_time)
        start_time = timeit.default_timer()
        N = [f2(W,p,pp,pp2,q, Fm, X, ind1, ind2, G) for ind1, p in enumerate(Y) for ind2, pp in enumerate(Y.transpose())]
        T2.append(timeit.default_timer()-start_time)
        CF[:,:,k+1] = np.array(bufferT).reshape([m,m])
    FT.append(np.sum(T))
    FT2.append(np.sum(T2))
"""