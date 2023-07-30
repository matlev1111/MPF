import numpy as np
from joblib import Parallel, delayed
from joblib import parallel_backend
import timeit
from numba import jit

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
    for _ in range(b-1):
        v  = (v*vv)%prime
        
    return v

def matrix_exp_left(W,X,p):
    """
    Function allows to raise matrix W by matrix power X from the left side modulo p
    """
    n1, _ = W.shape
    _, m2 = X.shape
    M = np.zeros([n1,m2])
    for i in range(m2):
        for j in range(n1):
            M[i,j] = 1
            for k in range(m2):
                tmp = mod_exp(W[k,j],X[i,k],p)
                M[i,j] = M[i,j]*tmp % p
    return M.astype('int32')

@jit
def f_s(W,X,p,m2):
    M = 16#1
    """for k in range(m2):
        tmp = mod_exp(W[k],X[k],p)
        M = np.mod(M*tmp,p)"""
    return M**3

m = 2
W = np.array([[1, 2],[3,4]])
X = np.array([[1,2],[3,4]])

p = 563
bufferT = []
for i in range(2):
    for j in range(2):
        bufferT.append(f_s(W[:,j],X[i,:],p,m))

print(matrix_exp_left(W,X,p))
print(np.array(bufferT).reshape([2,2]))

m = 4
W = np.random.randint(0,p,size=(m,m))
X = np.random.randint(0,p,size=(m,m))


start_time = timeit.default_timer()
bufferT = Parallel(n_jobs = 8, backend='threading', verbose=0)(delayed(f_s)(W[:,w],X[x,:],p,m) for x in range(m) for w in range(m))
print(timeit.default_timer()-start_time)
print(bufferT)
start_time = timeit.default_timer()
N = Parallel(n_jobs = 1)(delayed(f_s)(W[:,w],X[x,:],p,m) for x in range(m) for w in range(m))
print(timeit.default_timer()-start_time)
print(N)



def cube(x):
    return x**3
 
"""start_time = timeit.default_timer()
result = Parallel(n_jobs=2, prefer="threads", verbose=2)(delayed(cube)(i) for i in range(1,1000))
finish_time = timeit.default_timer()
print(finish_time-start_time)
start_time = timeit.default_timer()
result = Parallel(n_jobs=1, prefer="threads", verbose=2)(delayed(cube)(i) for i in range(1,1000))
finish_time = timeit.default_timer()
print(finish_time-start_time)"""
"""from time import sleep
print(Parallel(n_jobs=2, verbose=10)(delayed(sleep)(.2) for _ in range(10)))
print(Parallel(n_jobs=1, verbose=10)(delayed(sleep)(.2) for _ in range(10)))"""


import time
import concurrent.futures
import time


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(3) as executor:
        start_time = time.perf_counter()
        result = list(executor.map(cube, range(1,1000)))
        finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    with concurrent.futures.ProcessPoolExecutor(1) as executor:
        start_time = time.perf_counter()
        result = list(executor.map(cube, range(1,1000)))
        finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    #print(result)