import numpy as np
import math


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
                tmp = np.mod(np.power(W[k,j],X[i,k]),p)
                M[i,j] = M[i,j]*tmp % p
    return M

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
                tmp = np.mod(np.power(W[i,k],X[k,j]),p)
                M[i,j] = M[i,j]*tmp % p
    return M

def hadamard_prod(A,B,p):
    return np.mod(np.multiply(A,B),p)

def mulinv(number, modulo):
    if(math.gcd(number, modulo) != 1):
        #"Inverse element does not exist"
        return -1
    else:
        m, n = np.int64(number),np.int64(modulo)
        s,t,u,v = np.int64(1), np.int64(0), np.int64(0), np.int64(1)
        while(n > 0) :
            q=np.int64(math.floor((np.double(m))/np.double(n)))
            r=np.int64(m-q*n)
            m, n =n, r
            u, s =s-q*u, u
            v, t =t-q*v, v
            out = s
            if(out < 0):
                out = modulo+out
    return out

def hadamard_inv(M,p):
    H = None
    fm = np.vectorize(lambda x: mulinv(x,p))
    H=fm(M)
    return H


