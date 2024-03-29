import numpy as np
import math

def mod_exp(g,n,p):
    """
    Modular exponenet function
    Input:
        g - generator
        n - power
        p - group order
    Output:
        r = g^n mod p
    """
    r = 1
    while(n > 0):
        if(np.mod(n,2) != 0):
            r = np.mod(np.multiply(r,g),p)
        n = math.floor(n/2)
        g = np.mod(np.multiply(g,g),p)
    return r

def matrix_exp_left(W,X,p):
    """
    Function allows to raise matrix W by matrix power X from the left side modulo p
    Input:
        W - base matrix
        X - matrix power
        p - group order
    Output:
        M =  ^X W mod p
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
    return M.astype('int64')

def matrix_exp_right(W,X,p):
    """
    Function allows to raise matrix W by matrix power X from the right side modulo p
    Input:
        W - base matrix
        X - matrix power
        p - group order
    Output:
        M =  W ^X mod p
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
    return M.astype('int64')

def hadamard_prod(A,B,p):
    """
    Hadamar product
    Input:
        A - first term
        B - second term
        p - group order
    Output:
        Hadamard multiplication result
    """
    return np.mod(np.multiply(A,B),p)

def mulinv(number, modulo):
    """
    Multiplicatively inverse element
    Input:
        number - element which inverse we want to find
        modulo - group order
    Output:
        out - Inverse element
    """
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
    """
    Inverse with respect of Hadamar operator
    Input:
        M - Initial matrix
        p - group order
    Output:
        H - inverse with respect to hadamard product operator
    """
    H = None
    fm = np.vectorize(lambda x: mulinv(x,p))
    H=fm(M)
    return H

def Shifting_bits(row,k):
    """
    Bit shifting by k positions
    Input:
        row - bits row
        k - how many bits
    Output:
        Shifted bits row
    """
    part = row[2:k+2]
    return row[:2]+row[k+2:] + part
