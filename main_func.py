import pandas as pd
import numpy as np


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
