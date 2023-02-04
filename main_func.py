import pandas as pd
import numpy as np

"""
function M = matrix_exp_right(A, B, p)
    % Kelia matrica B matriciniu laipsniu A is desines
    [n1 n2] = size(A);
    [m1 m2] = size(B);
    for i = 1:n1
        for j = 1:m2
            M(i, j) = 1;
            for k = 1:n2
                tmp = powermod(B(i, k), A(k, j), p);
                M(i, j) = mod(M(i, j) * tmp, p);
            end
        end
    end
end
"""


def matrix_exp_left(W,X,p):
    """
    Function allows to raise matrix W by matrix power X from the left side modulo p
    """
    M = np.matrix()
    n1, _ = W.shape
    _, m2 = X.shape
    for i in range(m2):
        for j in range(n1):
            M[i,j] = 1
            for k in range(m2):
                tmp = pow(W[k,j],X[i,k],p)
                M[i,j] = M[i,j]*tmp % p
    return M

def matrix_exp_right(W,X,p):
    """
    Function allows to raise matrix W by matrix power X from the right side modulo p
    """
    M = np.matrix()
    _, n2 = W.shape
    m1, _ = X.shape
    for i in range(n2):
        for j in range(m1):
            M[i,j] = 1
            for k in range(n2):
                tmp = pow(W[i,k],X[k,j],p)
                M[i,j] = M[i,j]*tmp % p
    return M