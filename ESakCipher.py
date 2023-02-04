"""
function numbers = genG(p)
q = (p-1)/2;
%q = 2^19-1;
index = 3;
while(powermod(index,q,p)~= 1 && mod(powermod(index,1,p)*powermod(index,2,p),p)~=powermod(index,1,p))
index = index + 1;
end
numbers = [];
for i=1:q
    numbers = [numbers powermod(index,i,p)];
end
end
"""
import numpy as np
from sympy.ntheory import discrete_log

def GenG(p):
    q = (p-1)//2
    index = 3
    while (np.mod(np.power(index,q),p) != 1 and np.mod(np.mod(np.power(index,1),p) * np.mod(np.power(index,2),p),p) != np.mod(np.power(index,1),p)):
        index = index + 1
    numbers = []
    for i in range(q):
        numbers.append(np.mod(np.power(index,i),p))
    return index, numbers


def Gmap(index, mat):
    m = len(mat)
    M = np.zeros(m)
    fm = np.vectorize(lambda x: index[x])
    M=fm(mat)
    return M

def imap(ind, num,p):
    m = len(num)
    M = np.zeros(m)
    fm = np.vectorize(lambda x: discrete_log(p, x, ind))
    M=fm(num)
    return M

