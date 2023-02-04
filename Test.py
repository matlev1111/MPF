from main_func import *

W = np.matrix([[1,2],[3,4]])
p = np.int32(7)
X = np.matrix([[1,2],[3,4]])

print(matrix_exp_left(W,X,p))
print(matrix_exp_right(W,X,p))
