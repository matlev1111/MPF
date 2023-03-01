import matplotlib.pyplot as plt
import pickle
import numpy as np
from PIL import Image

def Print_pic(C, s1, s2, d3):
    m = C[:,:,0].shape[0]
    if(d3):
        rgb = 3
        FC = np.zeros([s1,s2,3])
    else:
        rgb = 1
        FC = np.zeros([s1,s2])
    indx = 0
    for k in range(rgb):
        for i in range(s1//m):
            for j in range(s2//m):
                if(d3):
                    FC[i*m:(i+1)*m,j*m:(j+1)*m,k] = C[:,:,indx]
                else:
                    FC[i*m:(i+1)*m,j*m:(j+1)*m] = C[:,:,indx]
                indx += 1
    return np.array(FC)

file_name = 'coloredChips'
#Algorithm: M16, Esak
file_folder = 'Esak'
#Mode: CTR, CBC
mode = 'CTR'
#RGB: 1 - RGB; 0 - Grayscale
rgb_TF = 1

file = open(f'res/{mode}/{file_folder}/{file_name}', 'rb')
data, M1, s1, s2 = pickle.load(file)
file.close()
if(mode == 'CBC'):
    ans = Print_pic(data[:,:,1:], s1, s2, rgb_TF)
elif(mode == 'CTR'):
    ans = Print_pic(data[:,:,:], s1, s2, rgb_TF)

fig, ax = plt.subplots(figsize =(10, 7))
unique, counts = np.unique(np.array(ans), return_counts=True)
ax.bar(unique, counts,width=1,  edgecolor='black')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()