import pickle
import numpy as np
from PIL import Image

#Filename: freely chosen
file_name = 'coloredChips'
#Algorithm: M16, ESak
file_folder = 'AES'
#Mode: CTR, CBC
mode = 'CTR'
#RGB: 1 - RGB; 0 - Grayscale
rgb_TF = 1
rez_path = f"..\Figures\{mode}\AES\P{file_name}_RGB.jpg"

file = open(f'res/{mode}/{file_folder}/{file_name}', 'rb')
mode = 'CTR'
data, M1, s1, s2 = pickle.load(file)

print(data.shape)
file.close()
if(file_folder == 'ESak'):
    p = 563
    q = (p-1)/2
    data = np.mod(data,256)

def Print_pic(C, s1, s2, d3):
    """
    Form array for picture printing
    Inputs:
        C - Ciphertext
        s1 - number of rows
        s2 - number of columns
        d3 - boolean for rgb/grayscale
    Output:
        Reshaped ciphertext for printing
    """
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
    if(d3):
        im = Image.fromarray(FC[:,:,:].astype(np.uint8)).convert('RGB')
    else:
        im = Image.fromarray(FC).convert('L')
    return im

if(mode == 'CBC'):
    ans = Print_pic(data[:,:,1:], s1, s2, rgb_TF)
elif(mode == 'CTR'):
    ans = Print_pic(data[:,:,:], s1, s2, rgb_TF)

ans.save(rez_path)