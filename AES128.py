from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
import os 
import numpy as np
from PIL import Image
from timeit import default_timer as timer
from datetime import timedelta

IV = os.urandom(16)
key = get_random_bytes(16)

with open("images/linux.png","rb") as file1:
    data = file1.read()
    #print(len(data))

    cipher = AES.new(key, AES.MODE_CBC)
    start = timer()
    ciphertext = cipher.encrypt(pad(data,16))
    end = timer()
    print(timedelta(seconds=end-start))
print(ciphertext)
#ciphertext, tag = cipher.encrypt_and_digest(data)
#nonce = cipher.nonce
#print(int.from_bytes(ciphertext,'big'))