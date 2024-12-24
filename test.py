import random
import numpy as np
from collections import deque
memory = deque(maxlen=100)
for i in range (5):
    memory.append((i,i+1,i+2,i+3,i+4))
batch = random.sample(memory,2)
random.shuffle(batch)
arr = [1,2,3,4]
mx = np.max(arr)[0]
print(mx)
Y = np.zeros((5, 4))
# Y[0] = 1
# print(Y)