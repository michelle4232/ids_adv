import torch
import numpy as np


# create a numpy array's shape with 5, 3
test_normal = np.random.rand(5, 3)
print("test_normal: ", test_normal)
print("test_normal shape: ", test_normal.shape)


print("======================")
print("test_batch_normal", test_normal[np.random.randint(0,len(test_normal), 2)])
# create a tensor from the numpy array
print("======================")
print(np.random.uniform(0,1,(64,9)).shape)
# create a np array for fix number with shape (64, 9)
tmp = arr = np.full((64, 9), 7)
print("tmp: ", tmp)