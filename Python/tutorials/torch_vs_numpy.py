# encoding: utf-8
# using Python 3.7

import torch
import numpy as np

################### transform ###################

# Create a numpy matrix.
np_matrix = np.arange(6).reshape((2, 3))

# Transform numpy matrix to torch tensor.
torch_tensor = torch.from_numpy(np_matrix)

# Transform torch tensor to numpy matrix.
np_new_matrix = torch_tensor.numpy()

print(
    '\nTest transform:',
    '\norigional numpy matrix:\n', np_matrix,
    '\nto torch tensor:\n', torch_tensor,
    '\nto numpy matrix:\n', np_new_matrix
)

###################### abs ######################

data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data) # 32bit floating point

print(
    '\nTest abs:',
    '\norigional data:\n', data,
    '\nnumpy:\n', np.abs(data),
    '\ntorch:\n', torch.abs(tensor)
)

############### matrix calculates ###############

data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)

print(
    '\nTest matrix calculate (matrix multiply):',
    '\nnumpy:\n', np.matmul(data, data),
    '\ntorch:\n', torch.mm(tensor, tensor)
)

data = np.array(data)

print(
    '\nTest matrix calculate (matrix dot multiply):',
    '\nnumpy:\n', data.dot(data),
    '\ntorch:\n', tensor.dot(tensor) # throw error, reference link 'https://stackoverflow.com/questions/44524901/how-to-do-product-of-matrices-in-pytorch'
)

#################################################
