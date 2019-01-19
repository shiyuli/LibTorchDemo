# encoding: utf-8
# using Python 3.7

import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

t_out = torch.mean(tensor * tensor) # x^2
v_out = torch.mean(variable * variable)

print(t_out)
print(v_out)

# math: v_out = 1/4*sum(variable*variable)
# math: d(v_out)/d(variable) = 1/4*2*variable = variable/2
v_out.backward()
print(variable.grad)

# variable.data is the form of Tensor,
# while variable is the form of Variable,
# so you can not use variable.numpy() but variable.data.numpy()
# to get numpy matrix.
print(variable.data)
print(variable.data.numpy())
