# encoding: utf-8
# using Python 3.7

import torch
import torch.nn.functional as F

# the ordinary way to build neural networks
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_feature, n_hidden)
        self.output_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

net1 = Net(2, 10, 2)
print(net1)

# an easy and fast way to build neural networks
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),    # hidden_layer
    torch.nn.ReLU(),           # activation function
    torch.nn.Linear(10, 2),    # output_layer
)

print(net2)
