# encoding: utf-8
# using Python 3.7

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.unsqueeze
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size()) # noisy y data (tensor), shape=(100, 1)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden_layer, n_output):
        super(Net, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_features, n_hidden_layer)
        self.predict_layer = torch.nn.Linear(n_hidden_layer, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        x = self.predict_layer(x)
        return x

net = Net(1, 10, 1)
print(net)

plt.ion() # realtime draw
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5) # lr: learning rate
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
