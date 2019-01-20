# encoding: utf-8
# using Python 3.7

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1) # reproducible

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)    # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)   # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor) # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1),).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# The code below is deprecated in Pytorch 0.4. Now, asutograd directly supports tensors.
# x, y = Variable(x), Variable(y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_feature, n_hidden)
        self.output_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

net = Net(2, 10, 2)
# [0, 1] : x1, y1
# [1, 0] : x0, y0
print(net) # print the net architecture

plt.ion() # realtime draw
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02) # lr: learning rate
loss_func = torch.nn.CrossEntropyLoss() # the target label is not an one-hotted

for t in range(100):
    output = net(x)                 # input x and predict based on x
    loss = loss_func(output, y)     # must be (1. nn output, 2. target), the target label is not one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(output), 1)[1]
        print(prediction)
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
