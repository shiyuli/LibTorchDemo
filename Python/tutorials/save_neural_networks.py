import torch
import matplotlib.pyplot as plt

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size()) # noisy y data (tensor), shape=(100, 1)

plt.figure(1, figsize=(10, 3))
subplot_count = 0

def save_net(net):
    # save entire neural networks
    torch.save(net, 'net.pkl')

def save_parameters(net):
    # save neural networks parameters
    torch.save(net.state_dict(), 'net_parameters.pkl')

def restore_net():
    return torch.load('net.pkl')

def restore_parameters():
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    net.load_state_dict(torch.load('net_parameters.pkl'))

    return net

def show(prediction, title=None):
    global subplot_count
    
    plt.subplot(131 + subplot_count)
    plt.title(title)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    
    subplot_count += 1
    if subplot_count == 3:
        plt.show()

def main():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),    # hidden_layer
        torch.nn.ReLU(),           # activation function
        torch.nn.Linear(10, 1),    # output_layer
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

    show(prediction, 'Net1')

    save_net(net1)
    save_parameters(net1)

    net2 = restore_net()
    prediction = net2(x)
    show(prediction, 'Net2')

    net3 = restore_parameters()
    prediction = net3(x)
    show(prediction, 'Net3')

if __name__ == '__main__':
    main()
