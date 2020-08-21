import torch
import torchvision
import torch.functional as F
from scipy.ndimage.filters import gaussian_filter

from src.nn_classifier import *

TESTING = True
n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False  # should disable nondeterministic algorithms
torch.manual_seed(random_seed)


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./mnist_data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0,), (1,))  # (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0,), (1,))  # (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

import matplotlib.pyplot as plt
import numpy as np


# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     im = example_data[i][0]
#     I = np.dstack([im, im, im])
#     x = 5
#     y = 5
#     I[x, y, :] = [1, 0, 0]
#     # plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.imshow(I, interpolation='none')
#     print(example_data[i][0])
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# fig
# plt.show()

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

N_HIDDEN_LAYERS = 3
# model = NNClassifier(784, 10, learning_rate)
# model = DeepLinear(784, 10, N_HIDDEN_LAYERS, 0.1)
model = ConvNet()
if TESTING:
    model.load_state_dict(torch.load('./results_mnist_conv/model.pth'))


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        model.optimizer.zero_grad()
        # data = data.view(len(data), 784).to(model.device)
        data = data.to(model.device)
        target = target.to(model.device)
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = model.loss(output, target)
        loss.backward()
        model.optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), './results_mnist_conv/model.pth')
            torch.save(model.optimizer.state_dict(), './results_mnist_conv/optimizer.pth')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            print(data.shape)
            data = data.to(model.device)
            target = target.to(model.device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += model.loss(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test()
if not TESTING:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

def plot_no_layers():
    fig = plt.figure()

    k = 0
    for i in range(5):
        data = example_data[i+2][0].view(784).to(model.device)
        pred_vals = model.forward(data)
        print(pred_vals)
        pred_index = torch.argmax(pred_vals)

        weights_set = model.emb_layer.weight
        biases = model.emb_layer.bias

        mul_weights = torch.mul(data, weights_set)  # [500,1000]
        activated_weights_len = len(mul_weights)

        mul_weights = torch.t(mul_weights)  # [1000,500]

        # sum bias to the list of activated weights.
        mul_weights = mul_weights + biases.div(activated_weights_len)
        # activated_weights = torch.relu(torch.sum(mul_weights, 0))
        # for idx, val in enumerate(activated_weights):
        #     if val == 0.0:
        #         mul_weights[:, idx] = 0.0

        # for el in torch.t(mul_weights):
        #     node = el*255
        #     c = 0
        #     for el in node:
        #         if el == 0.0:
        #             c += 1
        data = example_data[i + 2][0].view(28, 28)

        print('predicted:', pred_index)
        k += 1
        plt.subplot(5, 11, k)
        plt.tight_layout()
        im = data
        plt.imshow(im, interpolation='none')
        plt.xticks([])
        plt.yticks([])
        for j in range(10):
            influent_weights = (torch.t(mul_weights))[j]
            vals, dim_idx = torch.sort(influent_weights, descending=True)

            I = np.dstack([im, im, im])
            for idx in dim_idx[:100]:
                y = idx % 28
                x = idx // 28
                I[x, y, :] = [1, 0, 0]
                #print(example_data[i][0])
            k += 1
            plt.subplot(5, 11, k)
            plt.imshow(I, interpolation='none')

            # plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])

        print('-----------')

    fig
    plt.show()


def plot_deep_linear_distributed_activations():
    fig = plt.figure()

    k = 0
    for data_idx in range(5):
        data = example_data[data_idx + 15][0].view(784).to(model.device)
        pred_vals = model.forward(data)
        print(pred_vals)
        pred_index = torch.argmax(pred_vals)

        in_weights = model.hidden_layers[0].weight  # [500,1000]
        mul_weights = torch.mul(data, in_weights)  # [500,1000]
        activated_weights_len = len(mul_weights)

        mul_weights = torch.t(mul_weights)  # [1000,500]

        # sum bias to the list of activated weights.
        mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)


        activated_weights = torch.relu(torch.sum(mul_weights, 0))
        for idx, val in enumerate(activated_weights):
            if val == 0.0:
                mul_weights[:, idx] = 0.0

        for i in range(1, N_HIDDEN_LAYERS):
            next_layer = torch.t(model.hidden_layers[i].weight)
            mul_weights = torch.matmul(mul_weights, next_layer)  # it will be [1000,4] in the end
            bias = model.hidden_layers[i].bias.div(activated_weights_len)  # it will be length = 4 in the end
            mul_weights = mul_weights + bias
            activated_weights = torch.relu(torch.sum(mul_weights, 0))

            for idx, val in enumerate(activated_weights):
                if val == 0.0:
                    mul_weights[:, idx] = 0.0
        print(torch.sum(mul_weights, 0))  # scores are almost equal to the activated ones. Some error due to computations
        bk_weights = mul_weights
        # print(torch.numel(mul_weights[mul_weights==0]))
        # normalize values in [0, 1]
        # if torch.numel(mul_weights[mul_weights == 0]) != (mul_weights.shape[0] * mul_weights.shape[1]):  # if the whole distributed scores are 0, then do not normalize
        #     print('ahah')
        #     print(data_idx)
        #     print(torch.numel(mul_weights[mul_weights == 0]), (mul_weights.shape[0] * mul_weights.shape[1]))
        #     mul_weights = mul_weights - mul_weights.min()
        #     mul_weights = torch.div(mul_weights, mul_weights.max())

        data = example_data[data_idx + 15][0].view(28, 28)

        print('predicted:', pred_index)
        k += 1
        plt.subplot(5, 11, k)
        plt.tight_layout()
        im = data
        plt.imshow(im, interpolation='none', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])

        for j in range(10):
            influent_weights = (torch.t(mul_weights))[j]
            # print(torch.t(bk_weights)[j])
            vals, dim_idx = torch.sort(influent_weights, descending=True)

            # I = np.dstack([im, im, im])
            I = np.zeros([28, 28])
            for val_idx, idx in enumerate(dim_idx):
                val = vals[val_idx]
                # if val > 0:
                y = idx % 28
                x = idx // 28
                # I[x, y, :] = [0, val, 1-val]
                I[x, y] = val  # 0 is violet, 1 is green, yellow is max val
                # print(example_data[i][0])
            k += 1
            plt.subplot(5, 11, k)
            plt.imshow(I, interpolation='none', vmin=0, vmax=1)
            val_to_print = '%.3f' % (pred_vals[j].item())
            if data_idx == 0:
                plt.title("class {}\n {}".format(j, val_to_print))
            else:
                plt.title("{}".format(val_to_print))

            # plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])

        print('-----------')

    fig
    plt.show()


def plot_deep_linear():
    fig = plt.figure()

    k = 0
    # data = example_data[data_idx + 11][0].view(784).to(model.device)
    # pred_vals = model.forward(data)
    # print(pred_vals)
    # pred_index = torch.argmax(pred_vals)

    in_weights = model.hidden_layers[0].weight  # [500,1000]

    # activated_weights = torch.mul(data, in_weights)  # [500,1000]
    # activated_weights_len = len(activated_weights)
    #
    # activated_weights = torch.t(activated_weights)  # [1000,500]

    # mul_weights = torch.mul(data, in_weights)  # [500,1000]
    activated_weights_len = len(in_weights)

    mul_weights = torch.t(in_weights)  # [1000,500]

    # sum bias to the list of activated weights.
    mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)

    # activated_weights = torch.relu(torch.sum(mul_weights, 0))
    # for idx, val in enumerate(activated_weights):
    #     if val == 0.0:
    #         mul_weights[:, idx] = 0.0


    # sum bias to the list of activated weights.

    # mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)

    # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)

    for i in range(1, N_HIDDEN_LAYERS):
        next_layer = torch.t(model.hidden_layers[i].weight)
        mul_weights = torch.matmul(mul_weights, next_layer)  # it will be [1000,4] in the end
        bias = model.hidden_layers[i].bias.div(activated_weights_len)  # it will be length = 4 in the end
        mul_weights = mul_weights + bias

        # activated_weights = torch.relu(torch.sum(mul_weights, 0))
        #
        # for idx, val in enumerate(activated_weights):
        #     if val == 0.0:
        #         mul_weights[:, idx] = 0.0
    # print(torch.sum(mul_weights, 0))  # scores are similar to the activated ones

    # normalize values in [0, 1]
    mul_weights = mul_weights - mul_weights.min()
    mul_weights = torch.div(mul_weights, mul_weights.max())

    #data = example_data[data_idx + 5][0].view(28, 28)

    #print('predicted:', pred_index)
    k += 1
    plt.tight_layout()
    #im = data

    print(mul_weights.shape)
    for j in range(10):
        im = np.zeros([28, 28])

        influent_weights = (torch.t(mul_weights))[j]
        vals, dim_idx = torch.sort(influent_weights, descending=True)

        # I = np.dstack([im, im, im])
        for val_idx, idx in enumerate(dim_idx[:150]):#[:100]):
            val = vals[val_idx]
            #if val > 0.8:
            y = idx % 28
            x = idx // 28
            # I[x, y, :] = [0, val, 1-val]
            im[x, y] = val
            # print(example_data[i][0])
        plt.subplot(1, 10, j+1)
        plt.imshow(im, interpolation='none')
        plt.xticks([])
        plt.yticks([])
        k += 1
        # plt.subplot(5, 11, k)
        # plt.imshow(I, interpolation='none')
        #
        # # plt.title("Ground Truth: {}".format(example_targets[i]))
        # plt.xticks([])
        # plt.yticks([])

    print('-----------')

    fig
    plt.show()


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def plot_conv_activation():
    model.eval()
    fig = plt.figure()

    k = 0
    for data_idx in range(5):
        data = example_data[data_idx + 15].view(1,1,28,28).to(model.device)

        # data = example_data[data_idx + 15][0].to(model.device)#.view(784).to(model.device)

        # model.fc1.register_forward_hook(get_activation('fc1'))
        #model.fc2.register_forward_hook(get_activation('fc2'))
        model.conv2.register_forward_hook(get_activation('conv2'))
        model.fc1.register_forward_hook(get_activation('fc1'))

        pred_vals = model(data)[0]  # single vector batch
        #print(pred_vals)
        pred_index = torch.argmax(pred_vals)
        # print(pred_index)

        fc1_w = model.fc1.weight  # [500, 1000]
        fc2_w = model.fc2.weight  # [4, 500]

        data = activation['conv2']
        data = F.relu(F.max_pool2d(data, 2))

        data = data.view(-1, 320)
        # bb = torch.relu(torch.matmul(data, torch.t(fc1_w)) + model.fc1.bias)

        bias_div = len(torch.t(fc1_w))

        mul_weights = torch.mul(data, fc1_w)
        mul_weights = torch.t(mul_weights)
        mul_weights = mul_weights + model.fc1.bias.div(bias_div)

        activated_weights = torch.relu(torch.sum(mul_weights, 0))
        for idx, val in enumerate(activated_weights):
            if val == 0.0:
                mul_weights[:, idx] = 0.0


        fc2_w = torch.t(fc2_w)
        mul_weights = torch.matmul(mul_weights, fc2_w)
        mul_weights = mul_weights + model.fc2.bias.div(bias_div)
        print(mul_weights)

        ''''
        We only need to compute "bias_div" once, because the input shape never changes. Therefore each bias
        needs to be divided always by the same quantity
        '''
        activated_weights = torch.relu(torch.sum(mul_weights, 0))
        for idx, val in enumerate(activated_weights):
            if val == 0.0:
                mul_weights[:, idx] = 0.0

        data = example_data[data_idx + 15][0].view(28, 28)
        print('predicted:', pred_index)
        #k += 1
        #plt.subplot(10, 21, k)
        plt.tight_layout()
        im = data
        plt.imshow(im, interpolation='none', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])

        for vec in torch.t(mul_weights):
            k = 0
            matr = vec.view(1,20,4,4)
            print(matr.shape)

            for el in matr[0]:
                print(el.shape)
                el = el.cpu().detach().numpy()
                print(el)
                k+=1
                plt.subplot(5, 4, k)
                plt.imshow(el)
            fig
            plt.show()

        exit(3)




    #     data = example_data[data_idx + 15][0].view(28, 28)
    #
    #     print('predicted:', pred_index)
    #     k += 1
    #     plt.subplot(5, 11, k)
    #     plt.tight_layout()
    #     im = data
    #     plt.imshow(im, interpolation='none', vmin=0, vmax=1)
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     for j in range(10):
    #         influent_weights = (torch.t(mul_weights))[j]
    #         # print(torch.t(bk_weights)[j])
    #         vals, dim_idx = torch.sort(influent_weights, descending=True)
    #
    #         # I = np.dstack([im, im, im])
    #         I = np.zeros([28, 28])
    #         for val_idx, idx in enumerate(dim_idx):
    #             val = vals[val_idx]
    #             # if val > 0:
    #             y = idx % 28
    #             x = idx // 28
    #             # I[x, y, :] = [0, val, 1-val]
    #             I[x, y] = val  # 0 is violet, 1 is green, yellow is max val
    #             # print(example_data[i][0])
    #         k += 1
    #         plt.subplot(5, 11, k)
    #         plt.imshow(I, interpolation='none', vmin=0, vmax=1)
    #         val_to_print = '%.3f' % (pred_vals[j].item())
    #         if data_idx == 0:
    #             plt.title("class {}\n {}".format(j, val_to_print))
    #         else:
    #             plt.title("{}".format(val_to_print))
    #
    #         # plt.title("Ground Truth: {}".format(example_targets[i]))
    #         plt.xticks([])
    #         plt.yticks([])
    #
    #     print('-----------')
    #
    # fig
    # plt.show()




# plot_deep_linear_distributed_activations()

plot_conv_activation()
exit(50)


fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    im = example_data[i][0]
    I = np.dstack([im, im, im])
    x = 5
    y = 5
    I[x, y, :] = [1, 0, 0]
    # plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.imshow(I, interpolation='none')
    print(example_data[i][0])
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
fig
plt.show()

exit(12)