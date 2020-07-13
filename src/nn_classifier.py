import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, lr):
        super().__init__()
        #self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.emb_layer = nn.Linear(input_dim, output_dim)
        # self.init_weights()

        # self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.loss = nn.MSELoss()

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights(self):
        initrange = 0.5
        self.emb_layer.weight.data.uniform_(-initrange, initrange)
        self.emb_layer.bias.data.zero_()

    def forward(self, words):
        x = self.emb_layer(words)
        return x


class DeepLinear(nn.Module):
    def __init__(self,
                 input_dim: int,
                 target_dim: int,
                 projection_layers: int,
                 dropout: float,
                 activation: nn.modules.activation = None,
                 lr=0.001):
        super().__init__()
        assert input_dim > target_dim
        projection_inputs = [input_dim // 2 ** i for i in range(0, projection_layers)]
        projection_outputs = [input_dim // 2 ** i for i in range(1, projection_layers)] + [target_dim]

        self.dropout = nn.Dropout(p=dropout)

        self.activation = activation

        self.emb_layer = nn.Linear(input_dim, input_dim // 2)
        print(self.emb_layer.weight.shape)

        self.projection_layer = nn.Linear(input_dim // 2, target_dim)
        print(self.projection_layer.weight.shape)
        self.batch_norm_in = nn.BatchNorm1d(input_dim)
        self.batch_norm_mid = nn.BatchNorm1d(input_dim // 2)


        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # self.projection_layers = nn.ModuleList(
        #     [
        #         nn.Linear(input_dim, output_dim)
        #         for input_dim, output_dim in zip(projection_inputs, projection_outputs)
        #     ]
        # )
        # self.batch_norms = nn.ModuleList(
        #     [nn.BatchNorm1d(input_dim) for input_dim in projection_inputs]
        # )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data = self.batch_norm_in(data) #.transpose(1, 2)).transpose(1, 2)
        # if self.activation is not None:
        #    data = self.activation(data)
        data = self.emb_layer(data)
        data = self.dropout(data)
        # data = F.relu(data)

        # data = self.batch_norm_mid(data) #.transpose(1, 2)).transpose(1, 2)
        # if self.activation is not None:
        #     data = self.activation(data)
        data = F.relu(data)
        # data = self.dropout(data)
        data = self.projection_layer(data)
        data = F.relu(data)
        return data
    # def forward(self, data: torch.Tensor) -> torch.Tensor:
    #     for projection_layer, batch_norm in zip(self.projection_layers, self.batch_norms):
    #         # TODO: shape check for transpose
    #         data = batch_norm(data.transpose(1, 2)).transpose(1, 2)
    #
    #         if self.activation is not None:
    #             data = self.activation(data)
    #
    #         data = self.dropout(data)
    #         data = projection_layer(data)
    #
    #     return data


class DoubleLinear(nn.Module):
    def __init__(self,
                 input_dim: int,
                 target_dim: int,
                 projection_layers: int,
                 dropout: float,
                 activation: nn.modules.activation = None,
                 lr=0.001):
        super().__init__()
        assert input_dim > target_dim
        projection_inputs = [input_dim // 2 ** i for i in range(0, projection_layers)]
        projection_outputs = [input_dim // 2 ** i for i in range(1, projection_layers)] + [target_dim]

        self.dropout = nn.Dropout(p=dropout)

        self.activation = activation

        self.first_layer = nn.Linear(input_dim, input_dim // 2)
        # print(self.emb_layer.weight.shape)

        self.second_layer = nn.Linear(input_dim // 2, 100)
        self.out_layer = nn.Linear(100, 4)
        self.batch_norm_in = nn.BatchNorm1d(input_dim)
        self.batch_norm_mid = nn.BatchNorm1d(input_dim // 2)


        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # self.projection_layers = nn.ModuleList(
        #     [
        #         nn.Linear(input_dim, output_dim)
        #         for input_dim, output_dim in zip(projection_inputs, projection_outputs)
        #     ]
        # )
        # self.batch_norms = nn.ModuleList(
        #     [nn.BatchNorm1d(input_dim) for input_dim in projection_inputs]
        # )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data = self.batch_norm_in(data) #.transpose(1, 2)).transpose(1, 2)
        # if self.activation is not None:
        #    data = self.activation(data)
        data = self.first_layer(data)
        data = self.dropout(data)
        # data = F.relu(data)

        # data = self.batch_norm_mid(data) #.transpose(1, 2)).transpose(1, 2)
        # if self.activation is not None:
        #     data = self.activation(data)
        data = F.relu(data)
        # data = self.dropout(data)
        data = self.second_layer(data)
        data = F.relu(data)

        data = self.out_layer(data)
        data = F.relu(data)
        return data