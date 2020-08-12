import torch
import numpy as np
from pprint import pprint


class Dimension_operations:
    def __init__(self, embeddings, idx2word, word2idx, EMBED_DIM, labels):
        self.embeddings = embeddings
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.EMBED_DIM = EMBED_DIM
        self.labels = labels

    def no_hid_layer_labeling(self, model):
        weights = model.emb_layer.weight
        for pred_ind, label in enumerate(self.labels):
            print('\ntop dimensions for label', label)
            influent_weights = weights[pred_ind]
            _, dim_idx = torch.sort(influent_weights, descending=True)

            for idx in dim_idx[:5]:
                print('Dimension:', idx.item(), 'weight:', influent_weights[idx].item())
                top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][idx], reverse=True)[:5]
                # take the pair (word, value in that dimension) for top results
                # top_emb = [(idx2word[emb_idx], embeddings[emb_idx][idx]) for emb_idx, emb in top_emb]
                top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
                pprint(top_emb)

    def no_hid_layer_labeling_sum(self, model):
        weights = model.emb_layer.weight
        for pred_ind, label in enumerate(self.labels):
            print('top dimensions for label', label)
            influent_weights = weights[pred_ind]
            influent_weights, dim_idx = torch.sort(influent_weights, descending=True)

            # dim_idx contain indices of the most important weights for a certain class label.
            # dim_idx = 0 corresponds to the 1st dimension of the embeddings

            # sort embeddings from highest to smallest
            # according to the highest valued sums over the top dim_idx dimensions for a certain label
            # x[1] because we are enumerating, therefore x[1] are actual embeddings
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim_idx[:50].cpu().numpy()].sum(),
                             reverse=True)[:20]
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            pprint(top_emb)

            # influent_weights, dim_idx = torch.sort(influent_weights)
            # low_emb = sorted(enumerate(embeddings), key=lambda x: x[1][dim_idx[:10].cpu().numpy()].sum())[:10]
            # low_emb = [(idx2word[emb_idx]) for emb_idx, emb in low_emb]
            # pprint(low_emb)

    def single_layer_labeling(self, model):
        dim_label_pairs = dict()  # embedding dimension: (label, value)

        weights_in = model.emb_layer.weight  # [500, 1000]
        weights_proj = model.projection_layer.weight  # [4, 500]
        influent_weights_in_layer = torch.t(weights_in)  # [1000, 500]
        www = torch.matmul(influent_weights_in_layer, torch.t(weights_proj))  # [1000, 4]
        for weights_list_idx, weights_list in enumerate(www):
            best_label = torch.argmax(weights_list).item()
            best_value = weights_list[best_label].item()
            dim_label_pairs[weights_list_idx] = (best_label, best_value)

        dim_label_pairs = {k: v for k, v in sorted(dim_label_pairs.items(), key=lambda x: x[1][1], reverse=True)}

        for dim, pair in dim_label_pairs.items():
            print('Dimension:', dim)
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:5]
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            print(top_emb, '--->', self.labels[pair[0]], '( weight value:', pair[1], ')')

    def dimension_labeling_deep(self, model, N_HIDDEN_LAYERS):
        dim_label_pairs = dict()  # embedding dimension: (label, value)

        layer_mul = torch.t(model.hidden_layers[0].weight)
        for i in range(1, N_HIDDEN_LAYERS):
            next_layer = torch.t(model.hidden_layers[i].weight)
            layer_mul = torch.matmul(layer_mul, next_layer)
            print(layer_mul)

        for weights_list_idx, weights_list in enumerate(layer_mul):
            best_label = torch.argmax(weights_list).item()
            best_value = weights_list[best_label].item()
            dim_label_pairs[weights_list_idx] = (best_label, best_value)

        dim_label_pairs = {k: v for k, v in sorted(dim_label_pairs.items(), key=lambda x: x[1][1], reverse=True)}

        for dim, pair in dim_label_pairs.items():
            print('Dimension:', dim)
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:5]
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            print(top_emb, '--->', self.labels[pair[0]], '( weight value:', pair[1], ')')

    def input_word_dimension_labeling_no_hid(self, model, word: str):
        word_idx = self.word2idx[word]
        word_emb = np.array(self.embeddings[word_idx])
        data = torch.tensor(word_emb).float().to(model.device)
        pred = torch.argmax(model.forward(data))
        label = self.labels[pred]
        print('Prediction for word:', word, '-', label)

        # emb_layer has shape [4, 1000]
        weights = model.emb_layer.weight[pred]  # [1000]

        # multiply the weights linked to the predicted outupt value, with the input values.
        # With this one should get a more "accurate" weight value based on activations.
        pred_weights = torch.mul(data, weights)
        sorted_weights, weights_idx = torch.sort(pred_weights, descending=True)
        for i in range(10):
            print('Dimension:', weights_idx[i].item(), '--', sorted_weights[i].item())
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][weights_idx[i]], reverse=True)[:10]
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            print(top_emb)

    def input_word_dim_labeling_single(self, model, word: str):
        word_idx = self.word2idx[word]
        word_emb = np.array(self.embeddings[word_idx])
        data = torch.tensor(word_emb).float().to(model.device)
        in_weights = model.emb_layer.weight  # [500, 1000]
        out_weights = torch.t(model.projection_layer.weight)  # [500, 4] (it's transposed)
        pred_ind = torch.argmax(model.forward(data))

        print('Prediction for word %s:%s' % (word, self.labels[pred_ind]))

        print(data.shape)
        print(in_weights.shape)
        print(out_weights.shape)

        activated_emb_to_out_weights = torch.mul(data, in_weights)
        activated_emb_to_out_weights = torch.t(activated_emb_to_out_weights)

        activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)
        print(activated_emb_to_out_weights.shape)

        # TODO: Sum column values, add bias and check that obtained values are the same as the common activation vals.

        dimension_label_value_list = []
        for i in range(self.EMBED_DIM):
            dim_values = activated_emb_to_out_weights[i]
            label_ind = dim_values.argmax()
            dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))

        # TODO: Instead of getting biggest values for a certain dimension, take the most similar to the input ones.
        # TODO: This way we may understand why something seems to be misclassified.
        # TODO: E.g. top values are for words [nfl, basketball...] while similar values are [martian, astronauts...] (NOT SURE BUT MAY HAPPEN).
        dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        for i in range(len(dimension_label_value_list)):
            dim = dimension_label_value_list[i][0]
            label = self.labels[dimension_label_value_list[i][1]]
            value = dimension_label_value_list[i][2]
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:5]
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            print(
                'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
            print(top_emb)
        # label = labels[pred]

    def input_word_dimension_labeling_deep(self, model, N_HIDDEN_LAYERS, word: str):
        word_idx = self.word2idx[word]
        word_emb = np.array(self.embeddings[word_idx])
        data = torch.tensor(word_emb).float().to(model.device)
        pred = torch.argmax(model.forward(data))
        label = self.labels[pred]
        print('Prediction for word:', word, '-', label)
        in_weights = model.hidden_layers[0].weight  # [500,1000]

        # activated_weights = torch.mul(data, in_weights)  # [500,1000]
        # activated_weights_len = len(activated_weights)
        #
        # activated_weights = torch.t(activated_weights)  # [1000,500]

        mul_weights = torch.mul(data, in_weights)  # [500,1000]
        activated_weights_len = len(mul_weights)

        mul_weights = torch.t(mul_weights)  # [1000,500]

        # sum bias to the list of activated weights.
        mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)

        activated_weights = torch.relu(torch.sum(mul_weights, 0))
        for idx, val in enumerate(activated_weights):
            if val == 0.0:
                mul_weights[:, idx] = 0.0

        # sum bias to the list of activated weights.

        # mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)

        # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)

        for i in range(1, N_HIDDEN_LAYERS):
            next_layer = torch.t(model.hidden_layers[i].weight)
            mul_weights = torch.matmul(mul_weights, next_layer)  # it will be [1000,4] in the end
            activated_weights = torch.relu(torch.sum(mul_weights, 0))
            bias = model.hidden_layers[i].bias.div(activated_weights_len)  # it will be length = 4 in the end
            mul_weights = mul_weights + bias

            for idx, val in enumerate(activated_weights):
                if val == 0.0:
                    mul_weights[:, idx] = 0.0

        dimension_label_value_list = []
        for i in range(self.EMBED_DIM):
            dim_values = mul_weights[i]
            label_ind = dim_values.argmax()
            dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))

        dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        for i in range(len(dimension_label_value_list)):
            dim = dimension_label_value_list[i][0]
            label = self.labels[dimension_label_value_list[i][1]]
            value = dimension_label_value_list[i][2]
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:5]
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            print(
                'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
            print(top_emb)


    def emb_all_input_word_dimension_labeling_deep(self, model, N_HIDDEN_LAYERS, words: str):
        emb_list = []
        for word in words.split(' '):
            emb = self.emb_all.get(word)
            if emb is not None:
                emb_list.append(emb)
            else:
                print('no embedding for', word)
        mean_emb = np.mean(np.array(emb_list), axis=0)
        data = torch.tensor(mean_emb).float().to(model.device)
        pred = torch.argmax(model.forward(data))
        label = self.labels[pred]
        print('Prediction for:', words, '-', label)
        in_weights = model.hidden_layers[0].weight
        print(in_weights.shape)

        activated_weights = torch.mul(data, in_weights)

        activated_weights = torch.t(activated_weights)

        # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)

        for i in range(1, N_HIDDEN_LAYERS):
            next_layer = torch.t(model.hidden_layers[i].weight)
            activated_weights = torch.matmul(activated_weights, next_layer)
            print(next_layer.shape)

        dimension_label_value_list = []
        for i in range(self.EMBED_DIM):
            dim_values = activated_weights[i]
            label_ind = dim_values.argmax()
            dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))

        dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        for i in range(len(dimension_label_value_list)):
            dim = dimension_label_value_list[i][0]
            label = self.labels[dimension_label_value_list[i][1]]
            value = dimension_label_value_list[i][2]
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:5]
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            print(
                'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
            print(top_emb)
