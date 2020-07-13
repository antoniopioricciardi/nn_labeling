from pprint import pprint

import torch
import torchtext
import time

import time
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split

from torchtext.datasets import text_classification
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from src.embedding_handler import *
from src.trainer import Trainer
from src.nn_classifier import NNClassifier, DeepLinear, DoubleLinear

EMB_PATH = '../embeddings/SPINE_word2vec.txt'
# EMB_PATH = '../embeddings/cc.en.300.vec'
# EMB_PATH = '../embeddings/word2vec_original_15k_300d_train.txt'
emb_all = load_embeddings(EMB_PATH)

NGRAMS = 1


import os
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
# train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='./.data', vocab=None)
# vocab = train_dataset.get_vocab()

BATCH_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 1000
# NUN_CLASS = len(train_dataset.get_labels())

N_EPOCHS = 15
data_path = "./.data/ag_news_csv"
vocab, word2idx, idx2word, train_dataset, class_distribution, embeddings = create_train_dataset(os.path.join(data_path, 'train.csv'))
labels_counter = dict()
# for i in range(4):
#     labels_counter[i] = 0
# for _, label in train_dataset:
#     labels_counter[label] += 1
# print(labels_counter)

tokenizer = get_tokenizer("basic_english")

batches_list = generate_batches(train_dataset, BATCH_SIZE, embeddings)
model = NNClassifier(EMBED_DIM, 4, 0.001)
#model = DeepLinear(EMBED_DIM, 4, 1, 0.1, None)
# model = DoubleLinear(EMBED_DIM, 4, 1, 0.8, None)
# I THINK that a high dropout prob leads to a better labeling

train_len = int(len(batches_list)*0.9)
valid_len = len(batches_list) - train_len

batches_list_train, batches_list_val = random_split(batches_list, [train_len, valid_len])

for epoch in range(N_EPOCHS):
    start_time = time.time()
    tot_train_len = 0
    tot_val_len = 0
    train_loss = 0
    train_acc = 0
    val_acc = 0

    for batch in batches_list_train:
        model.zero_grad()
        tot_train_len += len(batch)
        data = torch.from_numpy(np.array([np.array(el[0]) for el in batch])).to(model.device)
        labels = torch.tensor([el[1] for el in batch]).to(model.device)
        pred = model(data.float())
        # print(pred.argmax(1), '-', labels)

        loss = model.loss(pred, labels)
        train_loss += loss
        train_acc += (pred.argmax(1) == labels).sum().item()
        loss.backward()
        model.optimizer.step()

    with torch.no_grad():
        for batch in batches_list_val:
            tot_val_len += len(batch)
            data = torch.from_numpy(np.array([np.array(el[0]) for el in batch])).to(model.device)
            labels = torch.tensor([el[1] for el in batch]).to(model.device)
            pred = model(data.float())
            val_acc += (pred.argmax(1) == labels).sum().item()
        # Adjust the learning rate
    # model.scheduler.step()

    train_loss = train_loss / tot_train_len
    train_acc = train_acc / tot_train_len
    val_acc = val_acc / tot_val_len

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    #print(f'\tLoss: {val_acc:.4f}(valid)\t|\tAcc: {val_acc * 100:.1f}%(valid)')
    print(f'\tAcc: {val_acc * 100:.1f}%(valid)')

# print(pred[len(pred)-1])
# print(pred)

data, label = batch[len(batch)-1]  # list, int

pred_ind = 1
# weights = model.emb_layer.weight


labels = ['World', 'Sports', 'Business', 'Sci/Tech']


# print(model.projection_layer.weight[0].shape)
# max_ind = model.projection_layer.weight[0].argmax()
# print(max_ind, '-', model.projection_layer.weight[0][max_ind])
# print(model.emb_layer.weight[max_ind])
# exit(4)

import torch

dim_label_pairs = dict()  # embedding dimension: (label, value)
# weights_proj = model.projection_layer.weight
# weights_in = model.emb_layer.weight

# influent_weights_in_layer = torch.t(weights_in)  # [1000, 500]


def single_layer_labeling():
    dim_label_pairs = dict()  # embedding dimension: (label, value)

    weights_in = model.emb_layer.weight
    weights_proj = model.projection_layer.weight
    influent_weights_in_layer = torch.t(weights_in)  # [1000, 500]

    www = torch.matmul(influent_weights_in_layer, torch.t(weights_proj))  # [1000, 4]
    for weights_list_idx, weights_list in enumerate(www):
        best_label = torch.argmax(weights_list).item()
        best_value = weights_list[best_label].item()
        dim_label_pairs[weights_list_idx] = (best_label, best_value)

    dim_label_pairs = {k: v for k, v in sorted(dim_label_pairs.items(), key=lambda x: x[1][1], reverse=True)}

    for dim, pair in dim_label_pairs.items():
        print('Dimension:', dim)
        top_emb = sorted(enumerate(embeddings), key=lambda x: x[1][dim], reverse=True)[:5]
        top_emb = [(idx2word[emb_idx]) for emb_idx, emb in top_emb]
        print(top_emb, '--->', labels[pair[0]], '( weight value:', pair[1], ')')


def no_hid_layer_labeling_sum():
    weights = model.emb_layer.weight
    for pred_ind, label in enumerate(labels):
        print('top dimensions for label', label)
        influent_weights = weights[pred_ind]
        influent_weights, dim_idx = torch.sort(influent_weights, descending=True)

        # dim_idx contain indices of the most important weights for a certain class label.
        # dim_idx = 0 corresponds to the 1st dimension of the embeddings

        # sort embeddings from highest to smallest
        # according to the highest valued sums over the top dim_idx dimensions for a certain label
        # x[1] because we are enumerating, therefore x[1] are actual embeddings
        top_emb = sorted(enumerate(embeddings), key=lambda x: x[1][dim_idx[:50].cpu().numpy()].sum(), reverse=True)[:20]
        top_emb = [(idx2word[emb_idx]) for emb_idx, emb in top_emb]
        pprint(top_emb)

        # influent_weights, dim_idx = torch.sort(influent_weights)
        # low_emb = sorted(enumerate(embeddings), key=lambda x: x[1][dim_idx[:10].cpu().numpy()].sum())[:10]
        # low_emb = [(idx2word[emb_idx]) for emb_idx, emb in low_emb]
        # pprint(low_emb)


def no_hid_layer_labeling():
    weights = model.emb_layer.weight
    for pred_ind, label in enumerate(labels):
        print('\ntop dimensions for label', label)
        influent_weights = weights[pred_ind]
        _, dim_idx = torch.sort(influent_weights, descending=True)

        for idx in dim_idx[:5]:
            print('Dimension:', idx.item(), 'weight:', influent_weights[idx].item())
            top_emb = sorted(enumerate(embeddings), key=lambda x: x[1][idx], reverse=True)[:5]
            # take the pair (word, value in that dimension) for top results
            # top_emb = [(idx2word[emb_idx], embeddings[emb_idx][idx]) for emb_idx, emb in top_emb]
            top_emb = [(idx2word[emb_idx]) for emb_idx, emb in top_emb]
            pprint(top_emb)


def double_layer_labeling():
    dim_label_pairs = dict()  # embedding dimension: (label, value)

    weights_in = model.first_layer.weight  # [500, 1000]
    weights_sec = model.second_layer.weight  # [100, 500]
    weights_out = model.out_layer.weight  # [4, 100]
    influent_weights_in_layer = torch.t(weights_in)  # [1000, 500]
    print(influent_weights_in_layer.shape)
    ww1 = torch.matmul(influent_weights_in_layer, torch.t(weights_sec))  # [1000, 100]
    ww2 = torch.matmul(ww1, torch.t(weights_out))  # [1000, 4]
    for weights_list_idx, weights_list in enumerate(ww2):
        best_label = torch.argmax(weights_list).item()
        best_value = weights_list[best_label].item()
        dim_label_pairs[weights_list_idx] = (best_label, best_value)

    dim_label_pairs = {k: v for k, v in sorted(dim_label_pairs.items(), key=lambda x: x[1][1], reverse=True)}

    for dim, pair in dim_label_pairs.items():
        print('Dimension:', dim)
        top_emb = sorted(enumerate(embeddings), key=lambda x: x[1][dim], reverse=True)[:5]
        top_emb = [(idx2word[emb_idx]) for emb_idx, emb in top_emb]
        print(top_emb, '--->', labels[pair[0]], '( weight value:', pair[1], ')')


#single_layer_labeling()
#no_hid_layer_labeling()
# no_hid_layer_labeling_sum()
# double_layer_labeling()

word = 'heir'
# word = 'microsoft'
# word = 'apple'
# word = 'macintosh'
# word = 'labor'
word_idx = word2idx[word]
word_emb = np.array(embeddings[word_idx])
data = torch.tensor(word_emb).float().to(model.device)
pred = torch.argmax(model.forward(data))
label = labels[pred]
print('Prediction for word:', word, '-', label)
weights = model.emb_layer.weight[pred]
pred_weights = torch.mul(data, weights)
sorted_weights, weights_idx = torch.sort(pred_weights, descending=True)
for i in range(10):
    print('Dimension:', weights_idx[i].item(), '--', sorted_weights[i].item())
    top_emb = sorted(enumerate(embeddings), key=lambda x: x[1][weights_idx[i]], reverse=True)[:10]
    top_emb = [(idx2word[emb_idx]) for emb_idx, emb in top_emb]
    print(top_emb)
# print(torch.matmul(data, pred_weights))



# '''STA ROBA FUNZIONA'''
# for pred_ind, label in enumerate(labels):
#     print('\ntop dimensions for label', label)
#     influent_weights = weights[pred_ind]
#     influent_weights, dim_idx = torch.sort(influent_weights, descending=True)
#
#     for idx in dim_idx[:5]:
#         print(idx)
#         top_emb = sorted(enumerate(embeddings), key=lambda x: x[1][idx], reverse=True)[:5]
#         top_emb = [(idx2word[emb_idx]) for emb_idx, emb in top_emb]
#         pprint(top_emb)
#
#
#


'''
ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

for token in ngrams_iterator(tokenizer(ex_text_str), 1):
    print(token + ' -- ' + str(vocab[token]))
#for i in range(VOCAB_SIZE):
#    print(i, '-', str(vocab[i]))


dataset_train = os.path.join(data_path, 'train.csv')
min_valid_loss = float('inf')

trainer = Trainer()

for epoch in range(N_EPOCHS):
    print('Training epoch', epoch)
    trainer.train(train_dataset, BATCH_SIZE)
'''