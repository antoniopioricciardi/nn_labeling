import os
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from torchtext.data.utils import get_tokenizer

words = []
vocab = set()
word2idx = dict()
embeddings = []
emb_all = dict()
class_distribution = dict()

# vocab.add('unk')
# word2idx['unk'] = 0
# embeddings[0] = []

tokenizer = get_tokenizer("basic_english")
stop_words = set(stopwords.words('english'))


def load_embeddings(filename):
    is_fasttext = 'cc.en' in filename
    with open(filename) as f:
        if is_fasttext:
            f.readline()
        for c, line in enumerate(f):
            line = line.split()
            word = line[0]
            if is_fasttext:
                if len(word) < 5:
                    continue
            emb = np.array(line[1:], dtype=np.float)
            emb_all[word] = emb
            if is_fasttext:
                if c == 200000:
                    break
    return emb_all


def preprocess(line):
    label = int(line[:2].replace('"', '')) - 1
    words_list = tokenizer(line[4:].replace('\\', ' ').replace('-', ' '))
    words_list = [w for w in words_list if not w in stop_words and w not in string.punctuation]
    return words_list, label


def preproces_bigger_dataset(line):
    words_list = tokenizer(line.replace('\\', ' ').replace('-', ' '))
    words_list = [w for w in words_list if not w in stop_words and w not in string.punctuation]
    return words_list


def set_word2idx(word):
    """
    Given a word:
    - If the word is new then add it and its related index to word2idx dict and return its index.
    - If the word already exists, then simply return its index
    :param word:
    :return: word2idx index of a word
    """
    idx = word2idx.get(word)
    if idx is None:
        idx = len(word2idx)
        word2idx[word] = idx
        embeddings.append(emb_all[word])
    return idx


def add_to_word_class_distribution(word, label):
    if word not in class_distribution.keys():
        c_dict = dict()
        for i in range(4):
            c_dict[i] = 0
        class_distribution[word] = c_dict
    class_distribution[word][label] += 1


def create_train_dataset(train_path):
    train_dataset = []  # will contain pairs (words_list, label) where label is the class value for the phrase
    with open(train_path) as f:
        for line in f:
            train_words = []
            words_list, label = preprocess(line)
            for word in words_list:
                emb = emb_all.get(word)
                if emb is not None:  # if there is an embedding for the word
                    vocab.add(word)
                    # add_to_word_class_distribution(word, label)
                    idx = set_word2idx(word)  # get its index
                    # embeddings[idx] = emb  # add the embedding to the dict, in the corresponding index
                    train_words.append(idx)
            train_dataset.append((np.array(train_words), label))
    idx2word = {w: k for k, w in word2idx.items()}
    return vocab, word2idx, idx2word, train_dataset, class_distribution, np.array(embeddings)

def create_train_dataset_bigger(directory):
    """
    Text is formatted differently. Each file is a training instance. A file can be formatted with newlines.
    :param directory:
    :return:
    """
    train_dataset = []
    labels = []
    labels_cnt = 0
    for domain in os.listdir(directory):
        f_cnt = 0
        if domain.endswith("Store"):
            continue
        labels.append(domain)
        for f in os.listdir(os.path.join(directory, domain)):
            if f.endswith(".txt"):
                #if f_cnt == 500:
                #    continue
                #f_cnt += 1
                with open(os.path.join(directory, domain, f), encoding="utf8") as file:
                    train_words = []
                    for line in file:
                        words_list = preproces_bigger_dataset(line)
                        for word in words_list:
                            emb = emb_all.get(word)
                            if emb is not None:  # if there is an embedding for the word
                                vocab.add(word)
                                # add_to_word_class_distribution(word, domain)
                                idx = set_word2idx(word)  # get its index
                                # embeddings[idx] = emb  # add the embedding to the dict, in the corresponding index
                                train_words.append(idx)
                if train_words:  # if the file is not empty for some reason, then add its preprocessed content to dataset
                    train_dataset.append((np.array(train_words), labels_cnt))
        labels_cnt += 1

    idx2word = {w: k for k, w in word2idx.items()}
    return vocab, word2idx, idx2word, train_dataset, class_distribution, np.array(embeddings), labels


def create_dev_dataset_bigger(directory):
    """
    Dev dataset
    Text is formatted differently. Each file is a training instance. A file can be formatted with newlines.
    :param directory:
    :return:
    """
    dev_dataset = []
    labels = []
    labels_cnt = 0
    for domain in os.listdir(directory):
        f_cnt = 0
        if domain.endswith("Store"):
            continue
        labels.append(domain)
        for f in os.listdir(os.path.join(directory, domain)):
            if f.endswith(".txt"):
                if f_cnt == 100:
                    continue
                f_cnt += 1
                with open(os.path.join(directory, domain, f), encoding="utf8") as file:
                    dev_words = []
                    for line in file:
                        words_list = preproces_bigger_dataset(line)
                        for word in words_list:
                            if word in vocab:
                                # if we have this word in our vocabulary, then add it to the dev dataset
                                idx = word2idx[word]
                                dev_words.append(idx)
                if dev_words:  # if the file is not empty for some reason, then add its preprocessed content to dataset
                    dev_dataset.append((np.array(dev_words), labels_cnt))
        labels_cnt += 1
    return dev_dataset


def get_word_idx(word):
    idx = word2idx.get(word)
    if idx is None:
        return len(word2idx)
    return idx


def generate_batches(data, batch_size, embeddings):
    batches = []
    batch = []
    for i, pair in enumerate(data):
        phrase = pair[0]
        label = pair[1]
        batch.append((np.mean(embeddings[phrase], axis=0), label))
        if (i+1) % batch_size == 0:
            batches.append(batch)
            batch = []
        if i + 1 + batch_size > len(data):
            break
    return batches



# def load_embeddings(filename):
#     with open(filename) as f:
#         for idx, line in enumerate(f):
#             line = line.split()
#             word = line[0]
#             words.append(word)
#             word2idx[word] = idx
#             emb = np.array(line[1:], dtype=np.float)
#             embeddings.append(emb)
#     return words, word2idx, embeddings
