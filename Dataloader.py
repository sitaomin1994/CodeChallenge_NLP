import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def encode_titles(titles, embed=True):
    t = Tokenizer(filters='', lower=False)
    t.fit_on_texts(titles)

    vocabulary = t.word_index
    vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
    vocabulary_inv[0] = '<PAD/>'
    vocabulary['<PAD/>'] = 0
    # convert title to sequences
    sequences = t.texts_to_sequences(titles)

    # pad sequences
    padded_seq = pad_sequences(sequences, padding="post", truncating="post")

    return np.array(padded_seq), vocabulary, vocabulary_inv


class Eluvio(Dataset):

    def __init__(self, data, titles, dense_col=None, target_col='log_up_votes', padded=True,
                 transform=None):
        """
        Args:
            data: dataframe contains features and labels
            target_col : target columns name
        """
        self.data = data
        self.titles = titles
        self.titles = np.array(self.titles)
        #print(type(self.titles))
        #print(self.titles.shape)

        if dense_col is None:
            dense_col = ['over_18', 'title_len', 'title_num_char']

        # feature information
        self.author = self.data['author_code'].values.reshape(-1,1)
        self.year = self.data['year_code'].values.reshape(-1,1)
        self.month = self.data['month'].values.reshape(-1,1)
        self.day = self.data['day'].values.reshape(-1,1)
        self.hour = self.data['hour'].values.reshape(-1,1)
        self.weekday = self.data['weekday'].values.reshape(-1,1)
        self.n_week = self.data['week'].values.reshape(-1,1)
        self.n_day = self.data['dayofyear'].values.reshape(-1,1)
        # self.quarter = data[[col for col in data if col.startswith('quarter')]].values
        # other dense features
        self.dense = data[dense_col].astype('float32').values

        # whether to pad sequence
        self.padded = padded

        # label
        self.label = data[target_col].values.reshape(-1,1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sentence = self.titles[idx]
        if self.padded:
            length_arr = np.where(sentence == 0)[0]
            if length_arr.shape[0] == 0:
                length = sentence.shape[0]
            else:
                length = np.where(sentence == 0)[0][0]
        else:
            length = sentence.shape[0]

        sample = {'title': self.titles[idx],
                  'length': length,
                  'year': self.year[idx],
                  'author': self.author[idx],
                  'month': self.month[idx],
                  'day': self.day[idx],
                  'hour': self.hour[idx],
                  'weekday': self.weekday[idx],
                  'n_week': self.n_week[idx],
                  'n_day': self.n_day[idx],
                  # 'quarter': self.quarter[idx],
                  'dense': self.dense[idx],
                  'label': self.label[idx]}

        return sample

    def _oneHotEncode(self, X):
        X = X.values.reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(X)
        return enc.transform(X).toarray()



class ToTensor(object):
    """Convert arrays in sample to Tensors."""

    def __call__(self, sample):
        result = {}

        for key, value in sample.items():
            result[key] = torch.from_numpy(value)

        return result


def split_train_test(X, y, titles):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=21)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=21)

    titles_train = titles[X_train.index]
    titles_validation = titles[X_val.index]
    titles_test = titles[X_test.index]

    return X_train, X_val, X_test, titles_train, titles_validation, titles_test

