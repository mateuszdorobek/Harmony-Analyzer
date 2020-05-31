import time
import warnings

import tqdm as tqdm

warnings.filterwarnings(action="ignore")
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import gensim
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tqdm.notebook import tqdm


class Metrics(Callback):
    def __init__(self, tr_data, val_data, model):
        self.validation_data = val_data
        self.train_data = tr_data
        self.model = model

    def cosine_distance_sum(self, A, B):
        return sum([1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) for a, b in zip(A, B)])

    def mean_cos_dist_sum(self, X_vali, y_vali):
        y_pred = self.model.predict(X_vali)
        return self.cosine_distance_sum(y_pred, y_vali) / y_vali.shape[0]

    def on_train_begin(self, logs={}):
        self.mean_cos_dist = []
        self.val_mean_cos_dist = []

    def on_epoch_end(self, epoch, logs={}):
        score = self.mean_cos_dist_sum(self.train_data[0][:1000], self.train_data[1][:1000])
        score_val = self.mean_cos_dist_sum(self.validation_data[0][:1000], self.validation_data[1][:1000])
        self.mean_cos_dist.append(score)
        self.val_mean_cos_dist.append(score_val)
        print('\nepoch {},\tloss {:3.4f},\tval_loss {:3.4f}\tmean_cos_dist {:3.4f}\tval_mean_cos_dist {:3.4f}.'.format(
            epoch, logs['loss'], logs['val_loss'], score.item(), score_val.item()))

    def on_train_end(self, logs={}):
        self.mean_cos_dist = {'mean_cos_dist': self.mean_cos_dist}
        self.val_mean_cos_dist = {'val_mean_cos_dist': self.val_mean_cos_dist}

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.validation_data


def get_index_dicts(sentences):
    unique_words = set(x for l in sentences for x in l)
    word2index = dict((c, i) for i, c in enumerate(unique_words))
    index2word = dict((i, c) for i, c in enumerate(unique_words))
    return word2index, index2word


def index_transform(sequences, word2index):
    sequences_indexed = []
    for seq in tqdm(sequences, total=len(sequences), desc='Indexing'):
        seq_indexed = []
        for word in seq:
            seq_indexed.append(word2index[word])
        sequences_indexed.append(seq_indexed)
    return sequences_indexed


def embedding_transform(sequences, model):
    sequences_embedded = []
    for seq in tqdm(sequences, total=len(sequences), desc=str(model).split("(")[0]):
        seq_embedded = []
        for word in seq:
            seq_embedded.append(model.wv[word])
        sequences_embedded.append(np.array(seq_embedded))
    return sequences_embedded


def prep_subsequences(songs, seq_len=3):
    # Remove too short songs
    songs = [song for song in songs if len(song) > seq_len + 1]
    x = []
    y = []
    for song in songs:
        for i in range(len(song)):
            if i < len(song) - seq_len + 1:
                x.append(song[i: i + seq_len])
            else:
                x.append(np.concatenate([song[i: len(song) - 1], song[: (i + seq_len + 1) % len(song)]]))
            y.append(song[(i + seq_len + 1) % len(song)])
    x = np.array(x)
    y = np.array(y)
    return x, y


def get_data(sentences, encoding_type, seq_len):
    if encoding_type == "Indexed":
        word2index, _ = get_index_dicts(sentences)
        words_encoded = index_transform(sentences, word2index)
    else:
        if encoding_type == "FastText":
            embedding_model = FastText(sentences=sentences, min_count=1, size=13, window=2)
        elif encoding_type == "Word2VecCBOW":
            embedding_model = Word2Vec(sentences=sentences, min_count=1, size=13, window=2, sg=0)
        elif encoding_type == "Word2VecSG":
            embedding_model = Word2Vec(sentences=sentences, min_count=1, size=14, window=2, sg=1)
        #         elif encoding_type == "Multihot":
        #             embedding_model = MultihotEmbedding(sentences=sentences)

        words_encoded = embedding_transform(sentences, embedding_model)
    X, y = prep_subsequences(words_encoded, seq_len=seq_len)
    if y.ndim == 1:
        X = np.expand_dims(X, axis=X.ndim)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    return X_train, X_test, y_train, y_test


def get_io_dimensions(X, y):
    input_shape = X.shape[1:]
    if y.ndim == 1:
        output_shape = 1
    elif y.ndim == 2:
        output_shape = y.shape[1]

    return input_shape, output_shape


def get_model(model_name, input_shape, output_shape, activation='tanh'):
    SimpleRNN_shallow = Sequential([SimpleRNN(output_shape, input_shape=input_shape, activation=activation)])
    LSTM_shallow = Sequential([LSTM(output_shape, input_shape=input_shape, activation=activation)])
    GRU_shallow = Sequential([GRU(output_shape, input_shape=input_shape, activation=activation)])
    # -------------Deeper Networks------------------
    GRU32 = Sequential([
        GRU(32, input_shape=input_shape),
        Dropout(0.2),
        Dense(output_shape, activation=None),
    ])
    GRU3264 = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        GRU(32),
        Dropout(0.4),
        Dense(output_shape, activation=None),
    ])
    LSTM32 = Sequential([
        LSTM(32, input_shape=input_shape),
        Dropout(0.2),
        Dense(output_shape, activation=None),
    ])
    LSTM3264 = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(32),
        Dropout(0.4),
        Dense(output_shape, activation=None),
    ])

    MODELS = {
        "SimpleRNN_shallow": SimpleRNN_shallow,
        "LSTM_shallow": LSTM_shallow,
        "GRU_shallow": GRU_shallow,
        "GRU32": GRU32,
        "GRU3264": GRU3264,
        "LSTM32": LSTM32,
        "LSTM3264": LSTM3264,
    }
    ASSERT_MSG = f"Model {model_name} is not availabale, choose one from:\n- " + "\n- ".join(MODELS.keys())
    assert model_name in MODELS.keys(), ASSERT_MSG
    model = MODELS[model_name]
    return model


def test_model(model, X_train, X_test, y_train, y_test, epochs=30):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=7)
    metrics = Metrics((X_train, y_train), (X_val, y_val))
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
    epochs = epochs
    batch_size = 150
    optimizer = 'adam'
    loss = 'mean_squared_error'
    model.compile(optimizer=optimizer, loss=loss)
    t0 = time.time()
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=metrics.get_val_data(),
                        callbacks=[metrics, es],
                        verbose=1).history
    t1 = time.time()
    print(f'Runtime: {t1 - t0}')
    runtime = t1 - t0
    print("--------------------------Evaluation--------------------------")
    test_evaluation = model.evaluate(X_test, y_test, batch_size=batch_size)
    return history, test_evaluation, runtime, model, metrics


warnings.filterwarnings(action="once")
plt.style.use('seaborn-white')
print("Using TensorFlow %s" % tf.__version__)
print("Using Gensim %s" % gensim.__version__)
print("Using Keras %s" % keras.__version__)
print("Using Sklearn %s" % sklearn.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices[0])
