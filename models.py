import warnings

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm_notebook as tqdm

import utilities as my_utils

warnings.filterwarnings(action="once")


def prep_data(songs, sample_len=4):
    # Remove too short songs
    songs = [chords for chords in songs if len(chords) > sample_len]
    x = []
    y = []
    for chords in songs:
        for i in range(len(chords)):
            if i < len(chords) - sample_len:
                x.append(chords[i: i + sample_len - 1])
            else:
                x.append(
                    chords[i: len(chords) - 1]
                    + chords[: (i + sample_len) % len(chords)]
                )
            y.append(chords[(i + sample_len) % len(chords)])
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)
    return x, y


def plot_loss(history, metric_name):
    val_loss = history["val_" + metric_name]
    plt.plot(val_loss, linewidth=3, label="valid")
    loss = history[metric_name]
    plt.plot(loss, linewidth=3, label="train")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.show()


def save_model(model, model_name):
    model._name = model_name
    model_json = model.to_json()
    with open("models/" + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/" + model_name + ".h5")
    print("Saved model to disk")


def load_model(model_name):
    json_file = open("models/" + model_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/" + model_name + ".h5")
    print("Loaded model from disk")
    return loaded_model


def encode_chords(X, y, model):
    X_embedded = []
    y_embedded = []
    for X_sample, y_sample in tqdm(zip(X, y), total=X.shape[0], desc=str(model).split("(")[0] + ' Encoding'):
        X_embedded.append(model.wv[X_sample])
        y_embedded.append(model.wv[y_sample])
    X_embedded = np.array(X_embedded)
    y_embedded = np.array(y_embedded)
    return X_embedded, y_embedded


def print_example_predictions(model, X_test, y_test, embedding):
    for i in range(20):
        x_test_ex = X_test[i]
        y_pred = model.predict(x_test_ex[np.newaxis, :])
        # print(embedding.wv.most_similar(y_test[i].reshape(1,-1))[0][0] in [d[0] for d in embedding.wv.most_similar(
        # y_pred)][:10], end="\t")
        print(embedding.wv.most_similar(x_test_ex[0].reshape(1, -1))[0][0], end="\t")
        print(embedding.wv.most_similar(x_test_ex[1].reshape(1, -1))[0][0], end="\t")
        print(embedding.wv.most_similar(x_test_ex[2].reshape(1, -1))[0][0], end="\t")
        print("(", embedding.wv.most_similar(y_test[i].reshape(1, -1))[0][0], end=" )\t")
        print([d[0] for d in embedding.wv.most_similar(y_pred)][:5])


def generate_song(init_seq, length=16):
    sample = ft.wv[init_seq]
    song = sample
    for i in range(length - sample.shape[0]):
        next_chord = model.predict(sample[np.newaxis, :])
        sample = np.vstack([sample[1:], next_chord])
        song = np.vstack([song, next_chord])
    song = [ft.wv.most_similar(chord.reshape(1, -1))[0][0] for chord in song]
    print("Sample Song")
    for i, c in enumerate(song):
        print(c, end="\t|\t")
        if i % 4 == 3:
            print("")


class Metrics(Callback):
    def __init__(self, tr_data, val_data):
        self.validation_data = val_data
        self.train_data = tr_data

    def cosine_distance_sum(self, A, B):
        return sum([1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) for a, b in zip(A, B)])

    def mean_cos_dist_sum(self, X_vali, y_vali):
        y_pred = model.predict(X_vali)
        return self.cosine_distance_sum(y_pred, y_vali) / y_vali.shape[0]

    def on_train_begin(self, logs={}):
        self.mean_cos_dist = []
        self.val_mean_cos_dist = []

    def on_epoch_end(self, epoch, logs={}):
        score = self.mean_cos_dist_sum(self.train_data[0][:1000], self.train_data[1][:1000])
        score_val = self.mean_cos_dist_sum(self.validation_data[0][:1000], self.validation_data[1][:1000])
        self.mean_cos_dist.append(score)
        self.val_mean_cos_dist.append(score_val)
        print('epoch {},\tloss {:3.4f},\tval_loss {:3.4f}\tmean_cos_dist {:3.4f}\tval_mean_cos_dist {:3.4f}.'.format(
            epoch, logs['loss'], logs['val_loss'], score, score_val))

    def on_train_end(self, logs={}):
        self.mean_cos_dist = {'mean_cos_dist': self.mean_cos_dist}
        self.val_mean_cos_dist = {'val_mean_cos_dist': self.val_mean_cos_dist}

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.validation_data


if __name__ == "__main__":
    ft = FastText.load("./embeddings/fastText.model")
    songs = my_utils.build_sentences()
    X, y = prep_data(songs, sample_len=4)
    X_embedded, y_embedded = encode_chords(X, y, ft)

    X_train, X_test, y_train, y_test = train_test_split(X_embedded, y_embedded, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    # model = Sequential(
    #     [
    #         LSTM(128, activation='relu', return_sequences=True, input_shape=X_train.shape[1:]),
    #         LSTM(128, activation='relu'),
    #         Dense(y_train.shape[1])
    #     ]
    # )

    model = load_model("LSTM_30_epoch")

    validation_split = 0.25
    epochs = 30
    batch_size = 5000
    loss = Huber()
    optimizer = Adam(lr=1e-3)
    metrics = Metrics((X_train, y_train),
                      (X_val, y_val))

    METRICS_NAMES = [
        "loss",
        "mean_cos_dist",
    ]
    model.compile(loss=loss, optimizer=optimizer)
    print(model.summary())
    history = model.fit(metrics.get_train_data()[0],
                        metrics.get_train_data()[1],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=metrics.get_val_data(),
                        callbacks=[metrics],
                        verbose=1).history

    history.update(metrics.mean_cos_dist)
    history.update(metrics.val_mean_cos_dist)
    for metric_name in METRICS_NAMES:
        plot_loss(history, metric_name)
    from tensorflow.python.keras.utils.vis_utils import plot_model

    plot_model(
        model,
        to_file="images/model/LTSTM_30_plot.png",
        show_shapes=True,
        show_layer_names=False,
    )

    print_example_predictions(model, X_test, y_test, ft)
    generate_song(['F^7', 'E-7', 'A7'], length=16)
    # save_model(model, "LSTM_30_epoch")
