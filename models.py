import warnings

import numpy as np
from gensim.models import FastText
from music21.chord import Chord
from music21.harmony import chordSymbolFigureFromChord as figureChord
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.layers import Dense, LSTM

import utilities as my_utils

warnings.filterwarnings(action="once")


def prep_data(songs, sample_len = 4):
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


# def decode(chord):
#     if not all(isinstance(x, np.int32) for x in chord):
#         raise Exception(
#             "Expected chord to be array([0, 1, 0, 0 , 1, ...]) instead got: ", chord
#         )
#     decoded_chord = []
#     for counter, value in enumerate(chord):
#         if value == 1:
#             decoded_chord.append(counter)
#     return decoded_chord


# def print_chords(encoded_chords):
#     chords_symbols = [figureChord(Chord(decode(chord))) for chord in encoded_chords]
#     print(*chords_symbols, sep="  | ")


# def threshold_prediction(pred, notes_num):
#     thresholded_pred = np.zeros(pred.shape, dtype=np.int32)
#     selected_notes = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[
#                      :notes_num
#                      ]
#     thresholded_pred[selected_notes] = 1
#     return thresholded_pred


def save_model(model):
    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/model.h5")
    print("Saved model to disk")


def load_model(json_path, h5_path):
    json_file = open(json_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_path)
    print("Loaded model from disk")
    return loaded_model


def encode_chords(X, y, model):
    X_embedded = []
    y_embedded = []
    for X_sapmle, y_sapmle in zip(X, y):
        X_embedded.append(model.wv[X_sapmle])
        y_embedded.append(model.wv[y_sapmle])
    X_embedded = np.array(X_embedded)
    y_embedded = np.array(y_embedded)
    return X_embedded, y_embedded


if __name__ == "__main__":
    songs = my_utils.build_sentences()
    X, y = prep_data(songs, sample_len = 4)
    ft = FastText.load("./embeddings/fastText.model")
    X, y = encode_chords(X, y, ft)

    # model = keras.Sequential(
    #     [
    #         LSTM(32, input_shape=(7, 12)),
    #         Dense(12, activation='sigmoid')
    #     ]
    # )
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=3, batch_size=8)
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print(scores)
    # print(model.summar())

    # save_model(model)
    # loaded_model = load_model('models/model.json', "models/model.h5")
    #
    # loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(loaded_model.summary())
    # scores = loaded_model.evaluate(X_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1] * 100))
    # for i in range(20):
    #     predict_chords(loaded_model, X_test[i])
