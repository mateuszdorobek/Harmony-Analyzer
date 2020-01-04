import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import ast
import matplotlib.pyplot as plt
# from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import random


# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-download-auto-examples-tutorials-run-word2vec-py


def word2vec(file_name):
    df = pd.read_csv("data/chords_string_rep_no_bass_aug_12.csv")
    data = [ast.literal_eval(chords_string) for chords_string in df["chords"]]
    model = Word2Vec(data, min_count=1, size=70, workers=3, window=4, sg=1)
    model.save("embeddings/" + file_name)
    return model


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = []  # positions in vector space
    labels = []  # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_matplotlib(x_vals, y_vals, labels, selected_indices, title):
    plt.figure(figsize=(10, 7))
    plt.scatter(x_vals, y_vals, alpha=0.7)
    # plt.ylim((22, 31))
    # plt.xlim((-7, 9))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # selected_indices = random.sample(range(len(labels)), 10)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]), size=16)
        plt.scatter(x_vals[i], y_vals[i], c='g')
        plt.plot(x_vals[i], y_vals[i])
    plt.show()


def plot_test_cases(model):
    dominants = ("Jazz Chords Embedding - Circle of Fifths",
                 ['C7', 'Db7', 'D7', 'Eb7', 'E7', 'F7', 'Gb7', 'G7', 'Ab7', 'A7', 'Bb7', 'B7'])
    majors = ("Major Chords", ['C^7', 'Db^7', 'D^7', 'Eb^7', 'E^7', 'F^7', 'Gb^7', 'G^7', 'Ab^7', 'A^7', 'Bb^7', 'B^7'])
    minors = ("Minor Chords", ['C-7', 'Db-7', 'D-7', 'Eb-7', 'E-7', 'F-7', 'Gb-7', 'G-7', 'Ab-7', 'A-7', 'Bb-7', 'B-7'])
    dominants_and_majors = ("Dominants and Majors",
                            ['C7', 'Db7', 'D7', 'Eb7', 'E7', 'F7', 'Gb7', 'G7', 'Ab7', 'A7', 'Bb7', 'B7', 'C^7', 'Db^7',
                             'D^7', 'Eb^7', 'E^7', 'F^7', 'Gb^7', 'G^7', 'Ab^7', 'A^7', 'Bb^7', 'B^7'])
    half_dim = ("Half-diminished Chords", ['Ch7', 'Dbh7', 'Dh7', 'Ebh7', 'Eh7', 'Fh7', 'Gbh7', 'Gh7', 'Abh7', 'Ah7', 'Bbh7', 'Bh7'])
    pregressions = ("II-V-I Chords", ['D-7', 'G7', 'C^7', 'E-7', 'A7', 'D^7', 'Gb-7', 'B7', 'E^7'])
    x_vals, y_vals, labels = reduce_dimensions(model)
    for test_case in [dominants, majors, minors, pregressions, dominants_and_majors, half_dim]:
        title, chords = test_case
        indices = []
        for chord in chords:
            indices.append(np.where(labels == chord)[0][0])
        plot_with_matplotlib(x_vals, y_vals, labels, indices, title)


if __name__ == "__main__":
    # w2v = word2vec(file_name = "word2vec.model")
    w2v = Word2Vec.load("embeddings/word2vec.model")
    print(w2v.wv.most_similar('E7')[:5])
    plot_test_cases(w2v)
