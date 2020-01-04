import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
# from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE


# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-download-auto-examples-tutorials-run-word2vec-py
def load_data():
    df = pd.read_csv("data/chords_string_rep_no_bass_aug_12.csv")
    return [ast.literal_eval(chords_string) for chords_string in df["chords"]]


def tSNE_reduction(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)
    vectors = model.wv[model.wv.vocab]  # positions in vector space
    labels = np.asarray(list(model.wv.vocab))  # keep track of words to label our data again later
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x = vectors[:, 0]
    y = vectors[:, 1]
    return x, y, labels


def plot_reduction(x, y, labels, selected_indices, title):
    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, alpha=0.7)
    # plt.ylim((22, 31))
    # plt.xlim((-7, 9))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # selected_indices = random.sample(range(len(labels)), 10)
    for i in selected_indices:
        plt.annotate(labels[i], (x[i], y[i]), size=16)
        plt.scatter(x[i], y[i], c='g')
        plt.plot(x[i], y[i])
    plt.show()


def plot_test_cases(model):
    dominants = ("Jazz Chords Embedding - Circle of Fifths",
                 ['C7', 'Db7', 'D7', 'Eb7', 'E7', 'F7', 'Gb7', 'G7', 'Ab7', 'A7', 'Bb7', 'B7'])
    majors = ("Major Chords", ['C^7', 'Db^7', 'D^7', 'Eb^7', 'E^7', 'F^7', 'Gb^7', 'G^7', 'Ab^7', 'A^7', 'Bb^7', 'B^7'])
    minors = ("Minor Chords", ['C-7', 'Db-7', 'D-7', 'Eb-7', 'E-7', 'F-7', 'Gb-7', 'G-7', 'Ab-7', 'A-7', 'Bb-7', 'B-7'])
    dominants_and_majors = ("Dominants and Majors",
                            ['C7', 'Db7', 'D7', 'Eb7', 'E7', 'F7', 'Gb7', 'G7', 'Ab7', 'A7', 'Bb7', 'B7', 'C^7', 'Db^7',
                             'D^7', 'Eb^7', 'E^7', 'F^7', 'Gb^7', 'G^7', 'Ab^7', 'A^7', 'Bb^7', 'B^7'])
    half_dim = (
        "Half-diminished Chords",
        ['Ch7', 'Dbh7', 'Dh7', 'Ebh7', 'Eh7', 'Fh7', 'Gbh7', 'Gh7', 'Abh7', 'Ah7', 'Bbh7', 'Bh7'])
    progressions = ("II-V-I Chords", ['D-7', 'G7', 'C^7', 'E-7', 'A7', 'D^7', 'Gb-7', 'B7', 'E^7'])
    x, y, labels = tSNE_reduction(model)
    for test_case in [dominants, majors, minors, progressions, dominants_and_majors, half_dim]:
        title, chords = test_case
        indices = []
        for chord in chords:
            indices.append(np.where(labels == chord)[0][0])
        plot_reduction(x, y, labels, indices, title)


def word2vec(sentences, file_name, sg):
    # Skip-Gram Model
    model = Word2Vec(sentences=sentences, min_count=1, size=70, window=4, sg=sg)
    model.save("embeddings/" + file_name)
    return model


def fast_text(sentences, file_name):
    model = FastText(sentences=sentences, min_count=1, size=70, window=4)
    model.save("embeddings/" + file_name)
    return model


if __name__ == "__main__":
    data = load_data()
    w2v = word2vec(data, file_name="word2vec.model")
    # w2v = Word2Vec.load("embeddings/word2vec.model")
    print(w2v.wv.most_similar('E7')[:5])
    plot_test_cases(w2v)

    # fastText = fast_text(data, "fastText.model")
    # print(fastText.wv.most_similar('E7')[:5])
    # plot_test_cases(fastText)
