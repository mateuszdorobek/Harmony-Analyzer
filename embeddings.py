import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import ast
import matplotlib.pyplot as plt
# from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

#https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-download-auto-examples-tutorials-run-word2vec-py


def word2vec(file_name):
    df = pd.read_csv("data/chords_string_rep_no_bass_aug_12.csv")
    data = [ast.literal_eval(chords_string) for chords_string in df["chords"]]
    model = Word2Vec(data, min_count=1, size=70, workers=3, window=4, sg=1)
    model.save("embeddings/"+file_name)
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


def plot_with_matplotlib(x_vals, y_vals, labels, selected_indices):

    plt.figure(figsize=(22, 22))
    plt.scatter(x_vals, y_vals, alpha=0.2)

    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]), size=14)
    plt.show()

def plot_test_cases(model):

    dominants = ['C7', 'Db7', 'D7', 'Eb7', 'E7', 'F7', 'Gb7', 'G7', 'Ab7', 'A7', 'Bb7', 'B7']
    # majors = ['C^7', 'Db^7', 'D^7', 'Eb^7', 'E^7', 'F^7', 'Gb^7', 'G^7', 'Ab^7', 'A^7', 'Bb^7', 'B^7']
    # minors = ['C-7', 'Db-7', 'D-7', 'Eb-7', 'E-7', 'F-7', 'Gb-7', 'G-7', 'Ab-7', 'A-7', 'Bb-7', 'B-7']
    # pregressions = ['D-7', 'G7', 'C^7', 'E-7', 'A7', 'D^7', 'Gb-7', 'B7', 'E^7']
    x_vals, y_vals, labels = reduce_dimensions(model)
    # , majors, minors, pregressions
    for test_case in [dominants]:
        indices = []
        for chord in test_case:
            indices.append(np.where(labels == chord)[0][0])
        plot_with_matplotlib(x_vals, y_vals, labels, indices)

if __name__ == "__main__":
    # w2v = word2vec(file_name = "word2vec.model")
    model = Word2Vec.load("embeddings/word2vec.model")
    print(w2v.wv.most_similar('E7')[:5])
    plot_test_cases(w2v)

