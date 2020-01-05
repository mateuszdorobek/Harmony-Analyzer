import ast

import matplotlib.axes
from matplotlib.offsetbox import AnchoredText
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText

from qualities import QUALITY_DICT, KEYS
from sklearn.manifold import TSNE
import pandas as pd
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text


# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-download-auto-examples-tutorials-run-word2vec-py
def tSNE_reduction(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)
    vectors = model.wv[model.wv.vocab]  # positions in vector space
    labels = np.asarray(list(model.wv.vocab))  # keep track of words to label our data again later
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x = vectors[:, 0]
    y = vectors[:, 1]
    return x, y, labels


def plot_reduction(x, y, labels, selected_indices, title, layout):
    plt.figure(figsize=(10, 7))
    size = 12
    weight = 'normal'
    if layout == 'big':
        size = 16
        weight = 'bold'
    plt.scatter(x, y, alpha=0.7, marker=".")
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontweight="bold", size=20)
    texts = []
    for i in selected_indices:
        texts.append(plt.annotate(labels[i], (x[i], y[i]), size=size, weight=weight))
        plt.scatter(x[i], y[i], c='orange', marker="o", edgecolors='k', s=40)
        plt.plot(x[i], y[i])
    adjust_text(texts, x[selected_indices], y[selected_indices], arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    adjust_text([plt.text(max(x), min(y), "Mateusz Dorobek 2020", alpha=0.5, color='gray', fontname="Helvetica",
                          size='x-large')])
    plt.savefig("images/embeddings/"+title)
    plt.show()



def plot_test_cases(model):
    df = pd.read_csv("data/validation/test_chords_groups_for_embedding.txt", header=None, sep=";")
    chords_groups = [ast.literal_eval(chords_string) for chords_string in df.values[:, 1]]
    titles = list(df.values[:, 0])
    x, y, labels = tSNE_reduction(model)
    for title, chords in zip(titles, chords_groups):
        indices = []
        for chord in chords:
            indices.append(np.where(labels == chord)[0][0])
        if "Circle of Fifths" in title:
            plot_reduction(x, y, labels, indices, title + " - " + str(model).split("(")[0], layout='normal')
        else:
            plot_reduction(x, y, labels, indices, title + " - " + str(model).split("(")[0], layout='big')
        # break

def generate_validation_file(file_name):
    test_list = []
    chord_types = list(QUALITY_DICT.keys())

    print('root_change test generation.')
    test_list.append(': root_change')
    for chord in chord_types:
        for note in range(len(KEYS)):
            for interval in range(1, len(KEYS)):
                for diff in range(1, len(KEYS)):
                    row = KEYS[note] + chord + " " + KEYS[(note + interval) % 12] + chord + " " + KEYS[
                        (diff + note) % 12] + chord + " " + KEYS[(diff + note + interval) % 12] + chord
                    test_list.append(row)

    print('chord_type_change test generation.')
    test_list.append(': chord_type_change')
    for chord_1 in chord_types:
        for chord_2 in chord_types:
            if chord_1 == chord_2:
                continue
            for note in range(len(KEYS)):
                for diff in range(1, len(KEYS)):
                    row = KEYS[note] + chord_1 + " " + KEYS[note] + chord_2 + " " + KEYS[
                        (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note) % 12] + chord_2
                    test_list.append(row)

    print('V-I_progression test generation.')
    test_list.append(': V-I_progression')
    chord_1 = '7'
    chord_2 = '^7'
    interval = 5
    for note in range(len(KEYS)):
        for diff in range(1, len(KEYS)):
            row = KEYS[note] + chord_1 + " " + KEYS[(note + interval) % 12] + chord_2 + " " + KEYS[
                (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note + interval) % 12] + chord_2
            test_list.append(row)

    print('II-V_progression test generation.')
    test_list.append(': II-V_progression')
    chord_1 = '-7'
    chord_2 = '7'
    interval = 5
    for note in range(len(KEYS)):
        for diff in range(1, len(KEYS)):
            row = KEYS[note] + chord_1 + " " + KEYS[(note + interval) % 12] + chord_2 + " " + KEYS[
                (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note + interval) % 12] + chord_2
            test_list.append(row)

    print('V/V-V_progression test generation.')
    test_list.append(': V/V-V_progression')
    chord = '7'
    interval = 5
    for note in range(len(KEYS)):
        for diff in range(1, len(KEYS)):
            row = KEYS[note] + chord + " " + KEYS[(note + interval) % 12] + chord + " " + KEYS[
                (diff + note) % 12] + chord + " " + KEYS[(diff + note + interval) % 12] + chord
            test_list.append(row)

    print('less_common_progression test generation.')
    test_list.append(': less_common_progression')
    chord_types = chord_types[:10] + chord_types[13:]
    for chord_1 in chord_types:
        for chord_2 in chord_types:
            if chord_1 == chord_2:
                continue
            for note in range(len(KEYS)):
                for interval in range(1, len(KEYS)):
                    for diff in range(1, len(KEYS)):
                        if random.random() < 0.1:
                            row = KEYS[note] + chord_1 + " " + KEYS[(note + interval) % 12] + chord_2 + " " + KEYS[
                                (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note + interval) % 12] + chord_2
                            test_list.append(row)
    print("Saving test file: " + file_name)
    pd.DataFrame(test_list).to_csv("data/validation/" + file_name, header=None, index=False)


def print_accuracy(model):
    word_analogies_file = "data/validation/test_chords_double_pairs.txt"
    print(str(model).split("(")[0], "accuracy:")
    model.wv.evaluate_word_analogies(word_analogies_file)


if __name__ == "__main__":
    # generate_validation_file(file_name="test_chords_double_pairs.txt")
    w2v = Word2Vec.load("embeddings/word2vec.model")
    fastText = FastText.load("./embeddings/fastText.model")
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    for m in [w2v, fastText]:
        # print_accuracy(m)
        plot_test_cases(m)
        print(m.wv.most_similar('C7')[:5])
        # break