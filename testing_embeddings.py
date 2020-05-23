import ast
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from sklearn.manifold import TSNE

import utilities as my_utils
from multihotembedding import MultihotEmbedding
from qualities import QUALITY_DICT, KEYS


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
    plt.savefig("images/embeddings/" + title)
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
    test_dict = {}
    chord_types = list(QUALITY_DICT.keys())

    print('root_change test generation.')

    for chord in chord_types:
        for note in range(len(KEYS)):
            for interval in range(1, len(KEYS)):
                for diff in range(1, len(KEYS)):
                    row = KEYS[note] + chord + " " + KEYS[(note + interval) % 12] + chord + " " + KEYS[
                        (diff + note) % 12] + chord + " " + KEYS[(diff + note + interval) % 12] + chord
                    test_list.append(row)
    test_dict[": root_change"] = test_list
    test_list = []

    print('chord_type_change test generation.')
    for chord_1 in chord_types:
        for chord_2 in chord_types:
            if chord_1 == chord_2:
                continue
            for note in range(len(KEYS)):
                for diff in range(1, len(KEYS)):
                    row = KEYS[note] + chord_1 + " " + KEYS[note] + chord_2 + " " + KEYS[
                        (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note) % 12] + chord_2
                    test_list.append(row)
    test_dict[": chord_type_change"] = test_list
    test_list = []

    print('V-I_progression test generation.')
    chord_1 = '7'
    chord_2 = '^7'
    interval = 5
    for note in range(len(KEYS)):
        for diff in range(1, len(KEYS)):
            row = KEYS[note] + chord_1 + " " + KEYS[(note + interval) % 12] + chord_2 + " " + KEYS[
                (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note + interval) % 12] + chord_2
            test_list.append(row)
    test_dict[": V-I_progression"] = test_list
    test_list = []

    print('II-V_progression test generation.')
    chord_1 = '-7'
    chord_2 = '7'
    interval = 5
    for note in range(len(KEYS)):
        for diff in range(1, len(KEYS)):
            row = KEYS[note] + chord_1 + " " + KEYS[(note + interval) % 12] + chord_2 + " " + KEYS[
                (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note + interval) % 12] + chord_2
            test_list.append(row)
    test_dict[": II-V_progression"] = test_list
    test_list = []

    print('V/V-V_progression test generation.')
    chord = '7'
    interval = 5
    for note in range(len(KEYS)):
        for diff in range(1, len(KEYS)):
            row = KEYS[note] + chord + " " + KEYS[(note + interval) % 12] + chord + " " + KEYS[
                (diff + note) % 12] + chord + " " + KEYS[(diff + note + interval) % 12] + chord
            test_list.append(row)
    test_dict[": V/V-V_progression"] = test_list
    test_list = []

    print('less_common_progression test generation.')
    chord_types = chord_types[:10] + chord_types[13:]
    for chord_1 in chord_types:
        for chord_2 in chord_types:
            if chord_1 == chord_2:
                continue
            for note in range(len(KEYS)):
                for interval in range(1, len(KEYS)):
                    for diff in range(1, len(KEYS)):
                        row = KEYS[note] + chord_1 + " " + KEYS[(note + interval) % 12] + chord_2 + " " + KEYS[
                            (diff + note) % 12] + chord_1 + " " + KEYS[(diff + note + interval) % 12] + chord_2
                        test_list.append(row)
    test_dict[": less_common_progression"] = test_list

    min_test_size = min([len(v) for v in test_dict.values()])
    for key in test_dict:
        test_dict[key] = list(np.random.choice(test_dict[key], size=min_test_size, replace=False))

    test_list = []
    for k, v in zip(test_dict.keys(), test_dict.values()):
        test_list.append(k)
        test_list = test_list + v
    print("Saving test file: " + file_name)
    pd.DataFrame(test_list).to_csv("data/validation/" + file_name, header=None, index=False)


def get_model_name(model, stripped=False):
    model_name = str(model).split("(")[0]
    if model_name == "Word2Vec":
        if model.sg:
            model_name += " Skip-Gram"
        else:
            model_name += " CBOW"

    embedding_size = str(model).split("=")[2].split(",")[0]
    model_name += " Size: " + embedding_size
    
    try:
        window = str(model.window)
    except:
        window = ""
    model_name += " Window: " + window
    
    model_name = model_name.replace(")", "")
    if stripped:
        model_name = model_name.replace(":", "").replace(" ", "").replace("-", "")
    return model_name


def print_accuracy(model, word_analogies_file):
    score = round(model.wv.evaluate_word_analogies(word_analogies_file)[0], 4)
    model_name = get_model_name(model)
    print(model_name, score)


if __name__ == "__main__":
    # generate_validation_file(file_name="test_chords_double_pairs.txt")
    sentences = my_utils.build_sentences()
    ft = FastText.load("embeddings/fastText.model")
    w2vCBOW = Word2Vec.load("embeddings/word2vecCBOW.model")
    w2vSG = Word2Vec.load("embeddings/word2vecSG.model")
    # mh = MultihotEmbedding(sentences=sentences)
    mh = MultihotEmbedding.load("embeddings/multihotembedding.model")
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    for m in [w2vCBOW, w2vSG, ft, mh]:
        print_accuracy(m, "data/validation/test_chords_double_pairs.txt")
        # plot_test_cases(m)
        # print(m.wv.most_similar('C7')[:5])
        # break
