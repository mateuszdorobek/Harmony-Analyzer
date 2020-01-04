import pandas as pd
import numpy as np
import ast
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("data/songs_and_chords.csv")
    data = [ast.literal_eval(chords_string) for chords_string in df["chords"]]

    lens = [len(chords) for chords in data]
    labels = np.arange(min(lens), max(lens) + 1)
    values = np.zeros(labels.shape)
    unique, counts = np.unique(lens, return_counts=True)
    values[unique] = counts
    width = 1
    plt.bar(labels, values, width)
    plt.xticks(labels + width * 0.5, labels)
    ax = plt.gca()
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 8 != 0:
            label.set_visible(False)
    ax.set_title("Chords Progression Lengths Distribution")
    ax.set_xlabel("Chords Progression Length")
    ax.set_ylabel("Number of Songs")
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.show()