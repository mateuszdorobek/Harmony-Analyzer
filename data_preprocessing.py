from typing import List
import pychord
import pandas as pd
import ast
import utilities as utils


def chords_string_rep(df, f_name):
    df['chords'].to_csv("data/"+f_name, header="chords", index=False)


def components(df, f_name, ignore_bass):
    df = pd.read_csv("data/chords_string_rep.csv")
    data = [ast.literal_eval(chords_string) for chords_string in df["chords"]]
    components = []
    for song in data:
        components.append(str(utils.get_components(song, ignore_bass=ignore_bass)))
    df = pd.DataFrame(components, columns=["components"])
    df.to_csv("data/"+f_name, index=False)

def multihots(df, f_name, components, size):
    data = pd.read_csv("data/"+components).values[:, 0]
    songs = [ast.literal_eval(chords_string) for chords_string in data]
    multihots = []
    for song in songs:
        mh = str(utils.multihot(song, size=size))
        multihots.append(mh)
    df = pd.DataFrame(multihots, columns=["multihots"])
    df.to_csv("data/"+f_name, index=False)


if __name__ == "__main__":
    df = pd.read_csv("data/songs_and_chords.csv")
    # chords_string_rep(df, f_name="chords_string_rep.csv")
    # components(df, f_name = "components_no_bass.csv", ignore_bass=True)
    # components(df, f_name = "components_with_bass.csv", ignore_bass=False)
    # multihots(df, f_name = "multihot_no_bass_full.csv", components="components_no_bass.csv", size='full')
    # multihots(df, f_name = "multihot_with_bass_full.csv", components="components_with_bass.csv", size='full')
    # multihots(df, f_name = "multihot_no_bass_12.csv", components="components_no_bass.csv", size=12)
    # multihots(df, f_name = "multihot_with_bass_12.csv", components="components_with_bass.csv", size=12)

