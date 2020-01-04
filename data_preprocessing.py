from typing import List
import pychord
import pandas as pd
import ast
import utilities as utils


def augmentation(out_file_name, in_file_name):
    df = pd.read_csv("data/" + in_file_name)
    data = [ast.literal_eval(chords_string) for chords_string in df["chords"]]
    augmented_data = []
    for song in data:
        for interval in range(12):
            transposed_song = []
            for c in song:
                chord = pychord.Chord(c)
                chord.transpose(interval)
                transposed_song.append(str(chord))
            augmented_data.append(str(transposed_song))
    df = pd.DataFrame(augmented_data, columns=["chords"])
    df.to_csv("data/" + out_file_name, index=False)


def chords_string_rep(df, f_name):
    df['chords'].to_csv("data/" + f_name, header="chords", index=False)


def components(out_file_name, in_file_name, ignore_bass):
    df = pd.read_csv("data/" + in_file_name)
    data = [ast.literal_eval(chords_string) for chords_string in df["chords"]]
    components = []
    for song in data:
        components.append(str(utils.get_components(song, ignore_bass=ignore_bass)))
    df = pd.DataFrame(components, columns=["components"])
    df.to_csv("data/" + out_file_name, index=False)


def multihots(out_file_name, in_file_name, size):
    data = pd.read_csv("data/" + in_file_name).values[:, 0]
    songs = [ast.literal_eval(chords_string) for chords_string in data]
    multihots = []
    for song in songs:
        mh = str(utils.multihot(song, size=size))
        multihots.append(mh)
    df = pd.DataFrame(multihots, columns=["multihots"])
    df.to_csv("data/" + out_file_name, index=False)


if __name__ == "__main__":
    df = pd.read_csv("data/songs_and_chords.csv")
    chords_string_rep(df, f_name="chords_string_rep_no_bass.csv")
    augmentation(out_file_name="chords_string_rep_no_bass_aug_12.csv", in_file_name="chords_string_rep_no_bass.csv")
    # components(out_file_name = "components_no_bass.csv", in_file_name = "chords_string_rep_aug_12.csv", ignore_bass=True)
    # components(out_file_name = "components_with_bass.csv", in_file_name = "chords_string_rep_aug_12.csv", ignore_bass=False)
    # multihots(out_file_name = "multihot_no_bass_full.csv", in_file_name = "components_no_bass.csv", size='full')
    # multihots(out_file_name = "multihot_with_bass_full.csv", in_file_name = "components_with_bass.csv", size='full')
    # multihots(out_file_name = "multihot_no_bass_12.csv", in_file_name = "components_no_bass.csv", size=12)
    # multihots(out_file_name = "multihot_with_bass_12.csv", in_file_name = "components_with_bass.csv", size=12)
