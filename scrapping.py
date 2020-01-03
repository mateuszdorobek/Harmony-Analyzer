import utilities as util
import pandas as pd
import os


def scrap_data():
    web_links = list(pd.read_csv("data/web_links.csv", header=None).values[:, 0])
    songs_urls = util.get_song_urls(web_links)
    df = util.extract_meta_data(songs_urls)
    os.system('cls')
    print("\nLoaded %d unique songs." % df.shape[0])
    answer = None
    while answer not in ("y", "n"):
        answer = input("Overwrite data in data/songs_and_chords? [y/n] ").lower()
        if answer == "y":
            df.to_csv("data/songs_and_chords.csv", index=False, header=True)
        elif answer == "n":
            break
        else:
            print("Write y or n.")


if __name__ == "__main__":
    scrap_data()
