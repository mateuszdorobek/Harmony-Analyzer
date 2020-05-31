from generator import *

pd.unique(SONG_TITLES['style'])
SONG_TITLES[SONG_TITLES['title'] == 'Now\'s The Time']
songs_aug = build_sentences(aug=True)
songs = build_sentences(aug=False)
# songs = remove_repetitions(songs)
embedding = Word2Vec.load("embeddings/word2vecSG14.model")
X_train, X_test, y_train, y_test = get_data(songs, encoding_type="Word2VecSG", seq_len=15)
model = load_model('Word2VecSGGRU3264')
model.compile(loss='mse', optimizer='adam')


def get_chord_embedding_interactive(x, embedding, range_choose):
    chords = [chord for chord, _ in embedding.wv.most_similar(x.reshape(1, -1), topn=range_choose)]
    print(*chords, sep="\t")
    illegal_choose = True
    while illegal_choose:
        selection = int(input(">>> "))
        if selection in range(-1, range_choose):
            illegal_choose = False
    print("----------------------------------------------------------")
    return embedding.wv[chords[selection]], selection


def generate_song_interactive(init_seq, length, embedding, model, index=None):
    i = index or np.random.randint(len(songs))

    init_seq = init_seq[i]
    input_seq_len = 16
    init_seq = init_seq[:input_seq_len]
    sample = embedding.wv[init_seq]
    song = init_seq
    model.predict(sample[-input_seq_len:][np.newaxis, :][:15], verbose=0)
    get_song_meta(i)
    pretty_print_song(song, input_seq_len)
    print("-----------------------Choose Chord-----------------------")
    range_choose = 8
    print(*range(range_choose), sep="\t")
    for i in range(length - sample.shape[0]):
        next_chord_prediction = model.predict(sample[-input_seq_len:][np.newaxis, :][:15], verbose=0)
        next_chord_prediction, sel = get_chord_embedding_interactive(next_chord_prediction, embedding, range_choose)
        if sel == -1:
            break
        sample = np.vstack([sample, next_chord_prediction])

    generated_song = [get_chord_name(chord, embedding, random=False) for chord in sample[input_seq_len:]]
    song = song + generated_song
    get_song_meta(index)
    pretty_print_song(song, input_seq_len)
    print()
    print(*song, sep=",")


warnings.filterwarnings(action="ignore")
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
generate_song_interactive(songs, 32, embedding, model, 659)
