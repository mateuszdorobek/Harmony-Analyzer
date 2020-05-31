from model_testing import *

warnings.filterwarnings(action="ignore")
plt.style.use('seaborn-white')
SONG_TITLES = pd.read_csv("data/songs_and_chords.csv")


def get_index_dicts(sentences):
    unique_words = set(x for l in sentences for x in l)
    word2index = dict((c, i) for i, c in enumerate(unique_words))
    index2word = dict((i, c) for i, c in enumerate(unique_words))
    return word2index, index2word


def index_transform(sequences, word2index):
    sequences_indexed = []
    for seq in tqdm(sequences, total=len(sequences), desc='Indexing'):
        seq_indexed = []
        for word in seq:
            seq_indexed.append(word2index[word])
        sequences_indexed.append(seq_indexed)
    return sequences_indexed


def embedding_transform(sequences, model):
    sequences_embedded = []
    for seq in tqdm(sequences, total=len(sequences), desc=str(model).split("(")[0] + " Encoding"):
        seq_embedded = []
        for word in seq:
            seq_embedded.append(model.wv[word])
        sequences_embedded.append(np.array(seq_embedded))
    return sequences_embedded


def get_chord_name(x, embedding, random=True):
    if random:
        #         indexes = np.arange(5)
        #         probs = -np.exp(indexes)
        #         probs = np.flip(probs)/sum(probs)
        #         index = np.random.choice(np.arange(5), size=1, p=probs)[0]
        index = np.random.randint(5)
    else:
        index = 0
    return embedding.wv.most_similar(x.reshape(1, -1))[index][0]


def print_example_predictions(model, X_test, y_test, embedding):
    for i in range(20):
        x_test_ex = X_test[i]
        y_pred = model.predict(x_test_ex[np.newaxis, :])
        print(get_chord_name(x_test_ex[-3], embedding, random=False), end="\t")
        print(get_chord_name(x_test_ex[-2], embedding, random=False), end="\t")
        print(get_chord_name(x_test_ex[-1], embedding, random=False), end="\t")
        print("(", get_chord_name(y_test[i], embedding, random=False), end=")\t", sep="")
        most_sim = [d[0] for d in embedding.wv.most_similar(y_pred)][:5]
        print("[", end="")
        print(*most_sim, ']', sep="\t")


def remove_repetitions_in_song(song):
    return [chord for idx, chord in enumerate(song) if idx == 0 or song[idx - 1] != chord]


def remove_repetitions(songs):
    return [remove_repetitions_in_song(song) for song in songs]


def get_song_meta(index):
    meta = SONG_TITLES.iloc[index]
    print(f"\nSong No: {index}/{len(SONG_TITLES)}\nTitle: "
          f"{meta.title}\nComposer: {meta.composer}\n{meta.style} in {meta.key}")


def pretty_print_song(song, input_seq_len):
    print("\n", "-" * 29, "ORIGINAL SONG", '-' * 29)
    for i, c in enumerate(song):
        if i % 4 == 0:
            print("\n" + str(i), end=':\t\t')
        if i == input_seq_len - 1:
            print(c, end="\t||\t")
            print("\n\n", "-" * 29, "AI GENERATION", '-' * 29)
        else:
            print(c, end="\t|\t")
    print("")


def generate_song(init_seq, length, embedding, model, index=None):
    i = index or np.random.randint(len(songs))
    get_song_meta(i)
    init_seq = init_seq[i]
    #     input_seq_len = model._feed_input_shapes[0][1]
    input_seq_len = 16
    init_seq = init_seq[:input_seq_len]
    sample = embedding.wv[init_seq]
    song = init_seq
    for i in range(length - sample.shape[0]):
        next_chord = model.predict(sample[-input_seq_len:][np.newaxis, :], verbose=0)
        sample = np.vstack([sample, next_chord])
    generated_song = [get_chord_name(chord, embedding) for chord in sample[input_seq_len:]]
    song = song + generated_song
    pretty_print_song(song, input_seq_len)
    print(*song, sep=",")


if __name__ == "__main__":
    songs_aug = build_sentences(aug=True)
    songs = build_sentences(aug=False)
    songs = remove_repetitions(songs)
    embedding = Word2Vec.load("embeddings/word2vecSG14.model")
    X_train, X_test, y_train, y_test = get_data(songs, encoding_type="Word2VecSG", seq_len=15)
    model = load_model('Word2VecSGGRU3264')
    model.compile(loss='mse', optimizer='adam')
    # print_example_predictions(model, X_test, y_test, embedding)

    for index in range(1, 50):
        generate_song(songs, 32, embedding, model, index)
        print("\n\n\n\n")
    # generate_song_interactive()
