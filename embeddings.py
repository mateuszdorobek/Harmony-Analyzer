import utilities as utils
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText


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
    vocab = utils.build_vocab()
    # w2v = word2vec(vocab, file_name="word2vec.model")
    # w2v = Word2Vec.load("embeddings/word2vec.model")
    # fastText = fast_text(data, "fastText.model")
    # fastText = FastText.load("./embeddings/fastText.model")
