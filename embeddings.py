import utilities as utils
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText

if __name__ == "__main__":
    sentences = utils.build_sentences()
    w2v = Word2Vec(sentences=sentences, min_count=1, size=70, window=4, sg=1)
    w2v.save("embeddings/word2vec.model")
    # w2v = Word2Vec.load("embeddings/word2vec.model")
    fastText = FastText(sentences=sentences, min_count=1, size=70, window=4)
    fastText.save("embeddings/fastText.model")
    # fastText = FastText.load("./embeddings/fastText.model")
