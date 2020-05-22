import utilities as my_utils
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText

from multihotembedding import MultihotEmbedding

if __name__ == "__main__":
    sentences = my_utils.build_sentences()
    w2v = Word2Vec(sentences=sentences, min_count=1, size=14, window=4, sg=1)
    w2v.save("embeddings/word2vec.model")
    # w2v = Word2Vec.load("embeddings/word2vec.model")
    ft = FastText(sentences=sentences, min_count=1, size=70, window=4)
    ft.save("embeddings/fastText.model")
    # ft = FastText.load("./embeddings/fastText.model")
    mh = MultihotEmbedding(sentences=sentences)
    mh.save("embeddings/multihotembedding.model")
    # mh = MultihotEmbedding.load("embeddings/multihotembedding.model")
