from gensim.models import Word2Vec
from gensim.models.fasttext import FastText

import utilities as my_utils
from multihotembedding import MultihotEmbedding

if __name__ == "__main__":
    sentences = my_utils.build_sentences()
    w2vCBOW = Word2Vec(sentences=sentences, min_count=1, size=13, window=2, sg=0)
    w2vCBOW.save("embeddings/word2vecCBOW.model")

    w2vSG = Word2Vec(sentences=sentences, min_count=1, size=14, window=2, sg=1)
    w2vSG.save("embeddings/word2vecSG.model")

    ft = FastText(sentences=sentences, min_count=1, size=13, window=2)
    ft.save("embeddings/fastText.model")

    mh = MultihotEmbedding(sentences=sentences)
    mh.save("embeddings/multihotembedding.model")
