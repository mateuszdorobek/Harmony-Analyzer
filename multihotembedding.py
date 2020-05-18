import _pickle as pickle
import logging
from itertools import chain
from numbers import Integral

import numpy as np
from gensim import matutils
from numpy import dot, float32, array, sqrt, newaxis
from six import string_types

import utilities as my_utils

logger = logging.getLogger(__name__)


# Following class is based on Genism WordEmbeddingsKeyedVectors class created by Shiva Manne.
# I modified for proposes of this project

class MultihotEmbedding:
    class Wv:
        def __init__(self, sentences):
            s = set()
            for row in sentences:
                s.update(set(row))
            self.vocab = dict.fromkeys(s, 0)
            for key in self.vocab.keys():
                encoding = my_utils.multihot(my_utils.get_components([key]), size=33)[0]
                self.vocab[key] = encoding
            self.index2word = list(self.vocab.keys())
            self.vectors = np.array(list(self.vocab.values()))
            self.vectors_norm = self._l2_norm(self.vectors, replace=False)

        @staticmethod
        def _l2_norm(m, replace=False):
            """Return an L2-normalized version of a matrix.

            Parameters
            ----------
            m : np.array
                The matrix to normalize.
            replace : boolean, optional
                If True, modifies the existing matrix.

            Returns
            -------
            The normalized matrix.  If replace=True, this will be the same as m.

            """
            dist = sqrt((m ** 2).sum(-1))[..., newaxis]
            if replace:
                m /= dist
                return m
            else:
                return (m / dist).astype(float32)

        def init_sims(self, replace=False):
            """Precompute L2-normalized vectors.

            Parameters
            ----------
            replace : bool, optional
                If True - forget the original vectors and only keep the normalized ones = saves lots of memory!

            Warnings
            --------
            You **cannot continue training** after doing a replace.
            The model becomes effectively read-only: you can call
            :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar`,
            :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity`, etc., but not train.

            """
            if getattr(self, 'vectors_norm', None) is None or replace:
                logger.info("precomputing L2-norms of word weight vectors")

        def word_vec(self, word, use_norm=False):
            """Get `word` representations in vector space, as a 1D numpy array.

            Parameters
            ----------
            word : str
                Input word
            use_norm : bool, optional
                If True - resulting vector will be L2-normalized (unit euclidean length).

            Returns
            -------
            numpy.ndarray
                Vector representation of `word`.

            Raises
            ------
            KeyError
                If word not in vocabulary.

            """
            if word in self.vocab:
                if use_norm:
                    result = self.vectors_norm[list(self.vocab.keys()).index(word)]
                else:
                    result = self.vectors[list(self.vocab.keys()).index(word)]

                result.setflags(write=False)
                return result
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

        def most_similar(self, positive=None, negative=None, topn=5, restrict_vocab=None, indexer=None):
            """Find the top-N most similar words.
            Positive words contribute positively towards the similarity, negative words negatively.

            This method computes cosine similarity between a simple mean of the projection
            weight vectors of the given words and the vectors for each word in the model.
            The method corresponds to the `word-analogy` and `distance` scripts in the original
            word2vec implementation.

            Parameters
            ----------
            positive : list of str, optional
                List of words that contribute positively.
            negative : list of str, optional
                List of words that contribute negatively.
            topn : int or None, optional
                Number of top-N similar words to return, when `topn` is int. When `topn` is None,
                then similarities for all words are returned.
            restrict_vocab : int, optional
                Optional integer which limits the range of vectors which
                are searched for most-similar values. For example, restrict_vocab=10000 would
                only check the first 10000 word vectors in the vocabulary order. (This may be
                meaningful if you've sorted the vocabulary by descending frequency.)

            Returns
            -------
            list of (str, float) or numpy.array
                When `topn` is int, a sequence of (word, similarity) is returned.
                When `topn` is None, then similarities for all words are returned as a
                one-dimensional numpy array with the size of the vocabulary.

            """
            if isinstance(topn, Integral) and topn < 1:
                return []

            if positive is None:
                positive = []
            if negative is None:
                negative = []

            self.init_sims()

            if isinstance(positive, string_types) and not negative:
                # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
                positive = [positive]

            # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
            positive = [
                (word, 1.0) if isinstance(word, string_types + (np.ndarray,)) else word
                for word in positive
            ]
            negative = [
                (word, -1.0) if isinstance(word, string_types + (np.ndarray,)) else word
                for word in negative
            ]

            # compute the weighted average of all words
            all_words, mean = set(), []
            for word, weight in positive + negative:
                if isinstance(word, np.ndarray):
                    mean.append(weight * word)
                else:
                    mean.append(weight * self.word_vec(word, use_norm=True))
                    if word in self.vocab:
                        idx = list(self.vocab.keys()).index(word)
                        all_words.add(idx)
            if not mean:
                raise ValueError("cannot compute similarity with no input")

            mean = matutils.unitvec(array(mean).mean(axis=0)).astype(float32)

            if indexer is not None and isinstance(topn, int):
                return indexer.most_similar(mean, topn)

            limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]
            dists = dot(limited, mean)
            if not topn:
                return dists
            best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
            # ignore (don't return) words from the input
            result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
            return result[:topn]

        def similarity(self, a, b):
            if a not in self.vocab.keys() or b not in self.vocab.keys():
                return None
            return np.dot(self.vocab[a], self.vocab[b]) / (
                    np.linalg.norm(self.vocab[a]) * np.linalg.norm(self.vocab[b]))

        def __getitem__(self, item):
            return [self.vocab.get(key) for key in item]

        @staticmethod
        def _log_evaluate_word_analogies(section):
            """Calculate score by section, helper for
            :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.evaluate_word_analogies`.

            Parameters
            ----------
            section : dict of (str, (str, str, str, str))
                Section given from evaluation.

            Returns
            -------
            float
                Accuracy score.

            """
            correct, incorrect = len(section['correct']), len(section['incorrect'])
            if correct + incorrect > 0:
                score = correct / (correct + incorrect)
                logger.info("%s: %.1f%% (%i/%i)", section['section'], 100.0 * score, correct, correct + incorrect)
                return score

        def evaluate_word_analogies(self, analogies, restrict_vocab=300000, case_insensitive=False, dummy4unknown=False):
            """Compute performance of the model on an analogy test set.

            This is modern variant of :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.accuracy`, see
            `discussion on GitHub #1935 <https://github.com/RaRe-Technologies/gensim/pull/1935>`_.

            The accuracy is reported (printed to log and returned as a score) for each section separately,
            plus there's one aggregate summary at the end.

            This method corresponds to the `compute-accuracy` script of the original C word2vec.
            See also `Analogy (State of the art) <https://aclweb.org/aclwiki/Analogy_(State_of_the_art)>`_.

            Parameters
            ----------
            analogies : str
                Path to file, where lines are 4-tuples of words, split into sections by ": SECTION NAME" lines.
                See `gensim/test/test_data/questions-words.txt` as example.
            restrict_vocab : int, optional
                Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
                This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
                in modern word embedding models).
            case_insensitive : bool, optional
                If True - convert all words to their uppercase form before evaluating the performance.
                Useful to handle case-mismatch between training tokens and words in the test set.
                In case of multiple case variants of a single word, the vector for the first occurrence
                (also the most frequent if vocabulary is sorted) is taken.
            dummy4unknown : bool, optional
                If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
                Otherwise, these tuples are skipped entirely and not used in the evaluation.

            Returns
            -------
            score : float
                The overall evaluation score on the entire evaluation set
            sections : list of dict of {str : str or list of tuple of (str, str, str, str)}
                Results broken down by each section of the evaluation set. Each dict contains the name of the section
                under the key 'section', and lists of correctly and incorrectly predicted 4-tuples of words under the
                keys 'correct' and 'incorrect'.

            """
            ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
            ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)
            oov = 0
            logger.info("Evaluating word analogies for top %i words in the model on %s", restrict_vocab, analogies)
            sections, section = [], None
            quadruplets_no = 0
            with open(analogies, 'rb') as fin:
                for line_no, line in enumerate(fin):
                    line = line.decode('unicode-escape')
                    if line.startswith(': '):
                        # a new section starts => store the old section
                        if section:
                            sections.append(section)
                            self._log_evaluate_word_analogies(section)
                        section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
                    else:
                        if not section:
                            raise ValueError("Missing section header before line #%i in %s" % (line_no, analogies))
                        try:
                            if case_insensitive:
                                a, b, c, expected = [word.upper() for word in line.split()]
                            else:
                                a, b, c, expected = [word for word in line.split()]
                        except ValueError:
                            logger.info("Skipping invalid line #%i in %s", line_no, analogies)
                            continue
                        quadruplets_no += 1
                        if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                            oov += 1
                            if dummy4unknown:
                                logger.debug('Zero accuracy for line #%d with OOV words: %s', line_no, line.strip())
                                section['incorrect'].append((a, b, c, expected))
                            else:
                                logger.debug("Skipping line #%i with OOV words: %s", line_no, line.strip())
                            continue
                        original_vocab = self.vocab
                        self.vocab = ok_vocab
                        ignore = {a, b, c}  # input words to be ignored
                        predicted = None
                        # find the most likely prediction using 3CosAdd (vector offset) method
                        sims = self.most_similar(positive=[b, c], negative=[a], topn=5, restrict_vocab=restrict_vocab)
                        self.vocab = original_vocab
                        for element in sims:
                            predicted = element[0].upper() if case_insensitive else element[0]
                            if predicted in ok_vocab and predicted not in ignore:
                                if predicted != expected:
                                    logger.debug("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                                break
                        if predicted == expected:
                            section['correct'].append((a, b, c, expected))
                        else:
                            section['incorrect'].append((a, b, c, expected))
            if section:
                # store the last section, too
                sections.append(section)
                self._log_evaluate_word_analogies(section)

            total = {
                'section': 'Total accuracy',
                'correct': list(chain.from_iterable(s['correct'] for s in sections)),
                'incorrect': list(chain.from_iterable(s['incorrect'] for s in sections)),
            }

            oov_ratio = float(oov) / quadruplets_no * 100
            logger.info('Quadruplets with out-of-vocabulary words: %.1f%%', oov_ratio)
            if not dummy4unknown:
                logger.info(
                    'NB: analogies containing OOV words were skipped from evaluation! '
                    'To change this behavior, use "dummy4unknown=True"'
                )
            analogies_score = self._log_evaluate_word_analogies(total)
            sections.append(total)
            # Return the overall score and the full lists of correct and incorrect analogies
            return analogies_score, sections

    def __init__(self, sentences=[], size=0):
        self.wv = self.Wv(sentences)
        self.vector_size = size

    def similarity(self, a, b):
        return self.wv.similarity(a, b)

    def __str__(self):
        return "%s(vocab=%s, size=%s)" % (
            self.__class__.__name__, len(self.wv.vocab), self.vector_size
        )

    # TODO not working - check testing_embeddings
    @staticmethod
    def load(filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        mh = MultihotEmbedding()
        mh.__dict__.update(tmp_dict)
        return mh

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
