import numpy as np
from sklearn.base import TransformerMixin
from pymystem3 import Mystem
from gensim.models import KeyedVectors
import re

class Word2VecVectorizer(TransformerMixin):
    def __init__(self, keyed_vectors_path):
        self.vectors = KeyedVectors.load_word2vec_format(keyed_vectors_path, binary=True)
        self.zeros = np.zeros(self.vectors.vector_size)
        self.word2vec_mystem = Mystem(entire_input=False)

    def _get_text_vector(self, text):
        token_vectors = []
        for token in self.tokenize_with_mystem_pos(text):
            try:
                token_vectors.append(self.vectors[token])
            except KeyError:  # не нашли такой токен в словаре
                # print ('key error at {}'.format(token))
                pass

        if not token_vectors:
            return self.zeros

        text_vector = np.sum(token_vectors, axis=0)
        return text_vector / np.linalg.norm(text_vector)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([self._get_text_vector(text) for text in X])

    def tokenize_with_mystem_pos(self, text):
        result = []

        for item in self.word2vec_mystem.analyze(text):
            if item['analysis']:
                lemma = item['analysis'][0]['lex']
                pos = re.split('[=,]', item['analysis'][0]['gr'])[0]
                token = f'{lemma}_{pos}'
            else:
                token = f'{item["text"]}_UNKN'

            result.append(token)
        return result
