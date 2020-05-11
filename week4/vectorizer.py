import numpy as np
from sklearn.base import TransformerMixin
from pymystem3 import Mystem
from gensim.models import KeyedVectors
import re

class Vectorizer(TransformerMixin):
    def __init__(self, keyed_vectors_path, vocab_size=5000, max_len=200):
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        self.pad_id = 0
        self.start_id = 1
        self.oov_id = 2
        self.index_offset = 3
        
        self.vectors = KeyedVectors.load_word2vec_format(keyed_vectors_path, binary=True)
        self.zeros = np.zeros(self.vectors.vector_size)
        self.word2vec_mystem = Mystem(entire_input=False)

#     def _get_text_vector(self, text):
#         token_vectors = []
#         for token in self.tokenize_with_mystem_pos(text):
#             try:
#                 token_vectors.append(self.vectors[token])
#             except KeyError:
#                 # print ('key error at {}'.format(token))
#                 pass

#         if not token_vectors:
#             return self.zeros

#         text_vector = np.sum(token_vectors, axis=0)
#         return text_vector / np.linalg.norm(text_vector)

    def fit(self, X, y=None):
        tokens = [self.tokenize_with_mystem_pos(t) for t in X]
        tokens = [item for sublist in tokens for item in sublist]
        w, c = np.unique(tokens, return_counts=1)
        self.frequent_words = list(w[c.argsort()[::-1]])[:self.vocab_size-self.index_offset]
        
        self.word_index = dict((word, j+self.index_offset) for (j, word) in enumerate(self.frequent_words))
        self.word_index['<PAD>'] = self.pad_id
        self.word_index['<START>'] = self.start_id
        self.word_index['<OOV>'] = self.oov_id
        self.inverted_word_index = dict((v,k) for (k,v) in self.word_index.items())
        
        embs = [self.vectors[word] for word in self.word_index if word in self.vectors]
        self.mean_emb = np.stack(embs).mean(0)
        self.mean_emb = self.mean_emb / np.linalg.norm(self.mean_emb)
        return self

    def transform(self, X, y=None):
        res = []
        lens = []
        for text in X:
            tokens = self.tokenize_with_mystem_pos(text)
            text_code = [self.start_id] + [self.word_index[token] if token in self.word_index else self.oov_id for token in tokens]
            #text_len = min(self.max_len, len(text_code))
            
            if len(text_code) <= self.max_len:
                text_len = len(text_code)
                text_code += [self.pad_id] * (self.max_len - len(text_code))
            else:
                text_len = self.max_len
                text_code = text_code[:self.max_len]
        
    
            res.append(text_code)
            lens.append(text_len)
        res = np.vstack(res)
        return res, lens

    def get_embeddings(self):
        embeddings = []
        for i in range(len(self.inverted_word_index)):
            word = self.inverted_word_index[i]
            if word in self.vectors:
                #print (word, 'in vectors')
                embeddings.append(self.vectors[word])
            else:
                embeddings.append(self.mean_emb)
        embeddings = np.stack(embeddings)
        return embeddings   
    
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