import json
import numpy as np
from sklearn.neighbors import NearestNeighbors

from normalizing import TextNormalizer
from vectorizing import LSIVectorizer

class Query (object):
    def __init__(self, normalizer, lsi_vectorizer):
        self.normalizer = normalizer
        self.lsi_vectorizer = lsi_vectorizer
        self.__build_term_nearest_neighbors()
    
    def get_similar_docs(self, queries, n_docs = 10):
        """
        queries : list of str.
            Each str is one query.
        Return:
            iterable of tuple(document index in the corpus, similarity to the query)
        """
        documents = self.normalizer.normalize( queries )
        similarities = self.lsi_vectorizer.get_similarities( documents )
        for query_similarities in similarities:
            query_similarities = sorted(enumerate(query_similarities), key=lambda item: -item[1])
            yield query_similarities[:n_docs]

    def get_similar_terms(self, terms, n_terms = 10):
        terms = list(self.normalizer.normalize( terms ))[0]
        ids = [ self.lsi_vectorizer.lexicon.token2id[term] for term in terms if term in self.lsi_vectorizer.lexicon.token2id ]
        distances, indices = self.term_neighbors.kneighbors( self.term_matrix[ids] )
        for i, similar_term_indices in enumerate(indices):
            yield similar_term_indices[:n_terms] # TODO: return similarity values too
            #yield (similar_term_indices[:n_terms], distances[i])

    def __build_term_nearest_neighbors(self):
        term_singular_vectors = self.lsi_vectorizer.lsi.projection.u
        singular_values = self.lsi_vectorizer.lsi.projection.s
        self.term_matrix = np.dot( term_singular_vectors, np.diag(singular_values) )
        #self.term_matrix = term_singular_vectors 
        #normalized_term_matrix = term_matrix / np.linalg.norm( term_matrix, axis = 0 )        
        self.term_neighbors = NearestNeighbors(n_neighbors = 100, algorithm='auto', radius=1.0, metric="cosine", n_jobs=1)
        self.term_neighbors.fit( self.term_matrix )

        
if __name__ == '__main__':
    corpus_path = '../../Data/corpus_1M.pkl'
    lexicon_path = '../../Data/lexicon'
    tfidf_path = '../../Data/tfidf'
    lsi_path = '../../Data/lsi'
    corpus_index_path = '../../Data/corpus_index'

    lsi_vectorizer = LSIVectorizer(corpus_path, n_factors = 128, lexicon_path = lexicon_path, tfidf_path = tfidf_path, lsi_path = lsi_path, corpus_index_path = corpus_index_path)
    #lsi_vectorizer.train()
    query = Query(TextNormalizer(), lsi_vectorizer)

    for term_indices in query.get_similar_terms(["piano"], n_terms = 10):
        print("================")
        for term_index in term_indices:
            print( lsi_vectorizer.lexicon[term_index] )
    
    for similarities in query.get_similar_docs(["animals"], n_docs = 20):
        print("================")
        for index_in_corpus, similarity_value in similarities:
            print(similarity_value, index_in_corpus, json.dumps(lsi_vectorizer.corpus[index_in_corpus]))

"""
piano
accordion
play
phwoar
sawblade
trog
sinper
kuvakei
starcarft
magneticballs
"""



"""
query: ["piano"]
output:
0.873118    140825  ["music", "play", "keboards", "live", "launch", "stand"]
0.87277204  968977  ["music", "drummer", "local", "band", "play", "mostly", "pop", "rock", "cover", "dedication", "weekend", "performance"]
"""

