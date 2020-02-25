import json
import numpy as np
from sklearn.neighbors import NearestNeighbors

from preprocessing import LemmaTokenizer
from vectorizing import LSIVectorizer

class Query (object):
    def __init__(self, tokenizer, lsi_vectorizer):
        self.tokenizer = tokenizer
        self.lsi_vectorizer = lsi_vectorizer
        self.__build_term_nearest_neighbors()
    
    def get_similar_docs(self, queries, n_docs = 10):
        """
        queries : list of str.
            Each str is one query.
        Return:
            iterable of tuple(document index in the corpus, similarity to the query)
        """
        documents = self.tokenizer.tokenize( queries )
        similarities = self.lsi_vectorizer.get_similarities( documents )
        for query_similarities in similarities:
            query_similarities = sorted(enumerate(query_similarities), key=lambda item: -item[1])
            yield query_similarities[:n_docs]

    def get_similar_terms(self, terms, n_terms = 10):
        terms = [ term[0] for term in self.tokenizer.tokenize( terms ) ]
        ids = [ self.lsi_vectorizer.lexicon.token2id[term] for term in terms if term in self.lsi_vectorizer.lexicon.token2id ]
        return self.__get_term_neighbors( self.term_matrix[ids], n_terms )

    def get_similar_terms_to_docs (self, queries, n_terms = 10):
        documents = self.tokenizer.tokenize( queries )
        documents = self.lsi_vectorizer.get_docs_lsi_vectors(documents, sparse = False)
        return self.__get_term_neighbors( list(documents), n_terms )

    def __build_term_nearest_neighbors(self):
        term_singular_vectors = self.lsi_vectorizer.lsi.projection.u
        singular_values = self.lsi_vectorizer.lsi.projection.s
        self.term_matrix = np.dot( term_singular_vectors, np.diag(singular_values) )
        self.term_neighbors = NearestNeighbors(n_neighbors = 100, algorithm='auto', radius=1.0, metric="cosine", n_jobs=1)
        self.term_neighbors.fit( self.term_matrix )

    def __get_term_neighbors(self, item_embeddings, n_neighbors):
        distances, indices = self.term_neighbors.kneighbors( item_embeddings )
        for i, similar_term_indices in enumerate(indices):
            similarities = [ (1. - distance) for distance in distances[i][:n_neighbors] ]
            yield zip(similar_term_indices[:n_neighbors], similarities)

def get_lsi_vectorizer (model_version, n_factors):
    corpus_path = '../../models/{}/corpus_{}.pkl'.format(model_version, model_version)
    lexicon_path = '../../models/{}/lexicon'.format(model_version)
    tfidf_path = '../../models/{}/tfidf'.format(model_version)
    lsi_path = '../../models/{}/lsi'.format(model_version)
    corpus_index_path = '../../models/{}/corpus_index'.format(model_version)

    return LSIVectorizer(   corpus_path, n_factors = n_factors, lexicon_path = lexicon_path, tfidf_path = tfidf_path,
                            lsi_path = lsi_path, corpus_index_path = corpus_index_path)

        
if __name__ == '__main__':

    model_version = "20M"
    n_factors = 128
    lsi_vectorizer = get_lsi_vectorizer (model_version, n_factors)
    #lsi_vectorizer.train()
    query = Query(LemmaTokenizer(), lsi_vectorizer)


    #print ( lsi_vectorizer.lsi.projection.s )
    """
    n_factors = 128:
    [286.95597187 248.47479514 226.45186444 213.68360994 210.4693429
    ....
    43.7983927   43.62051112  38.91262603]
    """

    for similarities in query.get_similar_docs(["piano"], n_docs = 20):
        print("================")
        for doc_index, similarity_value in similarities:
            print(similarity_value, doc_index, json.dumps(lsi_vectorizer.corpus[doc_index]))

    for similarities in query.get_similar_terms(["piano"], n_terms = 20):
        print("================")
        for term_index, similarity_value in similarities:
            print( similarity_value, json.dumps(lsi_vectorizer.lexicon[term_index]) )

    for similarities in query.get_similar_terms_to_docs(["piano"], n_terms = 20):
        print("================")
        for term_index, similarity_value in similarities:
            print( similarity_value, json.dumps(lsi_vectorizer.lexicon[term_index]) )
    
    
    #doc = lsi_vectorizer.corpus[100]    
    #doc = " ".join(doc)
    #print(json.dumps(doc))


"""
query: ["piano"]
output:
0.873118    140825  ["music", "play", "keboards", "live", "launch", "stand"]
0.87277204  968977  ["music", "drummer", "local", "band", "play", "mostly", "pop", "rock", "cover", "dedication", "weekend", "performance"]
"""

