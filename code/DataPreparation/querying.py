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
        # TODO: if len(query) is small => add similar terms to the query (http://thomas.deselaers.de/teaching/files/tutorial_icpr08/03_textBasedRetrieval.pdf)
        similarities = self.lsi_vectorizer.get_similarities( documents )
        for query_similarities in similarities:
            query_similarities = sorted(enumerate(query_similarities), key=lambda item: -item[1])
            yield query_similarities[:n_docs]

    def get_similar_terms(self, terms, n_terms = 10):
        terms = [ term[0] for term in self.tokenizer.tokenize( terms ) ]
        ids = [ self.lsi_vectorizer.lexicon.token2id[term] for term in terms if term in self.lsi_vectorizer.lexicon.token2id ]
        return self.__get_term_neighbors( self.term_matrix[ids], n_terms )

    def get_docs_keywords (self, queries, n_terms = 10):
        """
        Getting keywords for the documents.
        """
        # TODO: or just use get_similar_terms() for each word in a query.
        documents = self.tokenizer.tokenize( queries )
        #documents_tfidf = self.lsi_vectorizer.docs_to_tfidf( documents, sparse = False ) # original TF-IDF (sparse): (n_queries, n_terms_in_lexicon)
        documents = self.lsi_vectorizer.get_docs_lsi_vectors(documents, sparse = False)
        documents_tfidf = np.dot( list(documents), self.term_matrix.T ) # approximate TF-IDF: (n_queries, n_terms_in_lexicon)        
        for document_tfidf in documents_tfidf:
            term_indices = np.argsort(-document_tfidf)[:n_terms]
            tfidfs = document_tfidf[term_indices]
            yield zip( term_indices, tfidfs )

    def __build_term_nearest_neighbors(self):
        term_singular_vectors = self.lsi_vectorizer.lsi.projection.u
        singular_values = self.lsi_vectorizer.lsi.projection.s
        #self.term_matrix = np.dot( term_singular_vectors, np.diag(singular_values) )
        self.term_matrix = term_singular_vectors
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

    model_version = "34M"
    n_factors = 269

    lsi_vectorizer = get_lsi_vectorizer (model_version, n_factors)
    #lsi_vectorizer.train()
    query = Query(LemmaTokenizer(), lsi_vectorizer)

    print("================")
    print ( lsi_vectorizer.lsi.projection.s )

    print("================")
    lsi_vectorizer.lsi.print_topics()

    doc = "music performance"

    for similarities in query.get_similar_docs([doc], n_docs = 20):
        print("================")
        for doc_index, similarity_value in similarities:
            print(similarity_value, doc_index, json.dumps(lsi_vectorizer.corpus[doc_index]))
    
    for similarities in query.get_similar_terms(["drummer"], n_terms = 20):
        print("================")
        for term_index, similarity_value in similarities:
            print( similarity_value, json.dumps(lsi_vectorizer.lexicon[term_index]) )
    
    for similarities in query.get_docs_keywords([doc], n_terms = 20):
        print("================")
        for term_index, similarity_value in similarities:
            print( similarity_value, json.dumps(lsi_vectorizer.lexicon[term_index]) )


