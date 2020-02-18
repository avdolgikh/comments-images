import json

from normalizing import TextNormalizer
from vectorizing import LSIVectorizer

class Query (object):
    def __init__(self, normalizer, lsi_vectorizer):
        self.normalizer = normalizer
        self.lsi_vectorizer = lsi_vectorizer
    
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

        
if __name__ == '__main__':
    corpus_path = '../../Data/corpus_1M.pkl'
    lexicon_path = '../../Data/lexicon'
    tfidf_path = '../../Data/tfidf'
    lsi_path = '../../Data/lsi'
    corpus_index_path = '../../Data/corpus_index'

    lsi_vectorizer = LSIVectorizer(corpus_path, n_factors = 128, lexicon_path = lexicon_path, tfidf_path = tfidf_path, lsi_path = lsi_path, corpus_index_path = corpus_index_path)
    #lsi_vectorizer.train()
    query = Query(TextNormalizer(), lsi_vectorizer)

    
    
    for similarities in query.get_similar_docs(["piano"], n_docs = 20):
        print("================")
        for index_in_corpus, similarity_value in similarities:
            print(similarity_value, index_in_corpus, json.dumps(lsi_vectorizer.corpus[index_in_corpus]))


"""
query: ["piano"]
output:
0.873118    140825  ["music", "play", "keboards", "live", "launch", "stand"]
0.87277204  968977  ["music", "drummer", "local", "band", "play", "mostly", "pop", "rock", "cover", "dedication", "weekend", "performance"]
"""

