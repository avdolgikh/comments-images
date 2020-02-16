from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel
import pickle


class Query (object):
    def __init__(self, corpus_path):
        with open(corpus_path, 'rb') as file:
            corpus = pickle.load(file)
        # or use this way during the pre-processing?
        self.corpus = [ sum([ [ token for (token, tag) in sent] for sent in document ], []) for document in corpus ]

    def vectorize(self):
        """
        Doc-Term Matrix with TF-IDF
        """
        #self.lexicon = Dictionary( self.corpus ) # check if not None
        self.tfidf = TfidfModel([   self.lexicon.doc2bow( document )
                                    for document in self.corpus ],
                                id2word = self.lexicon )
        # save

        for document in self.corpus:
            yield self.tfidf[ self.lexicon.doc2bow(document) ]

    def build_lsa_model(self):
        self.lexicon = Dictionary( self.corpus )
        self.corpus_tfidf = list(self.vectorize())
        self.lsi = LsiModel( self.corpus_tfidf, num_topics = 128, id2word = self.lexicon)
        #self.lsi.save(tmp_fname)
        # model.add_documents(common_corpus[4:])  # update model with new documents
        self.corpus_lsi = self.lsi[ self.corpus_tfidf ] 

    def get_similar_terms(self, term, top=10):
        pass

    def get_similar_docs(self, doc, top=10):
        pass

    def get_similar_docs_for_term(self, term, top=10):
        pass

    def get_similar_docs_for_multiterms(self, terms, top=10):
        pass
    
        
if __name__ == '__main__':
    path = '../../Data/corpus_small.pkl'
    query = Query(path)
    #for vector in query.vectorize():
    #    print(vector)
    query.build_lsa_model()
    #topics = query.lsi.get_topics()
    #print(topics.shape)
    print( list(query.corpus_lsi) )


