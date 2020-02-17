from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim import similarities

import pickle

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class LSIVectorizer (object):
    """
    Build LSI representation for the corpus.
    """

    def __init__(self, corpus_path, n_factors):
        with open(corpus_path, 'rb') as file:
            self.corpus = pickle.load(file)        
        self.n_factors = n_factors
        self.tfidf = None
        self.lexicon = None
        

    def train(self):
        self.__to_lsi()

    def add_documents(self, documents):
        """
        documents : iterable of lists of strings which are tokens of a document
        """
        self.lsi.add_documents( self.__docs_to_tfidf( documents ) )

    def get_similarities(self, documents):
        """
        documents : iterable of lists of strings which are tokens of a document
        """
        return self.corpus_index[ self.lsi[ self.__docs_to_tfidf( documents ) ] ]



    def __to_tfidf(self):
        """
        Building corpus doc-term matrix with TF-IDF
        """
        self.lexicon = Dictionary( self.corpus )
        self.tfidf = TfidfModel([   self.lexicon.doc2bow( document )
                                    for document in self.corpus ],
                                id2word = self.lexicon ) # normalize=True <-- we can use it for a raw dirty corpus (?)
        # save (into a file)
        return self.__docs_to_tfidf(self.corpus)

    def __docs_to_tfidf(self, documents):
        """
        documents : iterable of lists of strings which are tokens of a document
        """
        if self.tfidf is not None and self.lexicon is not None:
            for document in documents:
                yield self.tfidf[ self.lexicon.doc2bow(document) ]

    def __to_lsi(self):
        self.corpus_tfidf = list(self.__to_tfidf())
        self.lsi = LsiModel( self.corpus_tfidf, num_topics = self.n_factors, id2word = self.lexicon)
        # save: self.lsi.save(fname)
        self.corpus_lsi = self.lsi[ self.corpus_tfidf ]
        self.corpus_index = similarities.MatrixSimilarity( self.corpus_lsi, num_features = len(self.lexicon), corpus_len = len(self.corpus)) # use Similarity for large corpus (> 2Gb)? # index=similarities.Similarity('E:\\cm_test',tfidf[corpus_tfidf],len(dictionary))
        return self.corpus_lsi
    
"""
lsi.projection.u.shape
lsi.projection.s.shape
topics = lsi.get_topics()
lsi.print_debug()
lsi.print_topic(11)
for topic_index in range(5):
    print( [ (lexicon[int(word_id)], value) for (word_id, value) in lsi.show_topic(topicno = topic_index, topn = 10)  ] )

to validate k - n_factors: gensim.models.lsimodel.clip_spectrum(s, k, discard=0.001)
LSI training is unique in that we can continue “training” at any point, simply by providing more training documents. This is done by incremental updates to the underlying model, in a process called online training.
Hierarchical Dirichlet Process, HDP is a non-parametric bayesian method (note the missing number of requested topics):
model = models.HdpModel(corpus, id2word=dictionary)
gensim uses a fast, online implementation based on 3. The HDP model is a new addition to gensim, and still rough around its academic edges – use with care.
"""

