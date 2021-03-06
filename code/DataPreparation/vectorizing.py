import gensim
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim import similarities

import pickle
import os.path

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class LSIVectorizer (object):
    """
    Build LSI representation for the corpus.
    """

    def __init__(self, corpus_path, n_factors, lexicon_path, tfidf_path, lsi_path, corpus_index_path):
        with open(corpus_path, 'rb') as file:
            self.corpus = pickle.load(file)        
        self.n_factors = n_factors
        self.lexicon_path = lexicon_path
        self.tfidf_path = tfidf_path        
        self.lsi_path = lsi_path
        self.corpus_index_path = corpus_index_path
        self.lexicon =  Dictionary.load(self.lexicon_path) \
                        if os.path.exists(self.lexicon_path) \
                        else None
        self.tfidf =    TfidfModel.load(self.tfidf_path) \
                        if os.path.exists(self.tfidf_path) \
                        else None
        self.lsi =  LsiModel.load(self.lsi_path) \
                    if os.path.exists(self.lsi_path) \
                    else None
        # TODO: move it to Query class?
        self.corpus_index = similarities.MatrixSimilarity.load(self.corpus_index_path) \
                            if os.path.exists(self.corpus_index_path) \
                            else None

    def train(self):
        self.__to_lsi()

    def get_similarities(self, documents):
        """
        documents : iterable of lists of strings which are tokens of a document
        """
        return self.corpus_index[ self.lsi[ self.docs_to_tfidf( documents ) ] ]

    def get_docs_lsi_vectors(self, documents, sparse = True):
        """
        documents : iterable of lists of strings which are tokens of a document
        """
        if not sparse:
            for item in self.lsi[ self.docs_to_tfidf( documents ) ]:
                yield gensim.matutils.sparse2full( item, self.n_factors)
        else:
            return self.lsi[ self.docs_to_tfidf( documents ) ]



    def __to_tfidf(self):
        """
        Building corpus doc-term matrix with TF-IDF
        """
        self.lexicon = Dictionary( self.corpus )
        self.lexicon.save(self.lexicon_path)

        self.tfidf = TfidfModel([   self.lexicon.doc2bow( document )
                                    for document in self.corpus ],
                                id2word = self.lexicon ) # normalize=True <-- we can use it for a raw dirty corpus (?)
        self.tfidf.save(self.tfidf_path)

        return self.docs_to_tfidf(self.corpus)

    def docs_to_tfidf(self, documents, sparse = True):
        """
        documents : iterable of lists of strings which are tokens of a document
        """
        if self.tfidf is not None and self.lexicon is not None:
            for document in documents:
                #yield self.tfidf[ self.lexicon.doc2bow(document) ]
                if not sparse:                    
                    yield gensim.matutils.sparse2full( self.tfidf[ self.lexicon.doc2bow(document) ], len(self.lexicon))
                else:
                    yield self.tfidf[ self.lexicon.doc2bow(document) ]

    def __to_lsi(self):
        self.corpus_tfidf = list(self.__to_tfidf())
        self.lsi = LsiModel( self.corpus_tfidf, num_topics = self.n_factors, id2word = self.lexicon)
        # , chunksize=20000 - default, we can use 200000 (more memory usage!?) and speed up a bit??
        # , power_iters=2 - default, will 1 speed up?
        # , extra_samples = 100 - default (add more for accuracy? - speed down?)
        self.lsi.save(self.lsi_path)

        self.corpus_lsi = self.lsi[ self.corpus_tfidf ]
        self.corpus_index = similarities.MatrixSimilarity( self.corpus_lsi, num_features = self.n_factors, corpus_len = len(self.corpus)) # use Similarity for large corpus (> 2Gb)? # index=similarities.Similarity('E:\\cm_test',tfidf[corpus_tfidf],len(dictionary))
        self.corpus_index.save(self.corpus_index_path)

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

