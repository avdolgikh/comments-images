import nltk
from nltk import sent_tokenize, pos_tag, wordpunct_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


class TextNormalizer(object):
    """
    Tokenize and normalize (lemmatize) textual corpus.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def normalize(self, documents):
        """
        documents : iterable of list of list of strings (tokens)
            documents --> sentences --> tokens
        """
        for document in documents:
            yield [ self.lemmatize(token, tag).lower()
                    for sent in sent_tokenize(document)
                    for (token, tag) in pos_tag(wordpunct_tokenize(sent))
                    if token.isalpha() and len(token) > 2 and not self.is_stopword(token)
                  ]

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ,
            }.get(pos_tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def is_stopword(self, token):
        return token.lower() in self.stopwords