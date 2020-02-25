

from wordpiece_tokenization import FullTokenizer        

class WordpieceTokenizer (object):
    """
    Using WordPiece tokenization https://arxiv.org/pdf/1609.08144.pdf.
    Core code is taken from BERT.
    """
    def __init__(self, vocab_file):
        self.tokenizer = FullTokenizer(vocab_file = vocab_file, do_lower_case = True)

    def tokenize(self, documents):
        """
        documents : iterable of list of strings (texts)
        Return: 
            iterator of list of tokens
        """
        for document in documents:
            yield self.tokenizer.tokenize(str(document))
    
    def tokenize_with_raw (self, documents):
        """
        documents : iterable of list of strings (texts)
        Return: 
            iterator of (raw document, list of tokens)
        """
        for document in documents:
            yield document, self.tokenizer.tokenize(str(document))


from nltk import sent_tokenize, pos_tag, wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import nltk        
from nltk.corpus import wordnet as wn

class LemmaTokenizer (object):
    """
    Tokenize and normalize (lemmatize) textual corpus.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def tokenize(self, documents):
        """
        documents : iterable of list of strings (texts)
        Return: 
            iterator of list of tokens
        """
        for document in documents:
            yield   [   self.__lemmatize(token, tag).lower()
                        for sent in sent_tokenize(document)
                        for (token, tag) in pos_tag(wordpunct_tokenize(sent))
                        if token.isalpha() and len(token) > 2 and not self.__is_stopword(token)
                    ]

    def tokenize_with_raw (self, documents):
        """
        documents : iterable of list of strings (texts)
        Return: 
            iterator of (raw document, list of tokens)
        """
        for document in documents:
            yield document, [   self.__lemmatize(token, tag).lower()
                                for sent in sent_tokenize(document)
                                for (token, tag) in pos_tag(wordpunct_tokenize(sent))
                                if token.isalpha() and len(token) > 2 and not self.__is_stopword(token)
                            ]

    def __lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ,
            }.get(pos_tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def __is_stopword(self, token):
        return token.lower() in self.stopwords


