import smart_open
import json
import nltk
from nltk import sent_tokenize, pos_tag, wordpunct_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import pickle


class CorpusReader(object):
    def __init__(self, path):
        self.path = path
        self.image_extensions = {".jpg", "jpeg", ".png", ".gif", ".bmp"}
        self.corpus_size = 0
        self.sent_lens = []
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def read(self):
        prev_line = ""
        with smart_open.smart_open(self.path, 'rb', encoding='utf-8') as file:
            for line in file:
                if line.endswith("]\n"):
                    document = json.loads(prev_line + line)
                    prev_line = ""
                    # 0: title, 1: message, 2: path to an image, 3: path to comments, 4: user nickname (?)
                    # filter only English?
                    if any(document[2].endswith(ext) for ext in self.image_extensions):
                        self.corpus_size += 1
                        yield document[:2]
                else:
                    prev_line += line

    def sents(self):
        for document in self.read():
            sentences = [ sentence for sentence in sent_tokenize(document[1]) ]
            self.sent_lens.append( len(sentences) )
            yield [ document[0] ] + sentences

    def tokenize(self):
        for document in self.sents():
            yield [ pos_tag(wordpunct_tokenize(sent))
                    for sent in document ]

    def normalize(self):
        # TODO: check empty sentences?
        for document in self.tokenize():
            yield   [   [   (self.lemmatize(token, tag).lower(), tag)
                            for (token, tag) in sent
                            if token.isalpha() and len(token) > 2 and not self.is_stopword(token)   ]
                        for sent in document    ]

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


if __name__ == '__main__':
    path = '../../Data/documents.gz'
    reader = CorpusReader(path) # 206 669 601

    corpus = []
    for document in reader.normalize():
        corpus.append( document )
        if reader.corpus_size % 100 == 0:
            print(reader.corpus_size)
            if reader.corpus_size == 1000000:
                with open('../../Data/corpus_1M.pkl', 'wb') as file:
                    pickle.dump(corpus, file, pickle.HIGHEST_PROTOCOL)

    with open('../../Data/corpus.pkl', 'wb') as file:
        pickle.dump(corpus, file, pickle.HIGHEST_PROTOCOL)

    with open('../../Data/corpus.pkl', 'rb') as file:
        corpus = pickle.load(file)


