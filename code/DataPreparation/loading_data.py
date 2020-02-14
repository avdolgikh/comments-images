import smart_open
import json
from nltk import sent_tokenize

class CorpusReader(object):
    def __init__(self, path):
        self.path = path

    def read(self):
        with smart_open.smart_open(self.path, 'rb', encoding='utf-8') as file:
            for document in file:
                document = json.loads(document)
                # 0: title, 1: message, 2: path to an image, 3: path to comments, 4: user nickname (?)
                # TODO: check if record[2] is path to image (filtering)
                yield document[:2]

    def sents(self):
        for document in self.read():            
            yield [ document[0] ] + [ sentence for sentence in sent_tokenize(document[1]) ]

if __name__ == '__main__':

    path = '../../Data/documents.gz'
    reader = CorpusReader(path)
    for document in reader.sents():
        print(document)

