import smart_open
import json
import pickle
import matplotlib.pyplot as plt

from normalizing import TextNormalizer


class CorpusReader(object):
    def __init__(self, path):
        self.path = path
        self.image_extensions = {".jpg", "jpeg", ".png", ".gif", ".bmp"}
        self.corpus_size = 0       

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
                        try:
                            if document[0] is not None and document[1] is not None:
                                yield ". ".join(document[:2])
                        except Exception as ex:
                            self.corpus_size -= 1
                            print(ex)
                else:
                    prev_line += line


def hist_document_lens (document_lens):
    plt.figure(figsize=(5, 6), dpi=150)
    plt.hist(document_lens, bins=100)
    plt.grid(True)
    plt.title("Number of words in a document.")
    plt.show()



if __name__ == '__main__':
    path = '../../Data/documents.gz'
    reader = CorpusReader(path) # 206 669 601
    normalizer = TextNormalizer()

    corpus = []
    document_lens = []
    
    for document in normalizer.normalize( reader.read() ):
        corpus.append( document )
        
        document_lens.append( len(document) )

        if reader.corpus_size % 100 == 0:
            print(reader.corpus_size)

            if reader.corpus_size == 12000000:
                with open('../../Data/corpus_12M.pkl', 'wb') as file:
                    pickle.dump(corpus, file, pickle.HIGHEST_PROTOCOL)

            if reader.corpus_size == 20000000:
                with open('../../Data/corpus_20M.pkl', 'wb') as file:
                    pickle.dump(corpus, file, pickle.HIGHEST_PROTOCOL)

            if reader.corpus_size == 100000000:
                with open('../../Data/corpus_100M.pkl', 'wb') as file:
                    pickle.dump(corpus, file, pickle.HIGHEST_PROTOCOL)

    with open('../../Data/corpus.pkl', 'wb') as file:
        pickle.dump(corpus, file, pickle.HIGHEST_PROTOCOL)

    with open('../../Data/corpus.pkl', 'rb') as file:
        corpus = pickle.load(file)

    hist_document_lens(document_lens)


    
    



