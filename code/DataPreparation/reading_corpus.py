import smart_open
import json
import pickle
import matplotlib.pyplot as plt
from itertools import islice

from preprocessing import LemmaTokenizer, WordpieceTokenizer



class CorpusReader(object):
    def __init__(self, path):
        self.path = path
        self.image_extensions = {".jpg", "jpeg", ".png", ".gif", ".bmp"}
        self.corpus_size = 0
        self.n_processed_lines = 0

    def read(self, start_line = None, stop_line = None):
        reading_started = False
        prev_lines = ""

        if start_line is not None:
            self.n_processed_lines = start_line

        with smart_open.smart_open(self.path, 'rb', encoding='utf-8') as file:
            for line in islice(file, start_line, stop_line):
                self.n_processed_lines += 1

                if line.startswith('["'):
                    reading_started = True

                if line.endswith("]\n"):
                    document = json.loads(prev_lines + line)
                    prev_lines = ""
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

                elif reading_started:
                    prev_lines += line

    


def hist_document_lens (document_lens):
    plt.figure(figsize=(5, 6), dpi=150)
    plt.hist(document_lens, bins=100)
    plt.grid(True)
    plt.title("Number of words in a document.")
    plt.show()


def combine_corpora(corpus_path_list, result_corpus_path):
    corpus = []
    for raw_data_path in corpus_path_list:
        with open(raw_data_path, 'rb') as file:
            corpus += pickle.load(file)
    with open(result_corpus_path, 'wb') as file:
        pickle.dump(corpus, file, pickle.HIGHEST_PROTOCOL)
    print( "Combined!" )



if __name__ == '__main__':

    raw_data_path = '../../Data/documents.gz'
    reader = CorpusReader(raw_data_path) # 206.7M lines, 40155032 docs (filtered)
    tokenizer = WordpieceTokenizer(vocab_file = '../../models/bert_vocab.txt')

    raw_docs = []
    corpus = []
    document_lens = []
    
    for raw_doc, tokenized_doc in tokenizer.tokenize_with_raw( reader.read() ):
        raw_docs.append( raw_doc )
        corpus.append( tokenized_doc )
        document_lens.append( len(tokenized_doc) )

        if reader.corpus_size % 100 == 0:
            print("lines: {}, docs: {}.".format( reader.n_processed_lines, reader.corpus_size ) )

    with open('../../Data/corpus_wp.pkl', 'wb') as file:
        pickle.dump(corpus, file, pickle.HIGHEST_PROTOCOL)

    with open('../../Data/raw_docs.pkl', 'wb') as file:
        pickle.dump(raw_docs, file, pickle.HIGHEST_PROTOCOL)

    with open('../../Data/corpus_wp.pkl', 'rb') as file:
        corpus = pickle.load(file)

    hist_document_lens(document_lens)


    
    



