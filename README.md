# General idea
Using search query, retrieve relevant documents.

# Ideas
- Since we only have unannotaited data (message, title, etc.), we should cluster it (LDA/LSI) and recognize the most important words (kind of distribution of words according to their importance) in each cluster.
- It must be sort of inverted indices (but with applying ML).
- SVD (during LSI) or like that can be applied.
So, it will be possible to calculate similarity between words and documents (in all combinations) as well as to rank all documents according to affinity to a query, etc.
- TF-IDF can be used as initial values of a doc-terms matrix.

# How to use
- use example from reading_corpus.py to read the corpus
- use example from querying.py to query documents using search words

# What else
- corpus: hists of length, word distribution, etc.
- choose the best value for n_factors
- languages? only English?

# Further
- We can train own embeddings - building own NNs (2 inputs; I have some ideas)
- Some pre-trained word2vec can be used as word-embeddings to make the system able to work with words out-of-vocab
- Incorporate POS, dependencies, etc.
- Use images (I have ideas)
- Use comments as additional text data




