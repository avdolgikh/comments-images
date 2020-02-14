# General idea
Using search query, retrieve relevant documents.

# Ideas
- Since we only have unannotaited data (message, title, etc.), we should cluster it (LDA/LSI) and recognize the most important words (kind of distribution of words according to their importance) in each cluster.
- It must be sort of inverted indices (but with applying ML).
- SVD (during LSA) or like that can be applied.
So, it will be possible to calculate similarity between words and documents (in all combinations) as well as to rank all documents according to affinity to a query, etc.
- TF-IDF can be used as initial values of a doc-terms matrix.
- Some pre-trained word2vec can be used as word-embeddings to make the system able to work with words outside the corpus ("documents.gz")
- Later we can desing a neural net model with sequence of words (embeddings) as an input and a document embedding as an output. The latter can be used in kNN search to provide user with M relevant documents.



