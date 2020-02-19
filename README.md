# General idea
Using search query, retrieve relevant documents.

# Ideas
- Since we only have unannotaited data (message, title, etc.), we should cluster it (LDA/LSI) and recognize the most important words (kind of distribution of words according to their importance) in each cluster.
- It must be sort of inverted indices (but with applying ML).
- SVD (during LSI) or like that can be applied.
So, it will be possible to calculate similarity between words and documents (in all combinations) as well as to rank all documents according to affinity to a query, etc.
- TF-IDF can be used as initial values of a doc-terms matrix.

# Current state
Since the current system is based on [Latent Semantic Indexing](https://en.wikipedia.org/wiki/Latent_semantic_analysis), it is trivial to make query (including multi-term query).
Any query is considered as a new document (bag-of-words). And the system just shows all similar document from the corpus (sorted by relevance).
The second subtask of providing user with similar documents for her new message - is the same: the message is a bag-of-words (a new document).

# How to use
- use example from reading_corpus.py to read the corpus
- use example from querying.py to query documents using search words

# What else
- corpus: hists of length, word distribution, etc.
- choose the best value for n_factors
- languages? only English?
- We can use user name (?) [document[4]] simply as a separate word in a document. Intuition: in this way we add kind of separates factors like grouping by specific user (her way of express herself, style, etc.)

# Further
- We can train own embeddings - building own NNs (2 inputs; I have some ideas)
- Some pre-trained word2vec can be used as word-embeddings to make the system able to work with words out-of-vocab
- Incorporate POS, dependencies, etc.
- Use images (I have ideas)
- Use comments as additional text data

# Using images
Let's say we have document representation (like LSI that we use now). We also can use pre-trained image representation (1d) - pre-softmax layer from ImageNet, GoogLeNet, etc.
We can train 2 neural converters - from doc and image - into a single space. We can search for neighbors of docs as well of images in this space. In this way we can enrich the output of the search system.
In general, we can use initial (GoogLeNet) image representation for that.
But the idea is to make even lower-dimension representation which corresponds particullary to the community of subreddits, abd use hidden semantic connections between texts and images that is traditional in this community .
![image-2-text.png](/docs/image-2-text.png  =650x420)



