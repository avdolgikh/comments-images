# Goal
Using search query, retrieve relevant documents.

# Current state
Since the current system is based on [Latent Semantic Indexing](https://en.wikipedia.org/wiki/Latent_semantic_analysis), it is trivial to make query (including multi-term query).
Any query is considered as a new document (bag-of-words). And the system just shows all similar document from the corpus (sorted by relevance).
The second subtask of providing user with similar documents for her new message - is the same: the message is a bag-of-words (a new document).
Note: for now we consider that the corpus is in English (will check it later and think what to do).

Now I try using WordPiece tokenization and traning models on data preprocessed in this way.

## How to use
- use example from reading_corpus.py to read the corpus
- use example from querying.py to query documents using search words

# Ideas

## Initial
- Since we only have unannotaited data (message, title, etc.), we should use unsupervised paradigm (LDA/LSI).
- It must be sort of inverted indices (but with applying ML).
- SVD (for LSI) or like that can be applied.
So, it will be possible to calculate similarity between words and documents (in all combinations) as well as to rank all documents according to affinity to a query, etc.
- TF-IDF can be used as initial values of a doc-terms matrix.

## "Twitter-message" issue
Since the specific of the corpus is short messages/comments with rare words with reduction, slang, mistakes, etc., we can use subwords as tokens (~ fastText or [WordPiece](https://arxiv.org/pdf/1609.08144.pdf))

In terms of handling the problem of twit-length/slang/lexical mistakes, we can also use BERT model for document vectorization
(we can merge last layer to get (768,) vector as a presentation of a document). 
- We can use this index (<n_docs>, 768) itself for information retrival (using Nearest Neighbors on cosine similarity).
- Or we can build LSI above this (<n_docs>, 768) representation (instead of on TF-IDF).
Particlularly, we would like to use char-n-grams (like WordPiece), meanwhile avoiding BoW and incorporating an order in a sequence into the IR.
BERT (as a sequence model) or other RNN/Attemtion approaches can kelp with that.

Probably, n-grams (like 3-grams of words) are usefull too. It is more appropiate for BoW (than char-n-grams).

## What else
- corpus: hists of length, word distribution, etc. To have more intuition about corpus itself.
- observing the final vocabulary - to "understand" it.
- choose the best value for n_factors.
We can compute LSI for very big number of factors, and use only k first coordinates of vectors... But we should have kind of validation set (like annotated search results for specific queries).
Such an extrinsic approach seems better than just observing discarded energy spectrum (~ opposite to explained variance?) of truncated SVD.
Anyway we will try choosing n_factors so that last values in lsi.projection.s will be near to zero.
- We can use user name (?) [document[4]] simply as a separate word in a document.
Intuition: in this way we add kind of separate factors or grouping by specific user (her way of express herself, style, etc.)

# Further

## Comments
- Use comments as additional text data

## Using images
Let's say we have document representation (like LSI that we use now). We also can use pre-trained image representation (1d) - pre-softmax layer from ImageNet, GoogLeNet, etc.
We can train 2 neural converters - from doc and image - into a single space. We can search for neighbors of docs as well of images in this space. In this way we can enrich the output of the search system.
In general, we can use initial (GoogLeNet) image representation for that.
But the idea is to make even lower-dimension representation which corresponds particullary to the community of subreddits,
and use hidden semantic connections between texts and images that is traditional in this community .
![image-2-text.png](/docs/image-2-text.png)

## OOV problem
In order to cover words outside the corpus, we can use public pre-trained word embeddings (word2vec, GloVe, fastText, BERT, etc.).
We can train 2 neural converters - from initial word embedding (for a term) and from document representation (based on initial embeddings of its words) - into a single space.
The training set, in this case, is the original TF-IDF matrix of the corpus.
In this way we will have our own representation that can be used in the same way we use LSI representation now (same operations can be applied).
Benefit: possibility of using oov words.
![lower-dimension-word-embeddings.png](/docs/lower-dimension-word-embeddings.png)

## Can companion text reflect image well?
If a community has own culture - there is no very big varience in images related to similar contexts - it is possible.
We can probably use the supervised learning approach to check this. The model could translate text to image (or vise versa). And if small error is reachable, the assumption can be used.



