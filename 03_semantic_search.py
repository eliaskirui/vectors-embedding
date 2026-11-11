"""
Step 3: build a tiny vector database and search it.

Idea: a vector database is three things:
  1. STORE:  embed every document once and keep the vectors
  2. QUERY:  embed the question the same way
  3. SEARCH: find the stored vectors most similar to the question vector

Here that's ~15 lines of numpy. Real vector DBs (Chroma, pgvector, Pinecone)
do the same thing, plus saving to disk and clever indexes so search stays
fast with millions of vectors.

We also run a dumb KEYWORD search side by side so you can see where each one wins.

Run:  python 03_semantic_search.py
"""

import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- 1. STORE --------------------------------------------------------------
documents = Path(__file__).with_name("sentences.txt").read_text().strip().splitlines()

# One row per document: a matrix of shape (num_docs, 384).
# normalize_embeddings=True makes every vector length 1, so cosine similarity
# becomes a plain dot product (length(a) * length(b) == 1).
doc_vectors = model.encode(documents, normalize_embeddings=True)
print(f"Stored {len(documents)} documents as a {doc_vectors.shape} matrix.\n")


# --- 2 + 3. QUERY and SEARCH ---------------------------------------------
def semantic_search(query, k=3):
    query_vector = model.encode(query, normalize_embeddings=True)
    scores = doc_vectors @ query_vector         # one dot product per document, all at once
    top = np.argsort(scores)[::-1][:k]          # indexes of the k highest scores
    return [(float(scores[i]), documents[i]) for i in top]


def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


STOPWORDS = {"the", "a", "my", "i", "to", "is", "and", "of", "on", "in", "with", "do", "how", "for", "what"}


def keyword_search(query, k=3):
    """Score = how many (non-trivial) words the document shares with the query."""
    q = words(query) - STOPWORDS
    scored = [(len(q & words(doc)), doc) for doc in documents]
    scored.sort(key=lambda s: s[0], reverse=True)
    return [s for s in scored[:k] if s[0] > 0]


def show(query):
    print(f'QUERY: "{query}"')
    print("  semantic:")
    for score, doc in semantic_search(query):
        print(f"    {score:.3f}  {doc}")
    print("  keyword:")
    hits = keyword_search(query)
    for score, doc in hits:
        print(f"    {score} words  {doc}")
    if not hits:
        print("    (no shared words: nothing found)")
    print()


# A few examples to start. Watch where keyword search fails.
for q in [
    "sending cash to family",     # no word overlap with the M-Pesa sentences
    "football",                    # the word 'football' appears nowhere!
    "my code is broken",
    "what should I cook tonight",
]:
    show(q)

print("Look closely: semantic search ALWAYS returns k results, even when the 3rd one")
print("is irrelevant (e.g. weather for 'my code is broken'). It has no idea of 'no match'.")
print("In RAG this matters: irrelevant chunks get pasted into the prompt. Fixes are a")
print("minimum score threshold, or letting the LLM ignore chunks that don't help.\n")

# Now try your own.
print("Type your own searches (empty line to quit).")
while True:
    try:
        q = input("search> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        break
    show(q)
