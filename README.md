# Project 2: vectors-from-scratch

**Goal:** understand what an embedding is, how similarity is measured, and what a vector database really does, by building one in ~15 lines.

Part of the elibyte learning path:
1. chat-cli → **2. vectors-from-scratch** → 3. mini-rag → 4. rag-pgvector → 5. rag-evals → 6. agent → 7. docintel

No API key needed. Everything runs on your laptop.

---

## Setup

```bash
cd ~/elibyte/vectors-from-scratch
source .venv/bin/activate
```

If `.venv` is missing: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`

The first run of step 2 downloads the embedding model (~90 MB) into `~/.cache/huggingface`. After that it works offline.

You'll see some startup noise, such as `NOTE: Redirects are currently not supported in MacOs` and a `Loading weights` progress bar. It's harmless and comes from PyTorch, the library that runs the model.

---

## The four steps

| File | The one idea it teaches |
|---|---|
| `01_what_is_a_vector.py` | A vector is a list of numbers. **Cosine similarity** measures whether two vectors point the same way. Plain Python, no libraries |
| `02_real_embeddings.py` | An **embedding model** turns text into a 384-number vector where similar *meaning* gives similar direction |
| `03_semantic_search.py` | A **vector database** = store vectors, embed the query, return the nearest. Compared side by side with keyword search |
| `04_visualize.py` | Squash 384 dimensions to 2 and *see* topics form clusters. Writes `embedding_map.png` |

---

## Concepts you should be able to explain after this project

**Vector:** an ordered list of numbers, like `[1, 1, 9]`. You can think of it as a point, or an arrow from the origin, in N-dimensional space.

**Embedding:** a vector produced by a model from a piece of text (or image or audio). The model was trained so that texts with similar meaning end up close together. Individual numbers aren't human-readable; meaning is in the overall pattern.

**Dimensions:** how many numbers are in each vector. This model uses 384; big commercial ones use 1024–3072. More dimensions can capture more nuance but cost more to store and search.

**Cosine similarity:** `dot(a, b) / (|a| × |b|)`. 1 = same direction, 0 = unrelated. It ignores vector length, which is what you want for meaning. If you **normalize** vectors to length 1 first, it becomes a plain dot product, which is why every vector DB recommends normalizing.

**Semantic vs keyword search:** keyword search matches *words*, so "football" finds nothing if the text says "Arsenal" and "striker". Semantic search matches *meaning*. But keyword search wins for exact things like names, codes and IDs (try searching "KRA"). That's why real systems use **hybrid search** (both), which you'll build in Project 4.

**Vector database:** stores vectors plus the original text and finds nearest neighbours fast. Our version compares the query against *every* vector (brute force). That's fine for thousands of vectors, too slow for millions. Real vector DBs use **approximate nearest neighbour (ANN)** indexes such as HNSW to skip most comparisons.

**The same embedding model must be used for documents and queries.** Vectors from different models live in different "spaces" and can't be compared.

---

## Exercises

1. **Add an animal** in step 1, say `"hyena": [5, 7, 3]`. Predict which animal it's closest to *before* running.
2. **Fool the model.** In step 2, find two sentences with the *same* meaning that score low, or *different* meanings that score high. (Try negation: "I love this" vs "I do not love this".)
3. **Find where keyword search wins.** In step 3, search `KRA`, `Mombasa`, `M-Pesa`. Then search `taxes`. Which method handles which?
4. **Add your own topic.** Add 4–5 sentences about music to `sentences.txt`, add `["music"] * 5` to `topics` in step 4, and pick a colour. Does a new cluster appear?
5. **Measure the cost.** In step 3, add `import time` and time `semantic_search`. Then copy `documents` 1000 times (`documents = documents * 1000`, re-encode) and time it again. This is why ANN indexes exist.
6. **Think ahead to RAG.** If `sentences.txt` were 500 pages of company documents, which sentences would you paste into the chat prompt from Project 1 to answer "what's our refund policy?" You've just described Project 3.
