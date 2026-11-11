"""
Step 2: real embeddings.

Idea: an EMBEDDING MODEL turns text into a vector, the same as our animal
lists but with 384 numbers chosen by a neural network. Texts with similar
MEANING get vectors that point in similar directions, even if they share
no words.

The model (all-MiniLM-L6-v2, ~90 MB) runs on your laptop: free, offline,
no API key. The first run downloads it; later runs are instant.

Run:  python 02_real_embeddings.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- 1. Look at one embedding -------------------------------------------
vec = model.encode("I paid my electricity bill using M-Pesa.")
print("An embedding is just a list of numbers:")
print("  type: ", type(vec).__name__)
print("  shape:", vec.shape, "  <- 384 dimensions")
print("  first 8 values:", np.round(vec[:8], 3))
print("  (No single number means anything on its own. Meaning lives in the pattern.)")


# --- 2. Cosine similarity, now with numpy ---------------------------------
def cosine_similarity(a, b):
    # Exactly the same formula as step 1, written with numpy.
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


pairs = [
    # same meaning, different words
    ("I paid my electricity bill using M-Pesa.", "I settled the power bill with mobile money."),
    # same topic
    ("The striker scored a goal.", "Arsenal won the football match."),
    # shared words, different meaning
    ("I sat on the river bank.", "I deposited cash at the bank."),
    # unrelated
    ("I paid my electricity bill using M-Pesa.", "Elephants live in the Maasai Mara."),
]

print("\nSimilarity between pairs of sentences:")
for a, b in pairs:
    score = cosine_similarity(model.encode(a), model.encode(b))
    bar = "#" * int(max(score, 0) * 40)
    print(f"\n  {score:.3f} {bar}\n    A: {a}\n    B: {b}")

print("\nNotice:")
print("  - Pair 1 shares almost no words but scores highest: the model captures MEANING.")
print("  - Pair 4 (unrelated) scores near zero.")
print("  - But compare pairs 2 and 3. The 'bank' pair shares a word yet means different")
print("    things, and it may still beat the football pair. Small models are imperfect;")
print("    shared words still pull scores up. Scores are RELATIVE, not absolute truth.")
