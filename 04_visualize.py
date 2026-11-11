"""
Step 4: SEE the meaning space.

Idea: we can't picture 384 dimensions, but we can squash them down to 2
while keeping as much of the spread as possible (PCA). Sentences about the
same topic should land near each other, forming clusters.

Saves: embedding_map.png

Run:  python 04_visualize.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

here = Path(__file__).parent
model = SentenceTransformer("all-MiniLM-L6-v2")
documents = (here / "sentences.txt").read_text().strip().splitlines()
vectors = model.encode(documents, normalize_embeddings=True)  # shape (30, 384)

# Topic labels, only used for colouring. The model never sees these.
# sentences.txt is grouped: 6 money, 5 football, 6 code, 5 food, 5 travel, 3 weather
topics = ["money"] * 6 + ["football"] * 5 + ["code"] * 6 + ["food"] * 5 + ["travel"] * 5 + ["weather"] * 3 + ["music"] * 5
assert len(topics) == len(documents), "update the topics list if you edit sentences.txt"

# --- PCA from scratch -------------------------------------------------------
# 1. centre the data so the average point is at the origin
centred = vectors - vectors.mean(axis=0)
# 2. SVD finds the directions along which the points are most spread out
_, _, directions = np.linalg.svd(centred, full_matrices=False)
# 3. project every 384-d point onto the top 2 directions -> (x, y)
points_2d = centred @ directions[:2].T
# (sklearn.decomposition.PCA does exactly this in one line.)

colours = {"money": "#2a9d8f", "football": "#e76f51", "code": "#264653",
           "food": "#e9c46a", "travel": "#8338ec", "weather": "#4895ef", "music": "#f4a261"}

fig, ax = plt.subplots(figsize=(13, 9))
for topic, colour in colours.items():
    idx = [i for i, t in enumerate(topics) if t == topic]
    ax.scatter(points_2d[idx, 0], points_2d[idx, 1], c=colour, s=80, label=topic)
for (x, y), text in zip(points_2d, documents):
    ax.annotate(text[:38] + ("…" if len(text) > 38 else ""), (x, y),
                fontsize=7, xytext=(5, 3), textcoords="offset points")
ax.set_title("30 sentences, 384 dimensions squashed to 2 (colour = topic, unknown to the model)")
ax.legend()
ax.grid(alpha=0.3)

out = here / "embedding_map.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved {out}. Open it and look for clusters.")
print("Squashing 384 dims to 2 loses information, so some overlap is expected.")
