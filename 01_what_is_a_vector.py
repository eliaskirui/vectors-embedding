"""
Step 1: what is a vector, and how do you compare two of them?

No libraries, no AI. Just lists of numbers and a bit of maths.

Idea: describe things with numbers, and "similar things" become
"numbers that point in a similar direction".

Run:  python 01_what_is_a_vector.py
"""

import math

# Describe some animals with 3 hand-picked features, each scored 0-10:
#            [size, danger, fluffiness]
animals = {
    "kitten":  [1, 1, 9],
    "puppy":   [2, 1, 8],
    "lion":    [7, 9, 6],
    "leopard": [6, 9, 5],
    "shark":   [6, 10, 0],
    "goldfish": [1, 0, 0],
}
# Each list is a VECTOR: a point (or an arrow) in 3-dimensional space.
# Real embeddings work the same way, just with hundreds of dimensions
# and features the model learned itself instead of ones we picked.


def dot(a, b):
    """Multiply matching positions and add them up."""
    return sum(x * y for x, y in zip(a, b))


def length(a):
    """How long the arrow is (Pythagoras, in any number of dimensions)."""
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a, b):
    """
    How much two arrows point the same way, ignoring how long they are.
      1.0  = same direction (very similar)
      0.0  = at right angles (unrelated)
     -1.0  = opposite directions
    """
    return dot(a, b) / (length(a) * length(b))


def euclidean_distance(a, b):
    """Straight-line distance between the two points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


print("Cosine similarity to 'kitten' (higher = more similar):")
query = animals["kitten"]
ranked = sorted(animals.items(), key=lambda kv: cosine_similarity(query, kv[1]), reverse=True)
for name, vec in ranked:
    print(f"  {name:<9} {vec}  cos={cosine_similarity(query, vec):.3f}  "
          f"dist={euclidean_distance(query, vec):.2f}")

print("\nCosine similarity to 'lion':")
query = animals["lion"]
ranked = sorted(animals.items(), key=lambda kv: cosine_similarity(query, kv[1]), reverse=True)
for name, vec in ranked:
    print(f"  {name:<9} cos={cosine_similarity(query, vec):.3f}")

# Search for something that isn't in our list at all:
# "a big, dangerous, not-fluffy thing"
mystery = [8, 9, 1]
best = max(animals, key=lambda n: cosine_similarity(mystery, animals[n]))
print(f"\nMystery animal {mystery} is most like: {best}")
print("That's semantic search in miniature: describe what you want as a vector,")
print("then find the nearest stored vector.")

# Why cosine and not distance? Cosine ignores LENGTH.
# A "big kitten" [2, 2, 18] points the same way as kitten [1, 1, 9]:
big_kitten = [2, 2, 18]
print(f"\nkitten vs big_kitten: cos={cosine_similarity(animals['kitten'], big_kitten):.3f} "
      f"but dist={euclidean_distance(animals['kitten'], big_kitten):.2f}")
print("Same direction = same 'meaning', even though the points are far apart.")
