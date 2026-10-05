"""Sweep semantic-cache thresholds over labeled question pairs.

For each threshold t, a pair "hits" if cosine(q1, q2) >= t.
  hit rate   = share of SAME pairs that hit      (want high)
  false hits = number of DIFFERENT pairs that hit (want 0)
Pick CACHE_HIGH at or above the highest DIFFERENT score (0 false hits without the LLM),
and CACHE_LOW near the lowest SAME score; pairs in between go to the LLM check.

Run: python -m scripts.eval_cache
"""
from __future__ import annotations

import numpy as np

from app.embeddings import embed

# (question 1, question 2, same answer?)
PAIRS: list[tuple[str, str, bool]] = [
    ("What is photosynthesis?", "Explain how photosynthesis works", True),
    ("What is photosynthesis?", "How do plants make their own food?", True),
    ("State Ohm's law", "What does Ohm's law say?", True),
    ("What is a food chain?", "Define food chain", True),
    ("Why is the sky blue?", "What makes the sky appear blue?", True),
    ("What is the function of the heart?", "What does the heart do?", True),
    ("What are the uses of baking soda?", "Give uses of sodium hydrogencarbonate", True),
    ("What is corrosion?", "Explain the meaning of corrosion", True),
    ("What is the pH of pure water?", "Pure water has what pH value?", True),
    ("What causes tooth decay?", "Why do our teeth decay?", True),
    ("What is the role of the placenta?", "Explain the function of the placenta", True),
    ("What is refraction of light?", "Define refraction of light", True),
    ("function of xylem", "structure of xylem", False),
    ("What is the function of xylem?", "What is the function of phloem?", False),
    ("What is a concave mirror?", "What is a convex mirror?", False),
    ("What is an acid?", "What is a base?", False),
    ("Resistance of 2 resistors in series", "Resistance of 3 resistors in series", False),
    ("What is aerobic respiration?", "What is anaerobic respiration?", False),
    ("Define myopia", "Define hypermetropia", False),
    ("What are the uses of baking soda?", "What are the uses of washing soda?", False),
    ("Explain the working of the human eye", "Explain the defects of the human eye", False),
    ("What is an exothermic reaction?", "What is an endothermic reaction?", False),
    ("What are dominant traits?", "What are recessive traits?", False),
    ("Advantages of AC over DC", "Advantages of DC over AC", False),
]

THRESHOLDS = np.round(np.arange(0.60, 0.981, 0.02), 2)


def main() -> None:
    q1 = embed([p[0] for p in PAIRS])
    q2 = embed([p[1] for p in PAIRS])
    sims = np.sum(q1 * q2, axis=1)  # normalized vectors: dot == cosine
    same = np.array([p[2] for p in PAIRS])

    print("Pair similarities:")
    for (a, b, lbl), s in sorted(zip(PAIRS, sims), key=lambda x: -x[1]):
        print(f"  {s:.3f}  {'SAME' if lbl else 'DIFF'}  {a!r} vs {b!r}")

    print(f"\n{'threshold':>9}  {'hit rate':>8}  {'false hits':>10}")
    for t in THRESHOLDS:
        hit = sims >= t
        print(f"{t:>9.2f}  {hit[same].mean():>8.0%}  {int(hit[~same].sum()):>7}/{(~same).sum()}")
    print(f"\nmin SAME = {sims[same].min():.3f}, max DIFF = {sims[~same].max():.3f}")


if __name__ == "__main__":
    main()
