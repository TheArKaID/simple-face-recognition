"""Calibration harness: measures where the thresholds should sit.

Run it inside the container, where dlib lives:

    docker build -t prime-face-recognizer:test .
    docker run -d --name fr -p 8001:8000 prime-face-recognizer:test
    docker exec fr python /app/tests/calibrate.py

Photos come from tests/images, named <identity><n>.jpg; files sharing a prefix
are the same person, with ALIASES recording any second name for one person.
Getting a label wrong inverts the result, so check the "closest impostor pairs"
list it prints - a suspiciously close pair usually means a mislabel rather than
a model failure.

Embedding every photo once and doing the arithmetic in numpy replaces ~820
HTTP round trips (each of which would re-embed both images) with 41 embeddings,
turning half an hour into under a minute.

Three things come out of this:
  1. the genuine/impostor distance distributions and whether they separate
  2. a threshold sweep, so the accept/review bands are chosen from data
  3. what the 1:N cross-check actually buys, by running the same decisions
     with it switched off
"""
import glob
import itertools
import os
import re
import statistics
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import engine
import matcher
from store import TenantIndex

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
# The repo owner confirmed z is the same person as y.
ALIASES = {"z": "y"}


def identity_of(stem):
    prefix = re.sub(r"\d+$", "", stem)
    return ALIASES.get(prefix, prefix)


# --- embed every photo once --------------------------------------------------
embeddings, identity, quality, unusable = {}, {}, {}, []
for path in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        result = engine.embed(Image.open(path))
        embeddings[stem] = result.embedding
        identity[stem] = identity_of(stem)
        quality[stem] = result.quality()
    except engine.FaceError as exc:
        unusable.append((stem, exc.reason, exc.detail))

names = sorted(embeddings)
idents = sorted(set(identity.values()))
print(f"photos usable: {len(names)}  identities: {len(idents)}")
print(f"identities: {', '.join(idents)}")
for ident in idents:
    shots = [n for n in names if identity[n] == ident]
    print(f"  {ident:<8} {len(shots)} photos: {', '.join(shots)}")

if unusable:
    print(f"\nphotos the quality gates refused ({len(unusable)}):")
    for stem, reason, detail in unusable:
        print(f"  {stem:<10} {reason:<20} {detail}")

print("")
print("=== per-photo quality (a generic embedding sits close to everyone) ===")
print("  photo      face_px   blur     bright   mean dist to others")
for n in names:
    others = [engine.distance(embeddings[n], embeddings[m]) for m in names
              if identity[m] != identity[n]]
    q = quality[n]
    print(f"  {n:<10} {q['face_pixels']:>6}  {q['blur_variance']:>8.1f}  "
          f"{q['brightness']:>6.1f}   {sum(others)/len(others):.3f}")

# --- pairwise distances ------------------------------------------------------
genuine, impostor = [], []
for a, b in itertools.combinations(names, 2):
    d = engine.distance(embeddings[a], embeddings[b])
    (genuine if identity[a] == identity[b] else impostor).append(((a, b), d))

g = sorted(d for _, d in genuine)
i = sorted(d for _, d in impostor)
print(f"\n=== 1:1 distance distributions ===")
print(f"genuine   n={len(g):<5} min {g[0]:.3f}  p50 {statistics.median(g):.3f}  "
      f"p95 {g[int(len(g)*0.95)]:.3f}  max {g[-1]:.3f}")
print(f"impostor  n={len(i):<5} min {i[0]:.3f}  p05 {i[int(len(i)*0.05)]:.3f}  "
      f"p50 {statistics.median(i):.3f}  max {i[-1]:.3f}")
print(f"separation: worst genuine {g[-1]:.3f} | closest impostor {i[0]:.3f} | gap {i[0]-g[-1]:+.3f}")

print("\nworst genuine pairs (candidates for a bad enrollment photo):")
for (a, b), d in sorted(genuine, key=lambda t: -t[1])[:8]:
    print(f"  {d:.3f}  {a} vs {b}")
print("\nclosest impostor pairs (if any of these are the same person, the labels are wrong):")
for (a, b), d in sorted(impostor, key=lambda t: t[1])[:10]:
    print(f"  {d:.3f}  {a} vs {b}")

# --- threshold sweep on the 1:1 decision ------------------------------------
print("\n=== 1:1 threshold sweep ===")
print("  thresh    FRR (genuine rejected)   FAR (impostor accepted)")
for t in [x / 100 for x in range(35, 71, 5)]:
    frr = sum(1 for d in g if d > t) / len(g)
    far = sum(1 for d in i if d <= t) / len(i)
    print(f"  {t:.2f}      {frr*100:5.1f}%  ({sum(1 for d in g if d > t)}/{len(g)})"
          f"            {far*100:5.2f}%  ({sum(1 for d in i if d <= t)}/{len(i)})")

# --- full system: enrol photo 1, probe with the rest ------------------------
def build_index(template_of):
    ids = np.array([identity_of(s) for s in template_of])
    mat = np.stack([embeddings[s] for s in template_of]).astype(np.float32)
    return TenantIndex(employee_ids=ids, matrix=mat, version=1)


enrolled_shot = {}
for ident in idents:
    shots = [n for n in names if identity[n] == ident]
    enrolled_shot[ident] = shots[0]
index = build_index([enrolled_shot[x] for x in idents])
probes = [n for n in names if n not in set(enrolled_shot.values())]

print(f"\n=== full system: {len(idents)} enrolled, {len(probes)} probe photos ===")


def evaluate(cross_check):
    original = config.CROSS_CHECK_ENABLED
    config.CROSS_CHECK_ENABLED = cross_check
    stats = {"genuine": {}, "impostor": {}}
    leaks = []
    try:
        for probe in probes:
            for claimed in idents:
                kind = "genuine" if identity[probe] == claimed else "impostor"
                decision = matcher.verify(index, claimed, embeddings[probe])
                stats[kind][decision.decision] = stats[kind].get(decision.decision, 0) + 1
                if kind == "impostor" and decision.match:
                    leaks.append((probe, claimed, decision.distance, decision.decision))
    finally:
        config.CROSS_CHECK_ENABLED = original
    return stats, leaks


for label, cc in (("cross-check ON", True), ("cross-check OFF", False)):
    stats, leaks = evaluate(cc)
    gen_total = sum(stats["genuine"].values())
    imp_total = sum(stats["impostor"].values())
    gen_ok = stats["genuine"].get("accept", 0) + stats["genuine"].get("review", 0)
    print(f"\n  {label}")
    print(f"    genuine  {gen_total:>4} attempts: " +
          ", ".join(f"{k} {v}" for k, v in sorted(stats['genuine'].items())) +
          f"   -> false reject {(gen_total-gen_ok)/gen_total*100:.1f}%")
    print(f"    impostor {imp_total:>4} attempts: " +
          ", ".join(f"{k} {v}" for k, v in sorted(stats['impostor'].items())) +
          f"   -> false accept {len(leaks)/imp_total*100:.2f}%")
    if leaks:
        print(f"    impostor attempts that got through ({len(leaks)}):")
        for probe, claimed, dist, dec in leaks:
            print(f"      {probe} claiming {claimed}: distance {dist:.3f} -> {dec}")


# --- who could be confused with whom ----------------------------------------
# Ranked by the worst-case margin an identity achieves across its own probe
# photos: how much closer they are to their own template than to the nearest
# other person's.  A thin margin means one bad attendance photo away from a
# review, and it is the list to work through when deciding who should be
# re-enrolled with several photos.
print("")
print("=== nearest-neighbour risk map ===")
print("  identity   photos  worst self-dist  nearest other  margin   nearest is")
rows = []
for ident in idents:
    own = [n for n in names if identity[n] == ident]
    probe_shots = [n for n in own if n != enrolled_shot[ident]]
    if not probe_shots:
        continue
    worst_self = max(engine.distance(embeddings[p_], embeddings[enrolled_shot[ident]])
                     for p_ in probe_shots)
    best_other, best_other_id = min(
        (engine.distance(embeddings[p_], embeddings[enrolled_shot[o]]), o)
        for p_ in probe_shots for o in idents if o != ident)
    rows.append((best_other - worst_self, ident, len(own), worst_self, best_other, best_other_id))

for margin, ident, n_photos, worst_self, best_other, other_id in sorted(rows):
    flag = "  <-- thin" if margin < 0.10 else ""
    print(f"  {ident:<10} {n_photos:>5}   {worst_self:>13.3f}  {best_other:>13.3f}  "
          f"{margin:>+7.3f}   {other_id}{flag}")


# --- does multi-photo enrollment actually help? -----------------------------
# Leave-one-out: each photo takes a turn as the attendance probe while every
# identity is enrolled from its remaining photos.  "single" keeps just one
# template per person, "multi" keeps all of them - so the comparison isolates
# the effect of extra enrollment photos on the same probes.
print("")
print("=== single-photo vs multi-photo enrollment (leave-one-out) ===")


def run_mode(multi):
    genuine_dists, margins, reviews, rejects = [], [], 0, 0
    impostor_dists, false_accepts = [], 0
    for probe in names:
        template_ids, template_vecs = [], []
        for ident in idents:
            shots = [n for n in names if identity[n] == ident and n != probe]
            if not shots:
                continue
            keep = shots[:config.MAX_TEMPLATES_PER_EMPLOYEE] if multi else shots[:1]
            for shot in keep:
                template_ids.append(ident)
                template_vecs.append(embeddings[shot])
        if identity[probe] not in template_ids:
            continue
        index_lo = TenantIndex(employee_ids=np.array(template_ids),
                               matrix=np.stack(template_vecs).astype(np.float32),
                               version=1)
        for claimed in sorted(set(template_ids)):
            decision = matcher.verify(index_lo, claimed, embeddings[probe])
            if claimed == identity[probe]:
                genuine_dists.append(decision.distance)
                if decision.margin is not None:
                    margins.append(decision.margin)
                if decision.decision == "review":
                    reviews += 1
                elif decision.decision == "reject":
                    rejects += 1
            else:
                impostor_dists.append(decision.distance)
                if decision.match:
                    false_accepts += 1
    return {
        "genuine_n": len(genuine_dists),
        "genuine_worst": max(genuine_dists),
        "genuine_median": statistics.median(genuine_dists),
        "margin_worst": min(margins),
        "margin_median": statistics.median(margins),
        "reviews": reviews,
        "rejects": rejects,
        "impostor_n": len(impostor_dists),
        "impostor_closest": min(impostor_dists),
        "false_accepts": false_accepts,
    }


single, multi = run_mode(False), run_mode(True)
print(f"  {'metric':<34}{'single':>12}{'multi':>12}   change")
for label, key, better_lower in [
    ("genuine distance, worst", "genuine_worst", True),
    ("genuine distance, median", "genuine_median", True),
    ("margin, worst", "margin_worst", False),
    ("margin, median", "margin_median", False),
    ("closest impostor distance", "impostor_closest", False),
]:
    a, b = single[key], multi[key]
    delta = b - a
    good = (delta < 0) if better_lower else (delta > 0)
    print(f"  {label:<34}{a:>12.3f}{b:>12.3f}   {delta:+.3f} "
          f"{'better' if good else 'worse' if delta else 'same'}")
for label, key in [("genuine attempts", "genuine_n"),
                   ("  flagged for review", "reviews"),
                   ("  falsely rejected", "rejects"),
                   ("impostor attempts", "impostor_n"),
                   ("  falsely accepted", "false_accepts")]:
    print(f"  {label:<34}{single[key]:>12}{multi[key]:>12}")
