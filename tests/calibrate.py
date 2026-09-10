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
import hashlib
import itertools
import json
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

# A controlled cross-engine comparison has to score both engines on the SAME
# photos.  Each engine refuses a different set, so the caller runs once per
# engine to learn what each can embed, intersects the two lists, and passes the
# intersection back in here.
ONLY = {s for s in os.getenv("FACE_CALIB_ONLY", "").split(",") if s}


def identity_of(stem):
    prefix = re.sub(r"\d+$", "", stem)
    return ALIASES.get(prefix, prefix)


# --- embed every photo once --------------------------------------------------
# Byte-identical duplicates would contribute a genuine pair at distance zero and
# an extra template that adds nothing, flattering whichever engine is measured.
seen_hashes, duplicates = {}, []
paths = []
for path in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))):
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    first = seen_hashes.get(digest)
    if first:
        duplicates.append((os.path.basename(path), os.path.basename(first)))
        continue
    seen_hashes[digest] = path
    paths.append(path)

if duplicates:
    print(f"skipped {len(duplicates)} byte-identical duplicate photos:")
    for dup, original in duplicates:
        print(f"  {dup} is a copy of {original}")

embeddings, identity, quality, unusable = {}, {}, {}, []
for path in paths:
    stem = os.path.splitext(os.path.basename(path))[0]
    if ONLY and stem not in ONLY:
        continue
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


# --- scale-free metrics, so two engines can be compared ---------------------
# dlib measures Euclidean distance over 128 dimensions, ArcFace cosine distance
# over 512: the raw numbers are not comparable, and FAR/FRR are already zero on
# this dataset, so neither settles whether a new engine is better.  These
# metrics do - they read the whole distribution rather than one operating point.
print("")
print("=== scale-free metrics ===")


def roc_auc(gen, imp):
    """P(a random impostor scores farther than a random genuine pair).

    Mann-Whitney U, so no sklearn dependency.  1.0 is perfect separation,
    0.5 is chance.
    """
    merged = sorted([(v, 0) for v in gen] + [(v, 1) for v in imp])
    rank_sum, i = 0.0, 0
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2          # 1-based average rank for the tie group
        rank_sum += sum(avg_rank for k in range(i, j) if merged[k][1] == 0)
        i = j
    n_g, n_i = len(gen), len(imp)
    u = rank_sum - n_g * (n_g + 1) / 2      # U for the genuine class
    return 1.0 - u / (n_g * n_i)            # flip: genuine scores are the LOWER ones


def equal_error_rate(gen, imp):
    lo, hi = min(gen + imp), max(gen + imp)
    best = None
    for k in range(2001):
        t = lo + (hi - lo) * k / 2000
        frr = sum(1 for v in gen if v > t) / len(gen)
        far = sum(1 for v in imp if v <= t) / len(imp)
        if best is None or abs(frr - far) < abs(best[1] - best[2]):
            best = (t, frr, far)
    return best


def d_prime(gen, imp):
    mg, mi = statistics.fmean(gen), statistics.fmean(imp)
    vg = statistics.pvariance(gen) if len(gen) > 1 else 0.0
    vi = statistics.pvariance(imp) if len(imp) > 1 else 0.0
    spread = ((vg + vi) / 2) ** 0.5
    return (mi - mg) / spread if spread else float("inf")


def rank1(multi_template):
    """Leave-one-out: is the nearest template the right person?

    This is exactly what the 1:N cross-check tests, so it is the metric that
    tracks protection against buddy punching.
    """
    correct = total = 0
    for probe in names:
        cand_ids, cand_vecs = [], []
        for ident in idents:
            shots = [n for n in names if identity[n] == ident and n != probe]
            if not shots:
                continue
            for shot in (shots if multi_template else shots[:1]):
                cand_ids.append(ident)
                cand_vecs.append(embeddings[shot])
        if identity[probe] not in cand_ids:
            continue
        dists = engine.distances(np.stack(cand_vecs).astype(np.float32), embeddings[probe])
        total += 1
        if cand_ids[int(dists.argmin())] == identity[probe]:
            correct += 1
    return correct, total


auc = roc_auc(g, i)
eer_t, eer_frr, eer_far = equal_error_rate(g, i)
dp = d_prime(g, i)
r1_single = rank1(False)
r1_multi = rank1(True)
thin = [r for r in rows if r[0] < 0.10]

print(f"  engine                       {engine.ENGINE_ID} ({engine.EMBEDDING_DIM}-d)")
print(f"  ROC AUC                      {auc:.4f}       (1.0 = perfect, 0.5 = chance)")
print(f"  EER                          {(eer_frr+eer_far)/2*100:.2f}%      at distance {eer_t:.3f}")
print(f"  d-prime                      {dp:.2f}")
print(f"  rank-1, single template      {r1_single[0]}/{r1_single[1]} = {r1_single[0]/r1_single[1]*100:.1f}%")
print(f"  rank-1, all templates        {r1_multi[0]}/{r1_multi[1]} = {r1_multi[0]/r1_multi[1]*100:.1f}%")
print(f"  identities with margin <0.10 {len(thin)}/{len(rows)}  ({', '.join(r[1] for r in thin)})")
print(f"  worst-case margin            {min(r[0] for r in rows):+.3f}")

baseline = {
    "engine_id": engine.ENGINE_ID,
    "embedding_dim": engine.EMBEDDING_DIM,
    "identities": len(idents),
    "photos_usable": len(names),
    "photos_refused": len(unusable),
    "photos_duplicate": len(duplicates),
    "genuine_pairs": len(g),
    "impostor_pairs": len(i),
    "roc_auc": round(auc, 4),
    "eer_pct": round((eer_frr + eer_far) / 2 * 100, 3),
    "eer_threshold": round(eer_t, 4),
    "d_prime": round(dp, 3),
    "rank1_single_pct": round(r1_single[0] / r1_single[1] * 100, 2),
    "rank1_multi_pct": round(r1_multi[0] / r1_multi[1] * 100, 2),
    "genuine_worst": round(g[-1], 4),
    "impostor_closest": round(i[0], 4),
    "separation_gap": round(i[0] - g[-1], 4),
    "worst_margin": round(min(r[0] for r in rows), 4),
    "thin_margin_identities": sorted(r[1] for r in thin),
    "photos": names,
}
HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, f"metrics_{engine.ENGINE_ID}.json")
with open(out, "w") as fh:
    json.dump(baseline, fh, indent=2, sort_keys=True)
print("")
print(f"  baseline written to {out}")
print("  freeze it as baseline_<engine>.json to compare future runs against")


# --- comparison against frozen baselines ------------------------------------
# Each engine's accepted numbers live in tests/baseline_<engine_id>.json.
# Comparing against the file rather than a remembered figure is the point:
# distances are not comparable across engines, so only the scale-free metrics
# below carry meaning in a cross-engine row.
SCALE_FREE = [
    ("roc_auc", "ROC AUC", True),
    ("eer_pct", "EER %", False),
    ("d_prime", "d-prime", True),
    ("rank1_single_pct", "rank-1 single %", True),
    ("rank1_multi_pct", "rank-1 multi %", True),
]
SCALE_BOUND = [
    ("separation_gap", "separation gap", True),
    ("worst_margin", "worst margin", True),
    ("genuine_worst", "worst genuine dist", False),
    ("impostor_closest", "closest impostor dist", True),
]

frozen = {}
for path in sorted(glob.glob(os.path.join(HERE, "baseline_*.json"))):
    with open(path) as fh:
        data = json.load(fh)
    frozen[data["engine_id"]] = data

print("")
if not frozen:
    print("=== no frozen baselines yet ===")
    print(f"  freeze this run:  cp {out} {os.path.join(HERE, 'baseline_' + engine.ENGINE_ID + '.json')}")
else:
    own = frozen.get(engine.ENGINE_ID)
    if own:
        print(f"=== regression check against baseline_{engine.ENGINE_ID}.json ===")
        drift = []
        for key, label, higher_better in SCALE_FREE + SCALE_BOUND:
            was, now = own.get(key), baseline.get(key)
            if was is None or now is None:
                continue
            delta = now - was
            if abs(delta) > 1e-6:
                drift.append((label, was, now, delta))
        if drift:
            print("  metrics moved since the baseline was frozen:")
            for label, was, now, delta in drift:
                print(f"    {label:<24}{was:>10.4f} -> {now:>10.4f}   {delta:+.4f}")
            print("  a refactor that changes nothing should show no drift here")
        else:
            print("  identical to the frozen baseline - no behavioural drift")

    others = {k: v for k, v in frozen.items() if k != engine.ENGINE_ID}
    for other_id, other in others.items():
        print("")
        print(f"=== {engine.ENGINE_ID} vs {other_id} ===")
        print(f"  {'metric':<24}{other_id[:14]:>15}{engine.ENGINE_ID[:14]:>15}   verdict")
        for key, label, higher_better in SCALE_FREE:
            a, b = other.get(key), baseline.get(key)
            if a is None or b is None:
                continue
            delta = b - a
            better = (delta > 0) if higher_better else (delta < 0)
            verdict = "better" if (abs(delta) > 1e-9 and better) else "worse" if abs(delta) > 1e-9 else "same"
            print(f"  {label:<24}{a:>15.4f}{b:>15.4f}   {verdict}")
        print(f"  {'-- distance-scale metrics below are NOT comparable across engines --':<40}")
        for key, label, higher_better in SCALE_BOUND:
            a, b = other.get(key), baseline.get(key)
            if a is None or b is None:
                continue
            print(f"  {label:<24}{a:>15.4f}{b:>15.4f}")
        a_thin = set(other.get("thin_margin_identities", []))
        b_thin = set(baseline.get("thin_margin_identities", []))
        print(f"  {'thin-margin identities':<24}{','.join(sorted(a_thin)) or '-':>15}"
              f"{','.join(sorted(b_thin)) or '-':>15}")
        fixed, broke = a_thin - b_thin, b_thin - a_thin
        if fixed:
            print(f"    no longer thin: {', '.join(sorted(fixed))}")
        if broke:
            print(f"    newly thin:     {', '.join(sorted(broke))}")
