"""Measure whether the liveness model actually separates live faces from spoofs.

Run inside the container, as with tests/calibrate.py:

    docker exec <container> python /app/tools/measure_liveness.py

Expects:
  tests/images/            genuine photos  (whatever is already there)
  tests/images/spoof/      presentation attacks - a phone or laptop screen
                           showing one of those faces, or a printed copy

Without the spoof folder this prints what it needs and stops. That refusal is
the point: a liveness threshold guessed rather than measured is worse than none
at all, because it is trusted. The numbers this prints - AUC, EER, and the
score distributions - are what FACE_LIVENESS_MIN_SCORE should be set from.
"""
import glob
import os
import statistics
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

# Measuring the model is this script's whole job, so force it on regardless of
# how the service is currently configured.
config.LIVENESS_MODE = "model"

import engine    # noqa: E402
import liveness  # noqa: E402

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "tests", "images")
SPOOF = os.path.join(IMAGES, "spoof")

print(f"engine  {engine.ENGINE_ID}")
print(f"weights {config.LIVENESS_MODEL_DIR}")
files = sorted(glob.glob(os.path.join(config.LIVENESS_MODEL_DIR, "*.onnx")))
for f in files:
    print(f"  {os.path.basename(f)}  {os.path.getsize(f)/1024:.0f} KB")
if not liveness.available():
    print("\nNo usable weights. Run tools/convert_minifas.py first.")
    sys.exit(2)

spoof_files = sorted(glob.glob(os.path.join(SPOOF, "*.jpg")) +
                     glob.glob(os.path.join(SPOOF, "*.png")))
if not spoof_files:
    print(f"""
No spoof samples in {SPOOF}

Nothing here can be validated without them, so this script stops rather than
printing numbers that look like evidence.  To produce them, 10 minutes:

  1. Open one of the existing faces (say tests/images/anggit1.jpg) full-screen
     on a phone or laptop.
  2. Photograph that screen with another phone, roughly as an employee would
     hold it for attendance.
  3. Repeat for 5-10 different faces, and vary it a little - different screen
     brightness, a slight angle, one printed copy if a printer is handy.
  4. Save as tests/images/spoof/spoof_<identity>_<n>.jpg

Then run this again.""")
    sys.exit(2)


def score_of(path):
    """Liveness score for the largest face in `path`, or None if unusable."""
    try:
        result = engine.embed(Image.open(path), quality_gates=False)
    except engine.FaceError as exc:
        return None, exc.reason
    live = liveness.check(result.image, result.bbox)
    if live is None:
        return None, "not_assessed"
    return live.score, None


def collect(paths, label):
    scores, skipped = [], []
    for path in paths:
        score, why = score_of(path)
        if score is None:
            skipped.append((os.path.basename(path), why))
        else:
            scores.append((os.path.basename(path), score))
    print(f"\n{label}: {len(scores)} scored, {len(skipped)} skipped")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    return scores


genuine_files = sorted(glob.glob(os.path.join(IMAGES, "*.jpg")))
genuine = collect(genuine_files, "genuine")
spoof = collect(spoof_files, "spoof")

if not genuine or not spoof:
    print("\nNeed both classes scored to say anything.")
    sys.exit(2)

g = sorted(s for _, s in genuine)
s_ = sorted(s for _, s in spoof)
print(f"\ngenuine  min {g[0]:.4f}  p05 {g[max(0,int(len(g)*0.05))]:.4f}  "
      f"p50 {statistics.median(g):.4f}  max {g[-1]:.4f}")
print(f"spoof    min {s_[0]:.4f}  p50 {statistics.median(s_):.4f}  "
      f"p95 {s_[min(len(s_)-1,int(len(s_)*0.95))]:.4f}  max {s_[-1]:.4f}")
print(f"separation: worst genuine {g[0]:.4f} | best spoof {s_[-1]:.4f} | "
      f"gap {g[0]-s_[-1]:+.4f}")

# Higher score means live, so the comparison runs the other way from distances.
pairs = [(v, 1) for v in g] + [(v, 0) for v in s_]
concordant = sum(1 for a in g for b in s_ if a > b) + 0.5 * sum(1 for a in g for b in s_ if a == b)
auc = concordant / (len(g) * len(s_))
print(f"\nROC AUC {auc:.4f}   (1.0 = perfect, 0.5 = chance)")

best = None
for k in range(1001):
    t = k / 1000.0
    frr = sum(1 for v in g if v < t) / len(g)      # live faces refused
    far = sum(1 for v in s_ if v >= t) / len(s_)   # spoofs accepted
    if best is None or abs(frr - far) < abs(best[1] - best[2]):
        best = (t, frr, far)
print(f"EER {(best[1]+best[2])/2*100:.2f}%  at threshold {best[0]:.3f}")

print("\nthreshold sweep:")
print("  thresh   live refused        spoofs accepted")
for k in range(1, 20):
    t = k / 20.0
    frr = sum(1 for v in g if v < t)
    far = sum(1 for v in s_ if v >= t)
    print(f"  {t:.2f}     {frr/len(g)*100:5.1f}% ({frr}/{len(g)})"
          f"        {far/len(s_)*100:5.1f}% ({far}/{len(s_)})")

print("\nworst genuine (would be refused first):")
for name, score in sorted(genuine, key=lambda t: t[1])[:5]:
    print(f"  {score:.4f}  {name}")
print("best spoof (would slip through first):")
for name, score in sorted(spoof, key=lambda t: -t[1])[:5]:
    print(f"  {score:.4f}  {name}")

print(f"""
Set FACE_LIVENESS_MIN_SCORE from the sweep above, not from the EER: refusing a
live employee costs a retake, while accepting a spoof records false attendance,
so the two errors are not worth the same.  Then set FACE_LIVENESS_MODE=model.""")
