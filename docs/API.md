# Face verification API

For the team integrating the HRIS with this service. Every field below was read
off the handlers in `main.py`, `matcher.py` and `engines/common.py` rather than
described from intent.

## What the HRIS has to do

Three calls, in this order:

1. **`POST /enroll`** once per employee, when their profile photo is set or
   changed. Send several photos if you have them.
2. **`POST /verify`** at every clock-in, with the attendance photo and the
   employee id being claimed.
3. **`DELETE /enroll/{tenant_id}/{employee_id}`** when an employee leaves.

`/verify` is the endpoint that actually protects attendance. It scores the photo
against **every** enrolled employee in the tenant, not only the claimed one, so
a lookalike colleague fails even when a straight two-photo comparison would have
passed. `POST /compare-fr` does not do this and is being retired.

## Before you start

**There is no authentication on any endpoint.** Anyone who can reach the port
can enrol their own face as any `employee_id`, or delete anyone's templates.
This is a deliberate deployment decision: the service must only be reachable
from the HRIS, never from a user-facing network. If that ever stops being true,
the lack of auth becomes the whole attack.

Images are **base64**, with or without a `data:image/jpeg;base64,` prefix. JPEG
and PNG both work. Anything wider or taller than 1600px is downscaled
server-side, so sending originals only costs bandwidth.

Every response has the same envelope:

```json
{ "status": "success" | "error", "message": "human text", "data": { ... } }
```

Errors carry `reason` (a stable machine code — branch on this, never on
`message`) and `errors` (detail for logs).

## `POST /enroll`

```json
{
  "employee_id": "EMP-1042",
  "images": ["<base64>", "<base64>", "<base64>"],
  "tenant_id": "default",
  "replace": true
}
```

`tenant_id` defaults to `"default"`, `replace` to `true` (drops that employee's
existing templates first; send `false` to add to them).

**Send more than one photo.** Measured on 30 employees, going from one template
to several moved the worst genuine distance from 0.368 to 0.326 and the worst
margin from 0.274 to 0.288. At most 5 are kept per employee. Photos taken at
different times are what help — near-identical copies spend a slot without
adding a pose or a lighting condition.

```json
{
  "status": "success",
  "data": {
    "tenant_id": "default",
    "employee_id": "EMP-1042",
    "templates_stored": 3,
    "rejected": [{ "index": 2, "reason": "low_quality_blur", "detail": "..." }],
    "engine_id": "insightface-buffalo-m"
  }
}
```

`200` with a non-empty `rejected` array means **some photos were unusable and
the rest were stored**. Check `templates_stored`, and surface `rejected` to
whoever uploaded the photo so they can replace it. `422` means none were usable.

## `POST /verify`

```json
{
  "employee_id": "EMP-1042",
  "image": "<base64>",
  "tenant_id": "default",
  "reference_image": "<base64>"
}
```

`reference_image` is optional and exists only for migration: if the employee has
no template yet, that photo is enrolled on the spot. Stop sending it once
everyone is enrolled.

**Templates also refresh themselves, quietly.** When a clock-in scores a clean
accept — no bystander in frame, no borderline distance, and a margin clearly
above the minimum, not just enough to avoid `review` — that photo is folded
into the employee's rolling template set (still capped at 5, oldest dropped),
but no more often than once every `FACE_AUTO_UPDATE_INTERVAL_DAYS` (30 by
default). This never affects the response — a skipped or failed refresh looks
identical to a normal accept. It exists because a template frozen at enrolment
time drifts away from a real face over months (haircut, glasses, weight), and
nothing else in this API keeps it current. Turn it off with
`FACE_AUTO_UPDATE=false` if that trade-off is not wanted.

```json
{
  "status": "success",
  "data": {
    "decision": "accept",
    "match": true,
    "requires_review": false,
    "distance": 0.2134,
    "runner_up_distance": 0.7781,
    "margin": 0.5647,
    "candidates_checked": 150,
    "reasons": [],
    "quality": { "faces_found": 1, "extra_faces": 0, "face_pixels": 214,
                 "blur_variance": 88.4, "brightness": 131.2 },
    "liveness": { "score": 0.9312, "is_live": true }
  }
}
```

`distance` is to the claimed employee and **lower means more similar**.
`runner_up_distance` is to the closest *other* employee; `margin` is the
difference. The runner-up's identity is deliberately withheld — you need the
score to judge confidence, not the name of the person someone was mistaken for.

`liveness` is `null` when liveness is disabled or uncalibrated, which means
**not assessed**, not "passed".

### Reading the decision

| `decision` | `match` | `requires_review` | What it means |
| --- | --- | --- | --- |
| `accept` | `true` | `false` | Record attendance. |
| `review` | `true` | `true` | Close enough to record, far enough to log. |
| `reject` | `false` | `false` | Do not record attendance. |

**`match` is `true` for `review` as well as `accept`.** A `review` is recorded
attendance that a human should look at later — the usual handling is to record
it and flag the row, not to block the employee at the door.

Treat `review` as a signal, not noise. On every dataset measured so far the
review band was **empty**: the thresholds sit inside a gap where nothing landed.
So a `review` in production is a face harder than anything the calibration has
seen, and a rising review rate is the earliest warning that the roster has
outgrown its thresholds. `tools/roster_audit.py` reads these back out.

### HTTP statuses

| Status | When | HRIS should |
| --- | --- | --- |
| `200` | A decision was reached, including `reject` | Read `data.decision` |
| `409` | `not_enrolled` — no templates for this employee | Call `/enroll` first; do not treat as a failed match |
| `422` | The photo is unusable, or a spoof was detected | Ask for a retake |
| `503` | `liveness_unavailable` | **Do not record attendance.** Alert operations |

`503` is deliberate: when liveness is required but its weights are missing, the
service refuses rather than quietly verifying identity without a spoof check. A
retry will not fix it — it needs someone to look at the deployment.

## `GET /enroll/{tenant_id}/{employee_id}`

```json
{ "status": "success",
  "data": { "enrolled": true, "templates_stored": 3,
            "engine_id": "insightface-buffalo-m" } }
```

## `DELETE /enroll/{tenant_id}/{employee_id}`

```json
{ "status": "success", "data": { "templates_removed": 3 } }
```

Face templates are biometric data, which is sensitive personal data under UU
PDP. **Call this when an employee leaves or is deleted in the HRIS.** Nothing
else expires them.

## `GET /health`

Returns store counts, liveness state, the active thresholds,
`max_templates_per_employee` — the cap on how many photos per employee are
kept (sending more to `/enroll` is not an error; the oldest are discarded) —
and `auto_update`: whether templates refresh themselves from clean clock-ins,
the minimum days between refreshes, and the margin a decision needs to clear
before it counts as clean enough to trust. Check this after any deploy that
changes auto-update's env vars — it is the only way to confirm the setting
that actually took effect, since it never shows up in a response body.

Two fields are worth alerting on: `liveness.available` going `false`, and
`store.stale_templates` being non-zero — that means templates exist from a
different model version and those employees cannot clock in until re-enrolled.

## `POST /compare-fr` — deprecated

The original two-image endpoint. Still answers, still has the same response
shape, and should not be built against: it compares two photos with no roster
behind it, so it cannot tell a lookalike colleague from the right person, and it
runs **no liveness check**. Migrate to `/verify`.

## Reason codes

Branch on these, not on `message`.

**The photo needs retaking** — `422` on `/verify`, or listed in `rejected` on
`/enroll`:

| Code | Meaning |
| --- | --- |
| `no_face` | No face found, or the detected box fell outside the image |
| `low_confidence` | A face was found but the detector was not sure enough |
| `face_too_small` | Face narrower than the minimum width in pixels |
| `low_quality_blur` | Out of focus |
| `low_quality_dark` / `low_quality_bright` | Under- or over-exposed |
| `invalid_image` | The payload did not decode to a readable image. `errors` names the cause: whitespace where `+` should be (form encoding), URL-safe `-`/`_`, base64 applied twice, HEIC from an iPhone, or a file that is not an image at all |
| `too_many_faces` | More people in frame than allowed |
| `ambiguous_subject` | Two faces of similar size — who is presenting is unclear |
| `multiple_faces` | More than one face, when bystanders are disallowed entirely |

**Liveness** — the photo looks like a photo of a photo:

| Code | Status | Meaning |
| --- | --- | --- |
| `spoof_suspected` | `422` | Screen, print or mask suspected |
| `liveness_unavailable` | `503` | Liveness required but not loadable |

**Identity** — appear in `data.reasons` on a `200`:

| Code | Meaning |
| --- | --- |
| `borderline_distance` | In the review band |
| `below_threshold` | Too far from the claimed employee |
| `identity_mismatch` | Someone else is a closer match than the claimed employee |
| `low_margin` | Claimed employee won, but not by enough — downgraded to review |
| `extra_faces_present` | Bystanders in frame; the subject still verified |
| `claimed_face_is_secondary` | The claimed employee is in frame but is **not** the person presenting — someone holding up a photo of them, typically |
| `not_enrolled` | No templates for this employee (`409`) |

**Internal** — a bug or a broken deployment, not the employee's fault:
`model_missing`, `alignment_failed`, `encoding_failed`.

A reason prefixed `reference_` (e.g. `reference_no_face`) refers to the
`reference_image` you sent, not the attendance photo.

## Things that will bite you

**Thresholds are not portable.** The numbers in `/health` belong to one specific
model pairing. `engine_id` names that pairing, and templates are stored against
it, so changing the model makes every stored template invisible rather than
wrongly comparable. If `engine_id` changes, **everyone must be re-enrolled** —
there is no migration.

**The safety margin shrinks as you hire.** Separation is a minimum over every
pair of employees, so each new hire is another chance that two people land
closer than anyone measured. Run `tools/roster_audit.py` after each enrolment
round; it names the two closest employees and says how much room is left.

**`review` is not an error.** Handling it as one will reject legitimate
employees and throw away the only early-warning signal the system has.
