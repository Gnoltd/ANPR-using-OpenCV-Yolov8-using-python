# Plate recognition accuracy improvements

## Context

`anpr_eval.py` (see README "Evaluation" section) was used to measure real
accuracy against two labeled image sets:

- `eval_images/`: 2 Vietnamese-format demo photos + 25 US plates from
  OpenALPR's public `benchmarks` repo.
- `eval_images_vn/`: 6 real Vietnamese photos (11 plates, hand-labeled by
  visual inspection), 5 sourced from `trungdinh22/License-Plate-Recognition`.

The Vietnamese-only run scored Detection Rate 72.7%, Recognition Accuracy
9.1%, Precision 12.5%, Recall 9.1%, F1 10.5% (TP=1, FP=7, FN=10). Inspecting
`preds_vn.csv` against `gt_vn.csv` showed the failures cluster into three
distinct, independent causes:

1. `filter_text`/`canonicalize_plate` in `DetectNP.py` only recognize the
   car-plate format (`NNL-NNN.NN`, single letter series, e.g. `18A-123.45`).
   Real motorbike plates commonly use a letter+digit series (`B1`, `V7`,
   `L1`, `U2`, `S1`, `V5`, e.g. `29-B1 256.62`), which falls through to a
   permissive "strip punctuation, keep if ≥5 chars" fallback that mangles
   the OCR'd text (`29B1-256.62` → `298I25662`).
2. Several near-miss predictions are single-character OCR confusions on
   visually similar glyphs: `30A-339.18` → `80A-339.18` (`3`/`8`),
   `29V5-2108` → `29I-521.08` (`V`/`I`).
3. In photos with 3 plates in frame, YOLO found only 1-2 boxes — it's
   missing the smaller/farther plates, not misreading them.

## Goals

- Extend plate-format normalization to correctly format motorbike-style
  letter+digit series plates, so they don't fall into the mangling fallback.
- Where an OCR'd plate doesn't resolve to a known owner, try safe
  single-character confusable substitutions and use one if it uniquely
  resolves to a registered plate.
- Empirically check whether adjusting `CONF_THRES`/`ANPR_IMGSZ` recovers
  more of the missed detections in crowded photos, without materially
  increasing false positives, using the two existing labeled image sets.

## Non-goals

- Blind digit/letter correction without an external signal to arbitrate
  (e.g. auto-"fixing" `3`→`8` or vice versa with no registry match). Format
  validity alone can't tell which digit is correct — both readings are
  syntactically valid plates. This will not move the accuracy numbers on
  unregistered benchmark images, and that's expected, not a bug.
- Retraining or fine-tuning the YOLO model. Any detection-recall gains here
  come only from confidence/image-size config, not model changes.
- Building a general fuzzy-matching layer for the registry lookup beyond
  the single-substitution correction described below.

## Design

### 1. Motorbike-format regex support (`DetectNP.py`)

Add two compiled patterns alongside the existing `_RE_PLATE_FULL` /
`_RE_PLATE_COMP` (car format, single-letter series):

```python
_RE_PLATE_MOTO_DOT = re.compile(r"^([0-9]{2})([A-Z][0-9])-?([0-9]{3})\.?([0-9]{2})$")
_RE_PLATE_MOTO_COMPACT = re.compile(r"^([0-9]{2})([A-Z][0-9])-?([0-9]{4})$")
```

`filter_text` and `canonicalize_plate` try patterns in order: car-dotted,
car-compact, moto-dotted, moto-compact. Whichever matches determines the
output format:

- moto-dotted match → `{province}{series}-{body[:3]}.{body[3:]}` (e.g. `29B1-256.62`)
- moto-compact match → `{province}{series}-{body}` (e.g. `29U2-7914`)

The existing "single digit at series position implies OCR misread a
letter" repair (`digit_to_letter` map) stays as-is and runs before all
four pattern attempts, same as today.

This is pure string/regex logic — testable without YOLO/EasyOCR/torch, in
a new `test_detectnp.py` (mirrors how `test_anpr_eval.py` isolates pure
logic from the ML pipeline).

### 2. Registry-assisted confusable-character correction (`DetectNP.py`)

New function in `DetectNP.py`:

```python
def correct_against_registry(plate: str, registry_df: pd.DataFrame) -> str:
    """If `plate` has no registry match, try single-character confusable
    substitutions; if exactly one substitution matches a registered plate,
    return that corrected plate. Otherwise return `plate` unchanged."""
```

Confusable pairs (bidirectional): `3↔8`, `I↔V`, `O↔0`, `1↔I`, `S↔5` — the
set directly observed in the eval failures plus the existing adjacent
digit/letter pairs already used elsewhere in this file, kept intentionally
small rather than a generic OCR-confusion table.

Algorithm: canonicalize `plate`; if it already matches a registered
`plate_norm`, return unchanged. Otherwise, for each character position,
try each applicable substitution, canonicalize the result, and check
against the registry. Collect all *distinct* registered matches found this
way. If exactly one distinct match, return it (in its registry-canonical
form). If zero or more than one, return the original plate unchanged
(ambiguous corrections are not applied).

`lookup_owner()` calls this after its current exact-match lookup fails,
before giving up.

Testable with an in-memory `pd.DataFrame` registry fixture — no real CSV
or ML dependency needed.

### 3. Detection-threshold experiment (not a code-behavior change by default)

Using a throwaway script (not committed), re-run `anpr_eval.py predict` +
`eval` over `eval_images/` and `eval_images_vn/` with `CONF_THRES` lowered
(e.g. 0.25 → 0.15) and/or `ANPR_IMGSZ` raised (640 → 960), and compare
`detection_rate`/`fp` before vs. after in the report. Only if a setting
clearly improves detection rate without materially increasing FP count,
propose it as a new default in `env.py` (reported to the user for approval
before committing — this changes runtime behavior for the whole app, not
just eval tooling). If no configuration helps, report that finding instead
of forcing a change.

## Testing

- `test_detectnp.py` (new): unit tests for `filter_text`, `canonicalize_plate`
  covering car format (regression), motorbike-dotted, motorbike-compact,
  and the digit-at-series-position repair, using real strings observed in
  the eval data (`29B1-256.62`, `29U2-7914`, etc.).
- Same file: unit tests for `correct_against_registry` with a fixture
  registry DataFrame, covering: exact match (no correction needed), single
  unambiguous correction, zero corrections found, and ambiguous (>1 match)
  correction (no change).
- After implementing #1, re-run `anpr_eval.py eval` against the existing
  `preds_vn.csv`/`gt_vn.csv` — expect `mean_cer` to drop for the
  motorbike-format entries even though several will still not be exact
  matches (the digit confusions from #2 are registry-scoped, not fixed for
  these unregistered benchmark plates).
- #3 is validated by its own before/after eval report, not unit tests.
