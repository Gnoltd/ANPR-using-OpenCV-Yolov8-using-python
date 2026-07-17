# Plate Recognition Accuracy Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `DetectNP.py`'s plate-format recognition and OCR candidate selection based on findings from evaluating the ANPR pipeline against real Vietnamese and US benchmark photos (see `docs/superpowers/specs/2026-07-17-plate-recognition-accuracy-design.md`).

**Architecture:** Four independent changes to `DetectNP.py`, each unit-tested in a new `test_detectnp.py` without needing YOLO/EasyOCR inference (only `re`/`pandas` string and DataFrame logic), plus one empirical (non-unit-tested) threshold experiment using the existing `anpr_eval.py` tooling.

**Tech Stack:** Python stdlib `re`, `unittest`, `pandas` (already a project dependency). No new dependencies.

## Global Constraints

- No new third-party dependencies — use only `re`, `pandas`, `unittest`, `csv`, `os`, `sys`, `types` (all either stdlib or already in `requirements.txt`).
- TDD required for every code task: write failing test → run it and confirm it fails for the right reason → implement → run and confirm it passes → commit. (Established project workflow — see prior commits in this session.)
- Existing car-plate format behavior (`18A-123.45` ↔ `18A12345`) must not regress. Every task touching `filter_text`/`canonicalize_plate` includes a regression test asserting car-format behavior is unchanged.
- `DetectNP.py` can only be imported after registering the `ANPR_Yolo` package name in `sys.modules` (see Task 1, Step 1) — this mirrors the existing bootstrap in `Run.py` and `anpr_eval.py`, not a new pattern.
- Task 4 (threshold tuning) must not change `env.py` defaults without the user reviewing the results table first — produce the report and stop; do not edit `env.py` in this task.

---

## File Structure

- **Modify `DetectNP.py`**: add motorbike-format regex patterns and a shared `_match_plate_format` helper (Task 1); add `strict` parameter to `filter_text` and a new `select_plate_text` function, rewire `ocr_it` to use it (Task 2); add `correct_against_registry` and wire it into `lookup_owner` (Task 3). All edits are additive or narrowly scoped replacements of existing functions — no restructuring of the file's existing flat-function layout, consistent with the rest of this codebase.
- **Create `test_detectnp.py`**: pure-logic unit tests, built up across Tasks 1–3 (each task appends its own test classes). No YOLO/EasyOCR calls — only imports `DetectNP` (which itself imports `Load.py`, which imports `ultralytics`/`easyocr` at module level, but never instantiates or runs them at import time).
- **Create `docs/superpowers/notes/2026-07-17-threshold-experiment-results.md`** (Task 4): the committed deliverable of the threshold experiment — a results table, not code.
- **Scratchpad-only, not committed**: `threshold_experiment.py` (Task 4) — a throwaway driver script, per the spec's explicit "not committed" instruction.

---

### Task 1: Motorbike-format regex support

**Files:**
- Modify: `DetectNP.py` (regex constants ~line 16-20, `filter_text` ~line 100-134, `canonicalize_plate` ~line 378-406)
- Test: `test_detectnp.py` (new)

**Interfaces:**
- Produces: `filter_text(text)` (unchanged signature for this task), `canonicalize_plate(s)` (unchanged signature), both now recognizing motorbike-format plates (`29B1-256.62`, `29U2-7914`) in addition to the existing car format (`18A-123.45`). Also produces a new module-level helper `_match_plate_format(t)` that Tasks 2 and later reuse implicitly (it's called internally by both `filter_text` and `canonicalize_plate`, not exported/used directly by other tasks).

- [ ] **Step 1: Write the failing tests**

Create `test_detectnp.py`:

```python
import os
import sys
import types


def _register_anpr_yolo_package():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if "ANPR_Yolo" not in sys.modules:
        pkg = types.ModuleType("ANPR_Yolo")
        pkg.__path__ = [here]
        pkg.__file__ = os.path.join(here, "__init__.py")
        pkg.__package__ = "ANPR_Yolo"
        sys.modules["ANPR_Yolo"] = pkg


_register_anpr_yolo_package()

import unittest
from DetectNP import filter_text, canonicalize_plate


class FilterTextCarFormatTests(unittest.TestCase):
    def test_dotted_car_plate_passthrough(self):
        self.assertEqual(filter_text("18A-123.45"), "18A-123.45")

    def test_compact_car_plate_gets_dash_and_dot(self):
        self.assertEqual(filter_text("18A12345"), "18A-123.45")


class FilterTextMotorbikeFormatTests(unittest.TestCase):
    def test_dotted_motorbike_plate_passthrough(self):
        self.assertEqual(filter_text("29B1-256.62"), "29B1-256.62")

    def test_compact_dotted_motorbike_plate_gets_formatted(self):
        self.assertEqual(filter_text("29B125662"), "29B1-256.62")

    def test_motorbike_plate_no_dot_four_digit_body(self):
        self.assertEqual(filter_text("29U2-7914"), "29U2-7914")

    def test_real_eval_sample_v7(self):
        self.assertEqual(filter_text("29V7-376.94"), "29V7-376.94")


class CanonicalizePlateMotorbikeFormatTests(unittest.TestCase):
    def test_dotted_motorbike_plate(self):
        self.assertEqual(canonicalize_plate("29B1-256.62"), "29B1-256.62")

    def test_compact_motorbike_plate_no_dot(self):
        self.assertEqual(canonicalize_plate("29U2-7914"), "29U2-7914")

    def test_car_plate_regression(self):
        self.assertEqual(canonicalize_plate("18A-123.45"), "18A-123.45")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify the motorbike ones fail**

Run: `.venv/Scripts/python.exe -m unittest test_detectnp -v`

Expected: `FilterTextCarFormatTests` tests PASS (existing behavior already supports car format). `FilterTextMotorbikeFormatTests` and `CanonicalizePlateMotorbikeFormatTests` FAIL — e.g. `test_dotted_motorbike_plate_passthrough` fails because today's fallback returns `"29B125662"` (punctuation stripped, no reformatting) instead of `"29B1-256.62"`.

- [ ] **Step 3: Implement motorbike regex support**

In `DetectNP.py`, replace:

```python
# Compiled plate-format patterns (reused on every frame in video mode)
_RE_PLATE_FULL  = re.compile(r"^([0-9]{2})([A-Z])-?([0-9]{3})\.?([0-9]{2})$")
_RE_PLATE_COMP  = re.compile(r"^([0-9]{2})([A-Z])([0-9]{5})$")
_RE_PLATE_CLEAN = re.compile(r"[^A-Z0-9\-\.]")
_RE_NON_ALNUM   = re.compile(r"[^A-Z0-9]")
```

with:

```python
# Compiled plate-format patterns (reused on every frame in video mode)
_RE_PLATE_FULL  = re.compile(r"^([0-9]{2})([A-Z])-?([0-9]{3})\.?([0-9]{2})$")
_RE_PLATE_COMP  = re.compile(r"^([0-9]{2})([A-Z])([0-9]{5})$")
# Motorbike-style series (letter+digit, e.g. B1/V7/L1/U2/S1/V5), dotted or 4-digit-no-dot body
_RE_PLATE_MOTO_DOT     = re.compile(r"^([0-9]{2})([A-Z][0-9])-?([0-9]{3})\.?([0-9]{2})$")
_RE_PLATE_MOTO_COMPACT = re.compile(r"^([0-9]{2})([A-Z][0-9])-?([0-9]{4})$")
_RE_PLATE_CLEAN = re.compile(r"[^A-Z0-9\-\.]")
_RE_NON_ALNUM   = re.compile(r"[^A-Z0-9]")


def _format_full(m):
    return f"{m.group(1)}{m.group(2)}-{m.group(3)}.{m.group(4)}"


def _format_comp(m):
    s = m.group(3)
    return f"{m.group(1)}{m.group(2)}-{s[:3]}.{s[3:]}"


def _format_moto_compact(m):
    return f"{m.group(1)}{m.group(2)}-{m.group(3)}"


# Tried in order: car-dotted, car-compact, moto-dotted, moto-compact. Car
# patterns are tried first so an ambiguous fully-compact string (no dash,
# e.g. could be read as 1-letter+5-digit OR 2-char-series+4-digit) resolves
# to the car interpretation, which is the more common format.
_PLATE_PATTERNS = (
    (_RE_PLATE_FULL, _format_full),
    (_RE_PLATE_COMP, _format_comp),
    (_RE_PLATE_MOTO_DOT, _format_full),
    (_RE_PLATE_MOTO_COMPACT, _format_moto_compact),
)


def _match_plate_format(t):
    """Try each known plate pattern against t; return the formatted plate
    string (e.g. "18A-123.45" or "29B1-256.62") or None if none match."""
    for pat, formatter in _PLATE_PATTERNS:
        m = pat.match(t)
        if m:
            return formatter(m)
    return None
```

Then replace `filter_text`:

```python
def filter_text(text):
    if not text:
        return ""

    t = text.upper().strip()
    t = t.replace("—", "-").replace("–", "-").replace("_", "-").replace(" ", "")
    t = _RE_PLATE_CLEAN.sub("", t)

    pat_full = _RE_PLATE_FULL
    pat_comp = _RE_PLATE_COMP
    m = pat_full.match(t)
    if m:  # định dạng đủ, chỉ cần chuẩn hoá dấu
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}.{m.group(4)}"
    m = pat_comp.match(t)
    if m:  # 12A34567 -> 12A-345.67
        s = m.group(3)
        return f"{m.group(1)}{m.group(2)}-{s[:3]}.{s[3:]}"


    if len(t) >= 3 and t[0:2].isdigit() and t[2].isdigit():
        digit_to_letter = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "4": "A"}
        c_fix = digit_to_letter.get(t[2])
        if c_fix:
            t2 = t[:2] + c_fix + t[3:]
            m = pat_full.match(t2) or pat_comp.match(t2)
            if m:
                if len(m.groups()) == 4:
                    return f"{m.group(1)}{m.group(2)}-{m.group(3)}.{m.group(4)}"
                else:
                    s = m.group(3)
                    return f"{m.group(1)}{m.group(2)}-{s[:3]}.{s[3:]}"


    compact = _RE_NON_ALNUM.sub("", t)
    return compact if len(compact) >= 5 else ""
```

with:

```python
def filter_text(text):
    if not text:
        return ""

    t = text.upper().strip()
    t = t.replace("—", "-").replace("–", "-").replace("_", "-").replace(" ", "")
    t = _RE_PLATE_CLEAN.sub("", t)

    matched = _match_plate_format(t)
    if matched:
        return matched

    if len(t) >= 3 and t[0:2].isdigit() and t[2].isdigit():
        digit_to_letter = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "4": "A"}
        c_fix = digit_to_letter.get(t[2])
        if c_fix:
            t2 = t[:2] + c_fix + t[3:]
            matched = _match_plate_format(t2)
            if matched:
                return matched

    compact = _RE_NON_ALNUM.sub("", t)
    return compact if len(compact) >= 5 else ""
```

Then replace `canonicalize_plate`:

```python
def canonicalize_plate(s: str) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).upper().strip()

    t = t.replace("—","-").replace("–","-").replace("_","-")

    t = re.sub(r"[^A-Z0-9\-\.]", "", t)

    trans = str.maketrans({"O":"0","Q":"0","I":"1","L":"1","Z":"2","S":"5","B":"8"})
    t = t.translate(trans)


    if len(t) >= 3 and t[:2].isdigit() and t[2].isdigit():
        fix_map = {"0":"O","1":"I","2":"Z","5":"S","8":"B","4":"A"}
        t = t[:2] + fix_map.get(t[2], t[2]) + t[3:]


    m = _RE_PLATE_FULL.match(t)
    if m:
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}.{m.group(4)}"

    m = _RE_PLATE_COMP.match(t)
    if m:
        last5 = m.group(3)
        return f"{m.group(1)}{m.group(2)}-{last5[:3]}.{last5[3:]}"

    compact = _RE_NON_ALNUM.sub("", t)
    return compact if len(compact) >= 5 else ""
```

with:

```python
def canonicalize_plate(s: str) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).upper().strip()

    t = t.replace("—","-").replace("–","-").replace("_","-")

    t = re.sub(r"[^A-Z0-9\-\.]", "", t)

    trans = str.maketrans({"O":"0","Q":"0","I":"1","L":"1","Z":"2","S":"5","B":"8"})
    t = t.translate(trans)


    if len(t) >= 3 and t[:2].isdigit() and t[2].isdigit():
        fix_map = {"0":"O","1":"I","2":"Z","5":"S","8":"B","4":"A"}
        t = t[:2] + fix_map.get(t[2], t[2]) + t[3:]


    matched = _match_plate_format(t)
    if matched:
        return matched

    compact = _RE_NON_ALNUM.sub("", t)
    return compact if len(compact) >= 5 else ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest test_detectnp -v`

Expected: all tests in `FilterTextCarFormatTests`, `FilterTextMotorbikeFormatTests`, `CanonicalizePlateMotorbikeFormatTests` PASS.

- [ ] **Step 5: Commit**

```bash
git add DetectNP.py test_detectnp.py
git commit -m "Add motorbike-format plate regex support to DetectNP.py

filter_text/canonicalize_plate previously only recognized the car-plate
format (single-letter series, e.g. 18A-123.45). Real motorbike plates
use a letter+digit series (B1, V7, L1, U2, S1, V5) which fell into the
permissive strip-punctuation fallback and formatted incorrectly."
```

---

### Task 2: Strict-format candidate preference in OCR

**Files:**
- Modify: `DetectNP.py` (`filter_text` — add `strict` param; new `select_plate_text` function; `ocr_it`'s candidate-selection loop)
- Test: `test_detectnp.py` (append)

**Interfaces:**
- Consumes: `_match_plate_format(t)` from Task 1 (used internally, no signature change).
- Produces: `filter_text(text, strict=False)` — new optional second parameter, default preserves all Task-1 behavior. New function `select_plate_text(candidates: list[str]) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `test_detectnp.py` (add `select_plate_text` to the import line, and add these classes before `if __name__ == "__main__":`):

```python
from DetectNP import filter_text, canonicalize_plate, select_plate_text


class FilterTextStrictModeTests(unittest.TestCase):
    def test_strict_true_returns_empty_for_unformatted_text(self):
        self.assertEqual(filter_text("RANDOMTEXT123", strict=True), "")

    def test_strict_true_still_matches_valid_car_format(self):
        self.assertEqual(filter_text("18A-123.45", strict=True), "18A-123.45")

    def test_strict_false_keeps_loose_fallback_behavior(self):
        self.assertEqual(filter_text("RANDOMTEXT123", strict=False), "RANDOMTEXT123")


class SelectPlateTextTests(unittest.TestCase):
    def test_strict_match_on_first_candidate_wins(self):
        candidates = ["18A-123.45", "18A 123.45 EXTRA"]
        self.assertEqual(select_plate_text(candidates), "18A-123.45")

    def test_strict_match_on_later_candidate_preferred_over_earlier_loose_match(self):
        candidates = ["10DTM5INWJRS6VEHICVFHP", "18A-123.45"]
        self.assertEqual(select_plate_text(candidates), "18A-123.45")

    def test_falls_back_to_loose_match_when_no_strict_match_exists(self):
        candidates = ["10DTM5INWJRS6VEHICVFHP", "SHORT"]
        self.assertEqual(select_plate_text(candidates), "10DTM5INWJRS6VEHICVFHP")

    def test_empty_string_when_nothing_qualifies(self):
        candidates = ["AB", "1"]
        self.assertEqual(select_plate_text(candidates), "")
```

(Replace the existing `from DetectNP import filter_text, canonicalize_plate` line at the top of the file with the updated import line shown above — don't duplicate the import.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest test_detectnp -v`

Expected: `FilterTextStrictModeTests` and `SelectPlateTextTests` FAIL — `filter_text() got an unexpected keyword argument 'strict'` / `cannot import name 'select_plate_text'`.

- [ ] **Step 3: Implement `strict` mode and `select_plate_text`**

In `DetectNP.py`, change the `filter_text` signature and final fallback (replace the function written in Task 1):

```python
def filter_text(text, strict=False):
    if not text:
        return ""

    t = text.upper().strip()
    t = t.replace("—", "-").replace("–", "-").replace("_", "-").replace(" ", "")
    t = _RE_PLATE_CLEAN.sub("", t)

    matched = _match_plate_format(t)
    if matched:
        return matched

    if len(t) >= 3 and t[0:2].isdigit() and t[2].isdigit():
        digit_to_letter = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "4": "A"}
        c_fix = digit_to_letter.get(t[2])
        if c_fix:
            t2 = t[:2] + c_fix + t[3:]
            matched = _match_plate_format(t2)
            if matched:
                return matched

    if strict:
        return ""

    compact = _RE_NON_ALNUM.sub("", t)
    return compact if len(compact) >= 5 else ""


def select_plate_text(candidates):
    """Given OCR candidate strings in preference order, return the first
    that strictly matches a known plate format; if none do, fall back to
    the first that satisfies the loose alphanumeric-length heuristic;
    else "" if nothing qualifies."""
    for cand in candidates:
        ft = filter_text(cand, strict=True)
        if ft:
            return ft
    for cand in candidates:
        ft = filter_text(cand)
        if ft:
            return ft
    return ""
```

In `ocr_it`, replace:

```python
    # Chọn ứng viên khớp pattern VN
    for cand in candidates:
        ft = filter_text(cand)
        if ft:
            # details
            all_items = [{"text": t, "conf": 0.0} for t in row_texts]
            return ft, {"all": all_items, "best_conf": max((c for *_ , c in items), default=0.0)}
```

with:

```python
    # Chọn ứng viên khớp pattern VN (ưu tiên khớp định dạng chuẩn trước khi rơi vào fallback)
    ft = select_plate_text(candidates)
    if ft:
        all_items = [{"text": t, "conf": 0.0} for t in row_texts]
        return ft, {"all": all_items, "best_conf": max((c for *_ , c in items), default=0.0)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest test_detectnp -v`

Expected: all tests pass, including Task 1's tests (regression check).

- [ ] **Step 5: Commit**

```bash
git add DetectNP.py test_detectnp.py
git commit -m "Prefer format-valid OCR candidates over the loose fallback

ocr_it tried candidates in a fixed order and accepted the first one
satisfying a permissive >=5-alphanumeric-character check, so a garbled
multi-row join could win over a later candidate that actually matched a
known plate format. select_plate_text now does a strict-format pass
over all candidates before falling back to the original loose pass."
```

---

### Task 3: Registry-assisted confusable-character correction

**Files:**
- Modify: `DetectNP.py` (new `_CONFUSABLE_PAIRS`, `_confusable_variants`, `correct_against_registry`; `lookup_owner` wiring)
- Test: `test_detectnp.py` (append)

**Interfaces:**
- Consumes: `canonicalize_plate(s)` from Task 1 (unchanged signature).
- Produces: `correct_against_registry(plate: str, registry_df: pandas.DataFrame) -> str`. `lookup_owner(plate, path=REGISTRY_CSV)` behavior extended (signature unchanged).

- [ ] **Step 1: Write the failing tests**

Append to `test_detectnp.py` (update the import line again, add `csv` and `tempfile` imports, and add these classes before `if __name__ == "__main__":`):

```python
import csv
import tempfile
import pandas as pd

from DetectNP import (
    filter_text, canonicalize_plate, select_plate_text,
    correct_against_registry, lookup_owner,
)


def _make_registry(rows):
    df = pd.DataFrame(rows, columns=["plate", "owner_name", "phone", "notes"])
    df["plate_norm"] = df["plate"].apply(canonicalize_plate)
    return df


class CorrectAgainstRegistryTests(unittest.TestCase):
    def test_exact_match_returns_plate_unchanged(self):
        registry = _make_registry([
            {"plate": "30A-339.18", "owner_name": "A", "phone": "", "notes": ""},
        ])
        self.assertEqual(correct_against_registry("30A-339.18", registry), "30A-339.18")

    def test_single_unambiguous_confusable_correction(self):
        registry = _make_registry([
            {"plate": "30A-339.18", "owner_name": "A", "phone": "", "notes": ""},
        ])
        self.assertEqual(correct_against_registry("80A-339.18", registry), "30A-339.18")

    def test_no_correction_when_no_registry_match_found(self):
        registry = _make_registry([
            {"plate": "30A-339.18", "owner_name": "A", "phone": "", "notes": ""},
        ])
        self.assertEqual(correct_against_registry("99Z-999.99", registry), "99Z-999.99")

    def test_no_correction_when_ambiguous(self):
        registry = _make_registry([
            {"plate": "30A-339.18", "owner_name": "A", "phone": "", "notes": ""},
            {"plate": "80A-339.13", "owner_name": "B", "phone": "", "notes": ""},
        ])
        self.assertEqual(correct_against_registry("80A-339.18", registry), "80A-339.18")


class LookupOwnerRegistryCorrectionTests(unittest.TestCase):
    def test_lookup_owner_applies_registry_correction(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "registry.csv")
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["plate", "owner_name", "phone", "notes"])
                writer.writerow(["30A-339.18", "Nguyen Van A", "0900000000", ""])
            result = lookup_owner("80A-339.18", path=path)
            self.assertIsNotNone(result)
            self.assertEqual(result["owner_name"], "Nguyen Van A")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest test_detectnp -v`

Expected: `CorrectAgainstRegistryTests` and `LookupOwnerRegistryCorrectionTests` FAIL — `cannot import name 'correct_against_registry'`.

- [ ] **Step 3: Implement `correct_against_registry` and wire it into `lookup_owner`**

In `DetectNP.py`, replace `lookup_owner`:

```python
def lookup_owner(plate: str, path: str = REGISTRY_CSV):
    if not plate:
        return None
    df = load_registry(path)
    key = canonicalize_plate(plate)
    row = df[df["plate_norm"] == key]
    if not row.empty and row.iloc[0].get("owner_name","").strip():
        r = row.iloc[0].to_dict()
        return {
            "plate": r.get("plate",""),
            "owner_name": r.get("owner_name",""),
            "phone": r.get("phone",""),
            "notes": r.get("notes","")
        }
    return None
```

with:

```python
_CONFUSABLE_PAIRS = (("3", "8"), ("I", "V"), ("O", "0"), ("1", "I"), ("S", "5"))


def _confusable_variants(text: str) -> set:
    variants = set()
    chars = list(text)
    for i, ch in enumerate(chars):
        for a, b in _CONFUSABLE_PAIRS:
            replacement = None
            if ch == a:
                replacement = b
            elif ch == b:
                replacement = a
            if replacement is not None:
                swapped = chars.copy()
                swapped[i] = replacement
                variants.add("".join(swapped))
    variants.discard(text)
    return variants


def correct_against_registry(plate: str, registry_df: pd.DataFrame) -> str:
    """If `plate` has no registry match, try single-character confusable
    substitutions; if exactly one substitution matches a registered plate,
    return that corrected plate. Otherwise return `plate` unchanged."""
    key = canonicalize_plate(plate)
    if not key:
        return plate

    known_norms = set(registry_df["plate_norm"])
    if key in known_norms:
        return plate

    matched_norms = {v for v in _confusable_variants(key) if v in known_norms}
    if len(matched_norms) != 1:
        return plate

    matched_norm = next(iter(matched_norms))
    row = registry_df[registry_df["plate_norm"] == matched_norm].iloc[0]
    return row["plate"]


def lookup_owner(plate: str, path: str = REGISTRY_CSV):
    if not plate:
        return None
    df = load_registry(path)
    key = canonicalize_plate(plate)
    row = df[df["plate_norm"] == key]
    if row.empty:
        corrected = correct_against_registry(plate, df)
        if corrected != plate:
            key = canonicalize_plate(corrected)
            row = df[df["plate_norm"] == key]
    if not row.empty and row.iloc[0].get("owner_name","").strip():
        r = row.iloc[0].to_dict()
        return {
            "plate": r.get("plate",""),
            "owner_name": r.get("owner_name",""),
            "phone": r.get("phone",""),
            "notes": r.get("notes","")
        }
    return None
```

Note: `correct_against_registry` must be defined textually before `lookup_owner` in the file (Python doesn't require this for module-level functions calling each other, but keep them adjacent for readability, matching the replacement above).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest test_detectnp -v`

Expected: all tests pass, including Tasks 1 and 2 (regression check).

- [ ] **Step 5: Commit**

```bash
git add DetectNP.py test_detectnp.py
git commit -m "Add registry-assisted confusable-character plate correction

When an OCR'd plate has no exact registry match, try single-character
confusable substitutions (3/8, I/V, O/0, 1/I, S/5); if exactly one
substitution resolves to a registered plate, use it. Ambiguous (>1
match) or unmatched corrections are left unchanged, since format
validity alone can't arbitrate which digit is correct."
```

---

### Task 4: Detection-threshold experiment

**Files:**
- Create (scratchpad, not committed): `threshold_experiment.py`
- Create (committed): `docs/superpowers/notes/2026-07-17-threshold-experiment-results.md`

**Interfaces:**
- Consumes: `anpr_eval.run_predictions(image_dir, out_csv) -> list[dict]`, `anpr_eval.read_csv_rows(path) -> list[dict]`, `anpr_eval.evaluate(gt_rows, pred_rows) -> dict`, `anpr_eval._register_anpr_yolo_package()` (all already implemented and tested). `DetectNP.CONF_THRES` / `DetectNP.ANPR_IMGSZ` module-level globals (read by `detect_fn` at call time, so reassigning them before calling `run_predictions` changes detection behavior for that call).

- [ ] **Step 1: Write the experiment script**

Create `threshold_experiment.py` in the scratchpad directory
(`C:\Users\dotha\AppData\Local\Temp\claude\C--Users-dotha-OneDrive-Desktop-ANPR-using-OpenCV-Yolov8-using-python\e33fd3f9-5ebe-48c4-a506-207285a647be\scratchpad`),
**not** in the repo:

```python
import os
import sys

REPO = r"C:\Users\dotha\OneDrive\Desktop\ANPR-using-OpenCV-Yolov8-using-python"
SCRATCH = r"C:\Users\dotha\AppData\Local\Temp\claude\C--Users-dotha-OneDrive-Desktop-ANPR-using-OpenCV-Yolov8-using-python\e33fd3f9-5ebe-48c4-a506-207285a647be\scratchpad"
sys.path.insert(0, REPO)
os.chdir(REPO)

import anpr_eval
anpr_eval._register_anpr_yolo_package()
from ANPR_Yolo import DetectNP  # must import via this qualified path so the
                                  # module object matches the one ANPR.py's
                                  # detect_fn actually reads CONF_THRES from

CONFIGS = [
    ("baseline", 0.25, 640),
    ("lower_conf", 0.15, 640),
    ("larger_imgsz", 0.25, 960),
    ("lower_conf_larger_imgsz", 0.15, 960),
]

IMAGE_SETS = [
    ("eval_images_vn", "gt_vn.csv"),
    ("eval_images", "gt.csv"),
]

results = []
for name, conf, imgsz in CONFIGS:
    DetectNP.CONF_THRES = conf
    DetectNP.ANPR_IMGSZ = imgsz
    for images_dir, gt_path in IMAGE_SETS:
        preds_path = os.path.join(SCRATCH, f"preds_{name}_{images_dir}.csv")
        pred_rows = anpr_eval.run_predictions(images_dir, preds_path)
        gt_rows = anpr_eval.read_csv_rows(gt_path)
        metrics = anpr_eval.evaluate(gt_rows, pred_rows)
        results.append((name, conf, imgsz, images_dir, metrics))

lines = [
    "# Detection-threshold experiment results", "",
    "Run against `eval_images_vn/` (real VN photos, 11 plates, includes "
    "3 crowded multi-plate scenes) and `eval_images/` (mixed US/VN "
    "sanity check) to see whether lowering CONF_THRES or raising "
    "ANPR_IMGSZ recovers missed detections in crowded scenes without a "
    "material false-positive cost. No env.py defaults were changed by "
    "this experiment.", "",
]
lines.append("| Config | CONF_THRES | ANPR_IMGSZ | Image set | Detection Rate | FP | Recognition Accuracy |")
lines.append("|---|---|---|---|---|---|---|")
for name, conf, imgsz, images_dir, m in results:
    dr = f"{m['detection_rate']:.1%}" if m['detection_rate'] is not None else "n/a"
    ra = f"{m['recognition_accuracy']:.1%}" if m['recognition_accuracy'] is not None else "n/a"
    lines.append(f"| {name} | {conf} | {imgsz} | {images_dir} | {dr} | {m['fp']} | {ra} |")

out_path = os.path.join(REPO, "docs", "superpowers", "notes", "2026-07-17-threshold-experiment-results.md")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
```

- [ ] **Step 2: Run it**

Run (from the repo root, expect several minutes on CPU — 4 configs × 2 image sets × up to 27 images each):

```bash
.venv/Scripts/python.exe "C:\Users\dotha\AppData\Local\Temp\claude\C--Users-dotha-OneDrive-Desktop-ANPR-using-OpenCV-Yolov8-using-python\e33fd3f9-5ebe-48c4-a506-207285a647be\scratchpad\threshold_experiment.py"
```

Expected: prints the results table to stdout and writes `docs/superpowers/notes/2026-07-17-threshold-experiment-results.md`.

- [ ] **Step 3: Read the results and report to the user**

Read `docs/superpowers/notes/2026-07-17-threshold-experiment-results.md`. Compare each non-baseline row's Detection Rate and FP count on `eval_images_vn` against the `baseline` row. Present the table to the user with a recommendation (adopt a specific config as the new `env.py` default, or none if nothing clearly wins) and **stop** — do not edit `env.py` in this task. Changing `env.py` requires a separate, explicit user go-ahead per the Global Constraints above.

- [ ] **Step 4: Commit the results file**

```bash
git add docs/superpowers/notes/2026-07-17-threshold-experiment-results.md
git commit -m "Record detection-threshold experiment results

Compares CONF_THRES/ANPR_IMGSZ combinations against eval_images_vn/ and
eval_images/ to check whether missed detections in crowded scenes can
be recovered without a material false-positive cost. No env.py changes
made here — see the results table for the recommendation."
```

---

## Self-Review Notes

- **Spec coverage:** Design section 1 (motorbike regex) → Task 1. Section 2 (strict candidate preference) → Task 2. Section 3 (registry-assisted correction) → Task 3. Section 4 (threshold experiment) → Task 4. Testing section's four bullet points map 1:1 to the test classes added in Tasks 1–3; the "re-run eval as a sanity check" bullet is intentionally left as a manual step for whoever executes this plan to run after Task 3, not a scripted task, since it's a one-off sanity check with no pass/fail assertion (expected outcome is "roughly unchanged").
- **Type consistency checked:** `select_plate_text` (Task 2) is used inside `ocr_it` exactly as defined; `correct_against_registry`'s `registry_df` parameter matches the `pd.DataFrame` shape produced by both `load_registry()` (existing) and the tests' `_make_registry()` helper (has `plate`, `owner_name`, `phone`, `notes`, `plate_norm` columns in both cases).
- **No placeholders:** every step has complete, runnable code; no "TODO"/"handle appropriately" language.
