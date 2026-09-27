# Golden raw-file facts

These JSON files are measurements taken from the LAS fixtures in `tests/fixtures/`.
They record what the file says: its SHA-256, version, wrap flag, declared NULL,
STRT / STOP / STEP as written, every curve line, how many depth steps the data
section contains, and, for each curve, how many samples match the declared NULL,
the first and last index-channel depth where the curve is not null, and the
minimum and maximum of those non-null samples.

They were produced by `tests/golden/extract_facts.py`. That reader is separate
from POLISH and from the legacy pipeline. It does not import `polish`, `lasio`,
or the monolith. Re-run it with:

```text
python -m tests.golden.extract_facts
```

The value/description delimiter is the last colon on the line (D65, CWLS LAS 2.0).

A NULL sample is one that `math.isclose` treats as the declared NULL with
relative tolerance `1e-6` and absolute tolerance `0`. WRAP NO means one data
line is one depth step. WRAP YES means one depth step may continue onto the
next lines until every curve has a value. The depth stored for a sample is the
value of curve index 0, the LAS index channel.

KEOUGH is not included. D5 is deferred until a regression well with a clean
public licence is available.

These files are not approved expectations yet. Tests may use them only to detect
that a fixture or the extractor changed. Tests must read counts from these files
rather than hardcoding a curve count, a depth-step count, or a prefix count.

Reviewer:
Date: Sept, 26, 2026
