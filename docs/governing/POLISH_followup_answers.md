POLISH — Six follow-up questions

Following up on your worksheet answers

Thank you for the worksheet — your answers are already in the plan. Six of
them need one more detail before the parts they affect can be built. For
each one I've included real numbers from the test wells, so you can answer
against actual data instead of a description.

Tick a box or write in the space. Leave anything blank you're unsure about
— a blank is better than a guess.

The wells used in the examples

Name | Well | Where it comes from | Depth unit | Curves in file
A | 15/9-19 (vendor-processed file) | Norwegian North Sea, block 15/9
(Volve area), public data | metres | 171
B | 30/3-5 S | FORCE 2020 public dataset, Norwegian North Sea | metres | 16
C | 34/10-19 | FORCE 2020 public dataset, Norwegian North Sea | metres | 18
D | 15/9-F-1 B | Norwegian North Sea, Volve area, public data | metres | 23
E | KEOUGH #12-34 | Kansas | feet | 21

Question 1 also uses 144 Norwegian wells (the FORCE 2020 set plus Volve
files) that were checked earlier.

1. When the bottom depth in the header doesn't match the data

Your answer: check whether the difference could be a rounding error; if
so, correct it; otherwise skip the well.

Every well has a declared bottom depth (STOP) in its header, and a last
depth in its data. In 144 real wells the two differ like this:

How the header STOP compares with the last data depth | Wells
Identical | 25
Differ only in the last decimal the header was written with (true
rounding) | 4
Differ by less than half a depth step | 58
Differ by half to one depth step | 56
Differ by one depth step or more | 0
Could not be checked | 1

Three examples (depth step 0.152 m in all three):

Well | Header STOP | Last data depth | Difference
25/7-2 | 4950.0 m | 4949.976 m | 0.024 m — header written with one
decimal, so this is plain rounding
15/9-14 | 3575.6577 m | 3575.6520 m | 0.006 m (0.04 of a step) — not
decimal rounding
15/9-15 | 3210.0056 m | 3209.856 m | 0.150 m (0.98 of a step) — not
decimal rounding

Which of these counts as "a rounding error"?

[ ] Only true decimal rounding (like 25/7-2). Result: 114 of the 144 wells
are skipped.
[ ] Anything less than one depth step (all three examples). Result: none
of the 144 wells are skipped.
[ ] Something else:

Diagnose further. Does (stop - start)/(nsam-1) match the sampling
interval? Can the difference be attributed to rounding? If so, correct as
follows: Consider common sampling intervals (fraction of feet when in
feet, fraction of metric when in metric - sometimes sample of feet
converted to metric - ie. 0.1524 m which is half fot in meters). Take
determined sampling interval and correct depth track.

"Correct accordingly" will mean: the processed file's header states the
real last depth from the data, and the change is recorded. The original
file is never altered. Is that right?

[ ] Yes
[ ] No — explain:

See the previous question

2. Your limits — impossible, or just suspicious?

The software needs to know which kind each limit is:

Impossible: the value cannot be real. POLISH flags it and asks you
whether to leave it out of the processed file. Leave out of the processed
file when values are impossible.

Suspicious: unusual but could be real. POLISH shows a warning and asks
whether to eliminate these points. If yes, then remember this answer for
all wells. User will consider area/lithology before commiting.

Here is what your limits catch in the five test wells:

Curve | Your limit | What the test wells show | Impossible or suspicious?
| Notes
GR | 0-300 GAPI | Well A: 11 values above 300, up to 1,568, at
3,703-3,705 m. Well E: 4 values above 300, all in the top 1.5 ft (563 at
0 ft). Well D: highest value 297.8. | Suspicious all three. |
RHOB | 1.5-3.5 g/cc | Well B: 4 values below 1.5 (lowest 1.332) at
4,126.0-4,126.5 m. The file's own lithology column says code 90000 there
(coal, if I read the FORCE key right). Well C: 119 values below 1.5
(lowest 1.379) between 289 and 915 m. Lithology column says code 30000
(sandstone). | Suspicious |
NPHI | -0.15-0.6 v/v | Well A: 12 values above 0.6, including 6.9, 8.8,
12.1, 15.7 and 61.9 - in a curve labelled v/v. Well C: 45 values above
0.6 (highest 0.69). Well B: 1 value (0.62), at the same depth as the
coal. | Suspicious |
DT | 40-300 us/ft | Well E: 7 values below 40 at 5,352-5,355 ft, dropping
to 0.002 at the bottom of the log. | Impossible |
PEF | -2-270 b/e | Well B: 1 value above 270 (334.5 at 3,898.5 m). Well
E: the PE curve has no unit in its header, so it won't be checked until a
unit is confirmed. | |
Resistivity | -0.02-20,000 ohm.m | Well D (RT): 40 values above 20,000,
up to 97,540, near 2,589 m. Well E: RILD has 442 and RILM 1,394 values
above 20,000, reaching 100,000. | Suspicious |
SP | -150-50 mV | Well C: 6,703 of its 13,224 values are above 50 mV
(highest 92.6). Well E: 866 above 50 and 60 below -150 (lowest -200 at
4.5 ft). | Values below -150 are impossible. Above 50 are suspicious |

SP: a fixed limit flags half of Well C. Should SP have a limit at all?

[ ] Yes, keep it as written
[ ] Yes, but different numbers:
[ ] No SP limit

No SP limit

### Final SP answer (18 Sept)

The 16 Sept "No SP limit" answer was a reaction to per-sample flagging volume (51% of Well C), not a rejection of the −150/50 mV numbers. Final rule: keep −150/50 mV, but flag at the log level — a log is flagged for review when more than 20% of its samples fall outside that range; individual out-of-range samples don't trigger anything on their own.

Any curve where the limit should depend on something else (units, rock
type, hole conditions)?

3. STEP blank or zero, and the depths are uneven

Your answer: check the depth curve and fill in the step if appropriate;
otherwise skip.

Example of the easy case: one of the 144 wells (15/9-19 SR, a processed
file) declares STEP = 0.0, but its data is evenly spaced every 0.1524 m.
POLISH will record the step as 0.1524 m and tell you.

The unclear case is when the depths are genuinely uneven (none of the 144
wells does this, but other files will). What should POLISH do?

[ ] Skip the well
[ ] Accept it as unevenly spaced, don't invent a step, and tell me
[ ] Something else:

Accept it as unevenly spaced, don't invent a step, and tell me

4. What "delete" means for a value outside the limits

Your answer: ask whether to delete the value, for this well only or for
all wells; never clip.

Example: Well B, RHOB = 1.332 g/cc at 4,126.3 m. If you choose "delete",
POLISH would: leave the value out of the processed file (blank at that
depth), keep it untouched in the original file, keep showing it on
screen, marked, so you can always see what was left out, record the
decision in the report.

"All wells" would mean: on every future well, values of this curve
outside this limit are left out automatically, and the report says so.

Is that what you meant?

[ ] Yes
[ ] No — explain:

Delete means mark as "no value" visually -999.25

5. Default tracks when the curve isn't in the file

Your default tracks, left to right: DTP, DTS, density, porosity, Sw, GR,
ILD.

In the five test wells: none has a water-saturation (Sw) curve. Only A
and D have shear sonic (DTS). Only E has a deep induction curve (RILD). A,
B, C and D have other resistivity curves instead (for example RDEP, RD,
RT).

When one of your default curves isn't there, what should the screen do?

[ ] Show an empty track labelled "not in this file"
[ ] Skip that track
[ ] Use the nearest equivalent (for example RDEP in place of ILD) and
label it clearly

If the nearest equivalent: which substitutes are acceptable for ILD?

Use the nearest equivalent (for example RDEP in place of ILD) and label
it clearly

If the nearest equivalent: which substitutes are acceptable for ILD?

6. KEOUGH - the eight feet

Your answer: skip and notify.

KEOUGH's header says the logged interval is 4,250 ft (TLI1) to 5,326 ft
(BLI1). The density data runs from 4,250 ft to 5,334 ft. The last
readings are:

Depth (ft) | RHOB (g/cc)
5,332.0 | 2.624
5,332.5 | 2.640
5,333.0 | 2.701
5,333.5 | 2.854
5,334.0 | 3.247

What should POLISH skip?

[ ] Only the density readings below 5,326 ft (5,326.5 - 5,334 ft)
[ ] The whole density curve
[ ] Something else:

Only the density readings below 5,326 ft (5,326.5 - 5,334 ft)

Anything you're unsure about, leave blank - a guess written down turns
into a decision nobody remembers making.
