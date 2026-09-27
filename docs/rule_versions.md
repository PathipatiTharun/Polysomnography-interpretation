# Adult apnea / hypopnea scoring rules: the published versions

The scorer in `psg/respiratory.py` is parameterised so that any of these definitions can be
selected (`psg/rules.py`, or the *Algorithm* panel in the app). All numbers below are taken
from the cited sources; where a source uses words rather than numbers ("complete cessation",
"clear decrease") the operationalisation used by this software is stated.

| id | Rule | Year | Apnea | Hypopnea flow drop | Min. duration | Confirmation | Extra requirement | Hypopnea O/C classification |
|---|---|---|---|---|---|---|---|---|
| `chicago1999` | AASM Task Force "Chicago criteria" [1][3] | 1999 | complete cessation ≥10 s (operationalised: ≥90 % drop) | >50 % alone, **or** smaller "clear" decrease (operationalised ≥30 %) | 10 s | ≥3 % desaturation **or** arousal (only needed for the smaller decrease) | – | – |
| `aasm2007_rec` | AASM Manual 2007, rule 4A recommended [2][3] | 2007 | ≥90 % drop, thermal sensor, ≥10 s | ≥30 % (nasal pressure) | 10 s | ≥4 % desaturation; arousal not accepted | ≥90 % of the event's duration must meet the amplitude drop | – |
| `aasm2007_alt` | AASM Manual 2007, rule 4B alternative [2][3] | 2007 | as 4A | ≥50 % | 10 s | ≥3 % desaturation **or** arousal | ≥90 % of duration | – |
| `aasm2012_rec` | AASM Manual v2.0, rule 1A recommended [4] | 2012 | ≥90 % drop of peak excursion ≥10 s | ≥30 % | 10 s | ≥3 % desaturation **or** arousal | duration-fraction rule removed; event counts if it begins **or** ends in a sleep epoch (note 3) | yes: obstructive if snoring / flattening / paradox, else central (rules 2–3) |
| `aasm2013_acc` | AASM Manual v2.0.2, rule 1B acceptable = CMS/Medicare [5] | 2013 | as v2.0 | ≥30 % | 10 s | ≥4 % desaturation; arousal excluded | as v2.0 | yes |
| `shhs` | Sleep Heart Health Study, AHI "a0h4" [6] | 1995–2000 | cessation / near-cessation of airflow ≥10 s (operationalised ≥90 %) | ≥30 % airflow **or** thoraco-abdominal excursion | 10 s | ≥4 % desaturation (apneas need none) | – | – |
| `aasm_current` | AASM Manual v2.6 (2020) and later, rule 1A [7] | 2020+ | as v2.0 | ≥30 % | 10 s | ≥3 % desaturation or arousal (1B = 4 % kept as acceptable) | as v2.0 | yes |

Rules shared by all versions and implemented once: baseline = breathing in the preceding
2 minutes (mean of stable breathing, or of the largest breaths when unstable); apnea
classification from effort — obstructive (effort continues), central (effort absent), mixed
(absent then resumes); severity AHI <5 normal, 5–15 mild, 15–30 moderate, ≥30 severe.

## Why the version matters

Ruehland et al. (2009) scored the same 328 patients under three definitions and found the
median AHI under the 2007 recommended rule was about **30 %** of the Chicago AHI, and under
the 2007 alternative rule about **60 %**; "approximately 40 % of patients previously classified
as positive for OSA using AHI-Chicago [would be] negative using AHI-Rec" [3]. The app makes
this visible: switching the rule re-scores the night in about a second and the AHI changes
accordingly.

## Confidence and explanation (this software, not part of any manual)

For every candidate event the scorer records each criterion it was tested against (airflow
drop, duration, desaturation/arousal, sleep, signal quality, effort) with the measured value
and the rule's threshold. The **confidence** is the smallest margin among the rule criteria,
mapped so that a value exactly at the threshold gives 50 % and a value comfortably past it
gives >95 %. An event with a 3.1 % desaturation under a 3 % rule is therefore a low-confidence
event; the same event under the 4 % rule is rejected, and the app can show it greyed out with
the reason. A separate **type confidence** says how clear the obstructive / central / mixed
label is (distance of the belt movement from the "effort absent" threshold; for hypopneas, the
number of obstructive cues found).

## Sources

1. American Academy of Sleep Medicine Task Force. *Sleep-related breathing disorders in adults:
   recommendations for syndrome definition and measurement techniques in clinical research.*
   Sleep 1999;22(5):667–689.
2. Iber C, Ancoli-Israel S, Chesson A, Quan SF. *The AASM Manual for the Scoring of Sleep and
   Associated Events*, 1st ed. AASM, 2007.
3. Ruehland WR, Rochford PD, O'Donoghue FJ, Pierce RJ, Singh P, Thornton AT. *The new AASM
   criteria for scoring hypopneas: impact on the apnea hypopnea index.* Sleep 2009;32(2):150–157.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC2635578
4. Berry RB et al. *Rules for scoring respiratory events in sleep: update of the 2007 AASM
   Manual.* J Clin Sleep Med 2012;8(5):597–619. https://pubmed.ncbi.nlm.nih.gov/23066376/ — and
   AASM, *The 2007 AASM Scoring Manual vs. the AASM Scoring Manual v2.0*, October 2012.
   https://aasm.org/wp-content/uploads/2017/11/Summary-of-Updates-in-v2.0-FINAL.pdf
5. AASM. *AASM clarifies hypopnea scoring criteria*, 23 September 2013.
   https://aasm.org/aasm-clarifies-hypopnea-scoring-criteria/
6. National Sleep Research Resource, SHHS variable `ahi_a0h4`.
   https://www.sleepdata.org/datasets/shhs/variables/ahi_a0h4 ; Nieto FJ et al. JAMA 2000;283:1829–1836.
7. AASM. *The AASM Manual for the Scoring of Sleep and Associated Events*, v2.6 (2020) and later.
   https://aasm.org/clinical-resources/scoring-manual/

## Effect on our test nights

Automatic staging throughout; the same prepared signals re-scored under each rule. Cells show the
automatic AHI and, in brackets, event sensitivity % / precision % against the technician's scoring.
The four UCDDB development nights were used to tune the signal processing; `unseen_night_A/B`
(UCDDB 010 and 022) were processed blind.

| Night | Expert AHI | 1999 Chicago | 2007 rec (4 %) | 2007 alt (50 %, 3 %/arousal) | 2012 rec (3 %/arousal) | 2013 acc / CMS (4 %) | SHHS (4 %) | v2.6+ rule 1A |
|---|---|---|---|---|---|---|---|---|
| ucddb018 | 2.2 (Normal) | 3.2 (22/15) | 0.5 (0/0) | 0.5 (0/0) | 1.2 (0/0) | 0.5 (0/0) | 0.5 (0/0) | 1.2 (0/0) |
| ucddb002 | 23.7 (Moderate) | 19.0 (49/70) | 10.3 (31/83) | 8.7 (23/72) | 17.5 (48/75) | 10.5 (32/83) | 10.5 (32/83) | 17.5 (48/75) |
| ucddb003 | 51.3 (Severe) | 46.5 (72/83) | 44.2 (70/85) | 24.9 (43/92) | 46.5 (72/83) | 45.4 (70/83) | 45.4 (70/83) | 46.5 (72/83) |
| ucddb025 | 94.8 (Severe) | 73.4 (67/84) | 72.1 (67/85) | 75.7 (66/80) | 73.1 (67/85) | 73.1 (67/85) | 73.1 (67/85) | 73.1 (67/85) |
| unseen_night_A | 33.6 (Severe) | 31.0 (35/48) | 23.0 (30/55) | 12.8 (17/56) | 29.5 (34/49) | 23.5 (31/56) | 23.5 (31/56) | 29.5 (34/49) |
| unseen_night_B | 7.0 (Mild) | 34.8 (63/12) | 3.6 (26/50) | 10.5 (26/17) | 16.4 (48/20) | 3.6 (26/50) | 3.6 (26/50) | 16.4 (48/20) |

What this shows:

- **The rule version can change the diagnosis category on its own.** On `unseen_night_B` the same
  night scores 3.6/h (normal) under the 4 % rules, 16.4/h (moderate) under the 3 %-or-arousal
  rule and 34.8/h (severe) under the Chicago criteria; the technician reported 7.0/h (mild).
- **The 2007 alternative rule (≥50 % drop) roughly halves the AHI** of the 30 % rules on the severe
  nights (24.9 vs 46.5 on ucddb003), consistent with Ruehland et al. 2009.
- **The Chicago "≥50 % drop counts alone" clause over-scores** on nights with unstable, shallow
  breathing but few desaturations (precision 12 % on night B), which is why later manuals required
  a desaturation or arousal for every hypopnea.

## Effect of the confidence cut-off

| Night | Scored events | ≥90 % conf | ≥75 % | ≥50 % | Rejected candidates | AHI at conf ≥0 / ≥75 / ≥90 % |
|---|---|---|---|---|---|---|
| ucddb018 | 5 | 1 | 4 | 5 | 63 | 1.2 / 1.0 / 0.2 |
| ucddb002 | 80 | 34 | 64 | 79 | 60 | 17.5 / 14.0 / 7.4 |
| ucddb003 | 263 | 159 | 219 | 263 | 26 | 46.5 / 38.7 / 28.1 |
| ucddb025 | 345 | 178 | 251 | 339 | 63 | 73.1 / 53.2 / 37.7 |
| unseen_night_A | 162 | 84 | 124 | 162 | 67 | 29.5 / 22.6 / 15.3 |
| unseen_night_B | 64 | 2 | 32 | 63 | 202 | 16.4 / 8.2 / 0.5 |

Raising the cut-off to 75 % removes the events that sit within a hair of a threshold and lowers
the AHI by roughly a fifth; at 90 % only clear-cut events remain. On the borderline night B almost
nothing survives 90 %, which agrees with the technician's low AHI. The cut-off is a review aid, not
a diagnostic setting: the physician should look at the low-confidence events, not silently drop them.
