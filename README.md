# PSG Interpreter

Automated polysomnography (PSG) interpretation for sleep physicians, with a PyQt5 viewer.
It loads an overnight EDF recording and does the following:

1. cleans the signals
2. stages sleep
3. detects EEG arousals
4. scores apneas and hypopneas using the **AASM Scoring Manual** adult rules
5. computes AHI, ODI and sleep architecture
6. writes a draft interpretation, for example *"Obstructive sleep apnea – severe (AHI 46/h)"*

## Quick start (Windows)

Double-click **`run.bat`** (first run creates the environment and installs packages), or:

```bash
.venv\Scripts\python app.py
```

### Doctor workflow: upload, and the system does the rest

1. **Upload.** Drag the EDF recording onto the window, or click **Upload recording…**. Several files can be dropped at once; they are processed in turn.
2. **Automatic processing.** A checklist shows each step as it happens:
   - file check
   - signal cleaning
   - sleep staging
   - arousals
   - AASM apnea/hypopnea scoring
   - oximetry and indices
   - report
   Unreadable or unsuitable files are rejected with a plain-language reason, e.g. "this is a hypnogram, not the recording".
3. **Results.** The app opens on the diagnosis and the first scored event. The report is **saved automatically** to `reports/<recording>/` as HTML, CSV and JSON.
4. **Home screen.** Lists all previous studies with diagnosis, AHI and severity. From there you can open the report or review the signals again.

The staging source is set in the toolbar before uploading; the scoring rule can be changed at any time (see below).

### Choosing the scoring rule, confidence and explanations

The window has three parts, left to right: **Algorithm** (which rule and thresholds), the **signals**, and the **results**.

- **Rule versions.** The *Algorithm* panel lists seven published adult scoring definitions, from the 1999 "Chicago criteria" through the AASM 2007 recommended/alternative rules, the 2012 v2.0 rule 1A, the 2013 rule 1B (CMS/Medicare), the Sleep Heart Health Study definition and the current manual. Each shows its summary, source and thresholds ([docs/rule_versions.md](docs/rule_versions.md)). Choosing one re-scores the loaded night in about a second; a full re-analysis is only needed when the staging source changes.
- **Sliders.** The hard-coded numbers (apnea drop, hypopnea drop, minimum duration, desaturation, arousal, hypopnea classification) are sliders and checkboxes. Moving one re-scores live and marks the parameters as *custom*; *Reset to rule values* restores the published numbers.
- **Confidence per event.** Every event carries a confidence: the smallest margin by which it clears the rule's thresholds (exactly at a threshold = 50 %, comfortably past it > 95 %). The *Confidence ≥* slider on the Events tab hides events below the cut-off and recomputes the AHI from the rest, so you can see how many events are clear-cut (e.g. "≥ 90 %: 84 events, ≥ 75 %: 124, all: 162").
- **Learned probability (P(tech)).** Next to the rule-margin confidence, each event also shows the probability that a technician would have scored it, from a logistic regression on the same measurements (airflow drop, duration, desaturation, arousal, share in sleep, signal quality, effort kept, apnea/hypopnea). It was trained on the six technician-scored nights with leave-one-night-out validation: pooled AUC **0.88** vs 0.80 for the hand-set margin, Brier 0.139 vs 0.187, and roughly calibrated (events predicted at 0.96 were confirmed 91 % of the time; at 0.12, 8 %). The confidence slider can be switched to this basis. Retrain with `python -m psg.calibrate ...` (see `psg/calibrate.py`); weights and validation numbers are stored in `psg/models/event_lr.json`. Caveat: it learned one laboratory's scoring habits from a handful of nights.
- **Why was this scored?** Selecting an event shows each criterion it was tested against, the measured value, the rule's threshold and the margin. Ticking *Rejected candidates* also lists the breathing reductions that did **not** meet the rule, greyed, with the reason ("SpO2 fell 2.0 % (rule needs ≥ 3 %)", "occurs entirely during wake").
- **Flowchart.** The *Flowchart* tab draws the decision steps for the current parameters (sleep → signal quality → ≥ 90 % drop? → ≥ 30 % drop? → desaturation/arousal? → effort on the belts → obstructive / central / mixed) and highlights the path the selected event took.

The same options exist on the command line: `--rule aasm2007_alt`, `--min-confidence 75`.

### Browser version (experimental, `web/`)

A static web page where the doctor drops the EDF file and the same `psg` analysis runs inside the browser (Pyodide), so the recording is never uploaded. `vercel.json` builds it for Vercel. Build and serve it locally with:

```bash
.venv\Scripts\python scripts\build_web.py
.venv\Scripts\python -m http.server 8765 --directory public
```

Status: the page and the analysis engine load, but a full end-to-end run in the browser has not been verified yet.

### Automatic sleep annotations and visualisation (`annotate.py`)

For recordings where breathing can't be scored (e.g. Sleep-EDF, whose airflow and chin channels are stored at 1 Hz), or when you want the sleep micro-structure as well:

```bash
.venv\Scripts\python annotate.py data\sleepedf\SC4001E0-PSG.edf
```

This writes `reports/annotations/<recording>/`:
- **`<name>_auto-annotations.edf`**: an EDF+ annotation file in the same format as the technician's `*-Hypnogram.edf`. Open it next to the recording in EDFbrowser, MNE or Polyman.
- **`<name>_annotations.csv`**: the same annotations as a table (onset, clock time, duration, label, channel, details).
- **`<name>_annotations.html`**, with three figures:
  - whole-night overview: technician vs automatic hypnogram, EEG spectrogram, event rasters, chin tone, temperature, breathing
  - one example 30-s epoch per stage with the detected events shaded
  - agreement with the technician

Annotations produced:
- Sleep stage per 30-s epoch (AASM names).
- Estimated lights off / on for long recordings.
- **Sleep spindles:** 11–16 Hz bursts, 0.5–3 s, in N2/N3.
- **Slow waves:** 0.3–2 Hz, negative peak ≤ −40 µV, peak-to-peak ≥ 75 µV, in N2/N3.
- **Rapid eye movements:** steep EOG deflections ≥ 40 µV in REM.
- **EEG arousals.**
- Artifact epochs.

### Batch mode (no GUI)

Uses the same processing and adds the studies to the same list:

```bash
.venv\Scripts\python analyze.py data\ucddb\*.rec data\sleepedf\SC4001E0-PSG.edf --out reports
```

To run the tests: `.venv\Scripts\python tests\test_scoring.py`

## Data used (PhysioNet, open access)

| Dataset | Nights | What it provides |
|---|---|---|
| [St. Vincent's / UCD Sleep Apnea DB (ucddb)](https://physionet.org/content/ucddb/1.0.0/) | 002, 003, 018, 025 | 14-channel PSG: EEG, EOG, EMG, ECG, flow, thoracic/abdominal belts, SpO2, snore, position. Includes technician-scored respiratory events and hypnograms. The nights range from normal to severe. |
| [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) | SC4001 | 22-h recording with EEG Fpz-Cz, EOG and EMG, plus an EDF+ hypnogram. Used as a **held-out** staging test; nothing was tuned on it. |

Download more nights into `data/ucddb/`. The app detects the `*_respevt.txt` / `*_stage.txt` / `*-Hypnogram.edf` annotation files automatically and compares against them.

## Pipeline

| Step | Module | What it does |
|---|---|---|
| Load | `psg/edf.py`, `psg/io.py` | Pure-NumPy EDF/EDF+ reader (memory-mapped, per-channel sampling rates, EDF+ annotations, works on `.rec`). Channels are mapped to roles (`flow`, `spo2`, `thorax`, …) whatever the lab named them. |
| Clean | `psg/preprocess.py` | Zero-phase Butterworth band/low/high-pass filters and 50 Hz notch. Flat-line and clipping masks. SpO2 cleaning: rejects values <50 % or >100 %, rejects jumps faster than 4 %/s, interpolates gaps ≤10 s and leaves longer gaps as NaN so no event is scored on missing data. |
| Stage | `psg/staging.py` | 30-s epochs. Features: relative band powers, slow-wave coverage (AASM 20 % rule), spindles, chin EMG tone, rapid eye movements. Soft rule memberships feed an HMM/Viterbi smoother. Recordings longer than 12 h get automatic lights-off/on detection. |
| Arousals | `psg/arousal.py` | Abrupt EEG frequency shift ≥3 s after ≥10 s of sleep, with EMG confirmation in REM. |
| Respiratory | `psg/respiratory.py` | See the rules below. Parameterised so any published rule version can be run; records per-event criteria, confidence and rejection reasons; `rescore()` re-applies new thresholds to cached signals in ~1 s. |
| Rule library | `psg/rules.py` | Seven published apnea/hypopnea definitions with citations, mapped to scorer parameters, plus the decision flowchart. |
| Oximetry | `psg/spo2.py` | ≥3 % desaturations, ODI, T90, nadir. |
| Interpret | `psg/pipeline.py` | AHI, OAHI/CAHI, REM/NREM AHI, supine/non-supine AHI, severity and diagnosis text, clinical flags (central-predominant, REM-related, positional, hypoxaemia, low sleep efficiency). |
| Report | `psg/report.py` | HTML report (hypnogram, event timeline, SpO2), event CSV, JSON. |
| Validate | `psg/evaluate.py`, `analyze.py` | Event matching against expert scoring, Cohen's κ for staging. |

### AASM respiratory rules implemented

- **Breath amplitude.** Peak-to-trough excursion of the band-limited flow signal, after subtracting the noise floor (cardiogenic oscillation).
- **Baseline.** Taken from the preceding 2 minutes: the median when breathing is stable, otherwise the mean of the largest breaths (the AASM rule for when there is no stable baseline). Detected events are excluded from the baseline.
- **Apnea.** Flow drops ≥90 % for ≥10 s.
  - *Obstructive*: effort continues on either belt.
  - *Central*: effort is absent.
  - *Mixed*: effort is absent at first, then resumes.
  - The belts are judged separately, because their sum cancels out during paradoxical breathing.
- **Hypopnea.** Flow drops ≥30 % for ≥10 s, **plus** either a ≥3 % desaturation or an arousal (AASM *recommended* rule). The *acceptable* CMS/Medicare rule (≥4 % desaturation only) can be selected in the toolbar.
  - Classified *obstructive* when snoring, inspiratory flattening, thoraco-abdominal paradox or preserved effort is present; otherwise *central*.
- **What is not counted.** Events during wake, events on bad or flat-lined flow, and runs longer than 180 s (sensor problem) are rejected, and the reason is recorded.
- **Severity.** AHI <5 normal, 5–15 mild, 15–30 moderate, ≥30 severe.

## Validation (fully automatic: automatic staging + automatic scoring)

| Night | Expert AHI | Automatic AHI | Severity (expert → auto) | Event sensitivity / precision | Staging κ |
|---|---|---|---|---|---|
| ucddb018 | 2.2 | 1.2 | Normal → Normal | – (only 9 events) | 0.76 |
| ucddb002 | 23.7 | 16.4 | Moderate → Moderate | 47 % / 77 % | 0.52 |
| ucddb003 | 51.3 | 46.1 | Severe → Severe | 71 % / 83 % | 0.64 |
| ucddb025 | 94.8 | 71.0 | Severe → Severe | 65 % / 84 % | 0.26 |
| SC4001 (held out) | – | – (no usable flow) | – | – | **0.66** |

Pooled over the four scored nights:

- **Severity class:** 4 of 4 correct.
- **AHI correlation:** r = 0.99.
- **Events:** 64 % sensitivity, 83 % precision.
- **Apnea vs hypopnea type:** ~80 % agreement.

These four nights were used to develop and tune the algorithm, so their numbers are optimistic.

### Blind test on unseen nights

Two more UCDDB nights (010 and 022) were processed exactly as a doctor's upload would be: plain `.edf` files, with no annotations next to them (`test_files/`). Only afterwards were the results compared with the technician's scoring (`test_files/answer_key/`).

| Night | Expert AHI | Automatic AHI (3 % rule) | Severity (expert → auto) | Event sensitivity / precision | Staging κ |
|---|---|---|---|---|---|
| ucddb010 → `unseen_night_A.edf` | 33.6 | 29.5 | Severe → **Moderate** (just under the 30 cut-off) | 34 % / 49 % | 0.32 |
| ucddb022 → `unseen_night_B.edf` | 7.0 | 15.6 | Mild → **Moderate** (over-called) | 48 % / 21 % | 0.67 |

The severity was wrong on both unseen nights, and event-level agreement is much lower than on the tuning nights. This is the more realistic estimate of current performance. Why:

- **The reference scoring does not follow the current AASM hypopnea rule.** The database documentation says stages were scored with the older Rechtschaffen & Kales rules and does not state the respiratory criteria. On night 022, only 55 % of the technician's hypopneas have a ≥3 % desaturation and only 4 are marked with an arousal.
- **Borderline events.** Many disagreements sit right at the thresholds (flow drop around 30 %, desaturation around 3 %), where small measurement differences flip the decision.
- **The rule choice matters.** Under the seven rule versions the automatic AHI for night B ranges from 3.6 (4 % rules, "normal") to 16.4 (3 %-or-arousal, "moderate") to 34.8 (Chicago 1999, "severe") against the technician's 7.0; the full table per rule and per confidence cut-off is in [docs/rule_versions.md](docs/rule_versions.md).

### Known limitations

- **Accuracy on unseen nights is limited** (see the blind test above). The algorithm needs validation on many more nights, ideally scored to the current AASM manual, before any clinical use.
- **AHI is underestimated on the tuning nights by 10–25 %.** Most missed events are borderline hypopneas. UCDDB has a single flow channel; AASM recommends a thermal sensor for apneas and nasal pressure for hypopneas.
- **Staging is weakest in very fragmented sleep** (ucddb025, κ 0.26), and N1 is rarely called. Human inter-scorer κ for N1 is only ~0.3–0.4.
- **Hypopnea obstructive/central subtyping** agrees only about half the time. The AASM marks it as optional, and it is shown with that caveat.
- **Arousal detection** is heuristic and has not been validated against expert arousal scoring.
- **Thresholds were tuned on 4 nights.** More nights, e.g. the rest of UCDDB, SHHS or MESA from NSRR, are needed before any clinical claim.
- **This is decision support, not a diagnostic device.** A sleep physician must review every report.

## GUI

- **Results header:** severity badge, diagnosis sentence, AHI, obstructive/central AHI, ODI, nadir SpO2, sleep time, arousal index, and how many events are clear-cut (≥ 90 % / ≥ 75 % / all), always visible above the signals.
- **Overview strip:** stage-coloured hypnogram, automatic vs technician event rug, SpO2 with the time below 90 % filled red. Click or drag to navigate.
- **Signal view:** coloured sleep-stage strip with epoch numbers above the traces, 30-s epoch grid, a time cursor across all channels, channel labels with units and amplitude scale, traces coloured by role. Hover an event for its summary; click it to select it, which highlights it, explains it and shows its path in the flowchart.
- **Signal view:** stacked channels with AASM display filters and 10 s to 10 min pages.
  - Colour-coded scored events on the respiratory channels, arousals shaded on the EEG, technician events drawn as bars for comparison.
  - "Show flow envelope" overlays the breath amplitude with the baseline and the 30 % / 90 % thresholds, so you can see *why* an event was scored.
- **Right panel:** channel picker (scales to 60+ signals), filterable event table (click to jump), report tab with diagnosis, indices, signal quality and agreement with the expert.
- **Keys:** `←/→` page, `N`/`P` next/previous event, `+`/`-` amplitude, `F5` analyse, `Ctrl+O` open.
- **Export:** HTML report + CSV + JSON.
