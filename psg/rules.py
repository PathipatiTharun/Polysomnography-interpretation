"""Library of published adult apnea/hypopnea scoring rules, and the decision flowchart they share.

Each `RuleVersion` is one published definition, with the numbers taken from the cited source.
`RuleVersion.params()` turns it into `ScoringParams` for the scorer, so the same signal
processing can be run under any rule; the UI lets the physician pick one and adjust the
numbers with sliders.  `flowchart()` describes the decision steps for display; every check
the scorer records is keyed to a step id so the path an event took can be highlighted.

Sources (see docs/rule_versions.md for the full table):
  [1] AASM Task Force. Sleep-related breathing disorders in adults: recommendations for
      syndrome definition and measurement techniques in clinical research ("Chicago
      criteria"). Sleep 1999;22(5):667-689.
  [2] Iber C et al. The AASM Manual for the Scoring of Sleep and Associated Events, 1st ed.
      AASM, 2007.  Rules 4A (recommended) and 4B (alternative).
  [3] Ruehland WR et al. The new AASM criteria for scoring hypopneas: impact on the apnea
      hypopnea index. Sleep 2009;32(2):150-157  (quotes [1] and [2] verbatim).
  [4] Berry RB et al. Rules for scoring respiratory events in sleep: update of the 2007 AASM
      Manual. J Clin Sleep Med 2012;8(5):597-619, and AASM, "The 2007 AASM Scoring Manual vs.
      the AASM Scoring Manual v2.0" (October 2012).
  [5] AASM. "AASM clarifies hypopnea scoring criteria", 23 September 2013 (Manual v2.0.2:
      rule 1A recommended, rule 1B acceptable).
  [6] Sleep Heart Health Study scoring (NSRR variable ahi_a0h4); Nieto FJ et al. JAMA 2000.
  [7] AASM Manual v2.6 (2020) and later: adult respiratory rules unchanged from v2.0 / v2.0.2.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from .respiratory import ScoringParams


@dataclass(frozen=True)
class RuleVersion:
    id: str
    name: str
    year: int
    publisher: str
    apnea_drop: float                 # fraction of baseline; 0.90 = ">= 90 % drop"
    hypopnea_drop: float              # fraction of baseline
    min_duration: float               # seconds
    desat: float                      # % points required to confirm a hypopnea
    arousal: bool                     # EEG arousal accepted instead of desaturation
    hypopnea_drop_alone: float | None = None  # drop that counts with no desat/arousal (Chicago)
    amplitude_fraction: float = 0.0   # share of the event that must meet the drop (2007: 0.9)
    classify_hypopneas: bool = False  # obstructive / central hypopneas defined by the rule
    sensors: str = ""
    summary: str = ""
    notes: str = ""
    source: str = ""
    url: str = ""

    def params(self, base: ScoringParams | None = None) -> ScoringParams:
        """ScoringParams for this rule; engineering settings are kept from `base`."""
        b = base or ScoringParams()
        return dataclasses.replace(
            b, rule_id=self.id, apnea_drop=self.apnea_drop, hypopnea_drop=self.hypopnea_drop,
            min_duration=self.min_duration, hypopnea_desat=self.desat, hypopnea_arousal=self.arousal,
            hypopnea_drop_alone=self.hypopnea_drop_alone, amplitude_fraction=self.amplitude_fraction,
            classify_hypopneas=self.classify_hypopneas,
        )

    def short(self) -> str:
        alt = f" (or ≥{self.hypopnea_drop_alone * 100:.0f}% drop alone)" if self.hypopnea_drop_alone else ""
        conf = f"≥{self.desat:g}% desaturation" + (" or arousal" if self.arousal else "")
        return (f"Apnea ≥{self.apnea_drop * 100:.0f}% drop ≥{self.min_duration:.0f} s · "
                f"Hypopnea ≥{self.hypopnea_drop * 100:.0f}% drop ≥{self.min_duration:.0f} s + {conf}{alt}")


_RULES = [
    RuleVersion(
        id="chicago1999", name="AASM Task Force 1999 (\"Chicago criteria\")", year=1999,
        publisher="American Academy of Sleep Medicine task force, Sleep 22:667",
        apnea_drop=0.90, hypopnea_drop=0.30, min_duration=10.0, desat=3.0, arousal=True,
        hypopnea_drop_alone=0.50,
        sensors="Nasal pressure or thoraco-abdominal (RIP) sum",
        summary="Apnea: complete cessation of airflow ≥10 s. Hypopnea: >50 % decrease in a valid measure "
                "of breathing ≥10 s, OR a clear but smaller decrease with ≥3 % desaturation or an arousal.",
        notes="'Complete cessation' is operationalised as a ≥90 % drop (sensor noise never reaches zero). "
              "'Clear decrease' below 50 % is operationalised as ≥30 %. First attempt at a common definition; "
              "gives the highest AHI of all rules.",
        source="[1] Sleep 1999;22(5):667-689, as quoted in [3] Ruehland et al. Sleep 2009",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2635578",
    ),
    RuleVersion(
        id="aasm2007_rec", name="AASM Manual 2007, rule 4A (recommended)", year=2007,
        publisher="AASM Manual for the Scoring of Sleep and Associated Events, 1st edition",
        apnea_drop=0.90, hypopnea_drop=0.30, min_duration=10.0, desat=4.0, arousal=False,
        amplitude_fraction=0.90,
        sensors="Apnea: oronasal thermal sensor. Hypopnea: nasal pressure",
        summary="Hypopnea: nasal pressure excursions drop ≥30 % from baseline for ≥10 s with ≥4 % "
                "desaturation; at least 90 % of the event's duration must meet the amplitude reduction. "
                "Arousals do not count.",
        notes="Strictest common rule; roughly 30 % of the Chicago AHI in Ruehland 2009. Same numbers as the "
              "CMS/Medicare requirement.",
        source="[2] AASM Manual 2007, quoted in [3]",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2635578",
    ),
    RuleVersion(
        id="aasm2007_alt", name="AASM Manual 2007, rule 4B (alternative)", year=2007,
        publisher="AASM Manual for the Scoring of Sleep and Associated Events, 1st edition",
        apnea_drop=0.90, hypopnea_drop=0.50, min_duration=10.0, desat=3.0, arousal=True,
        amplitude_fraction=0.90,
        sensors="Apnea: oronasal thermal sensor. Hypopnea: nasal pressure",
        summary="Hypopnea: nasal pressure excursions drop ≥50 % for ≥10 s with ≥3 % desaturation or an "
                "arousal; ≥90 % of the event's duration must meet the amplitude reduction.",
        notes="About 60 % of the Chicago AHI (Ruehland 2009).",
        source="[2] AASM Manual 2007, quoted in [3]",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2635578",
    ),
    RuleVersion(
        id="aasm2012_rec", name="AASM Manual v2.0 (2012), rule 1A (recommended)", year=2012,
        publisher="AASM Sleep Apnea Definitions Task Force (Berry et al.)",
        apnea_drop=0.90, hypopnea_drop=0.30, min_duration=10.0, desat=3.0, arousal=True,
        classify_hypopneas=True,
        sensors="Apnea: oronasal thermal sensor. Hypopnea: nasal pressure. Effort: RIP belts",
        summary="Apnea: peak excursion drops ≥90 % of pre-event baseline for ≥10 s. Hypopnea: drop ≥30 % "
                "for ≥10 s with ≥3 % desaturation or an arousal. Obstructive/central hypopnea "
                "classification defined (snoring, flattening, paradox). Event counts if it begins or ends "
                "in a sleep epoch.",
        notes="Merged the 2007 recommended/alternative rules into one; removed the '90 % of duration' "
              "requirement. Remains the AASM recommended rule.",
        source="[4] Berry et al. JCSM 2012;8(5):597-619; AASM v2.0 summary of updates",
        url="https://aasm.org/wp-content/uploads/2017/11/Summary-of-Updates-in-v2.0-FINAL.pdf",
    ),
    RuleVersion(
        id="aasm2013_acc", name="AASM Manual v2.0.2 (2013), rule 1B (acceptable) / CMS", year=2013,
        publisher="AASM; matches the CMS (Medicare) coverage definition",
        apnea_drop=0.90, hypopnea_drop=0.30, min_duration=10.0, desat=4.0, arousal=False,
        classify_hypopneas=True,
        sensors="As v2.0",
        summary="Hypopnea: drop ≥30 % for ≥10 s with ≥4 % desaturation; arousals excluded. Reinstated in "
                "2013 so that labs can report the AHI insurers require.",
        notes="Gives a lower AHI than rule 1A; patients with arousal-based events may fall below the "
              "treatment threshold under this rule.",
        source="[5] AASM, 'AASM clarifies hypopnea scoring criteria', 23 Sep 2013",
        url="https://aasm.org/aasm-clarifies-hypopnea-scoring-criteria/",
    ),
    RuleVersion(
        id="shhs", name="Sleep Heart Health Study (research definition, AHI a0h4)", year=2000,
        publisher="NHLBI Sleep Heart Health Study; National Sleep Research Resource",
        apnea_drop=0.90, hypopnea_drop=0.30, min_duration=10.0, desat=4.0, arousal=False,
        sensors="Thermistor airflow or thoraco-abdominal excursion (RIP)",
        summary="Hypopnea: ≥30 % reduction in airflow or thoraco-abdominal excursion for ≥10 s with ≥4 % "
                "desaturation. Apnea: cessation or near-cessation of airflow ≥10 s, no desaturation "
                "required (the 'a0h4' variant).",
        notes="The epidemiological definition behind the CMS 4 % rule. NSRR also publishes 3 % variants "
              "(ahi_a0h3) of the same recordings.",
        source="[6] NSRR SHHS variable ahi_a0h4; Nieto et al. JAMA 2000;283:1829",
        url="https://www.sleepdata.org/datasets/shhs/variables/ahi_a0h4",
    ),
    RuleVersion(
        id="aasm_current", name="AASM Manual v2.6 (2020) and later – rule 1A", year=2020,
        publisher="AASM Manual for the Scoring of Sleep and Associated Events",
        apnea_drop=0.90, hypopnea_drop=0.30, min_duration=10.0, desat=3.0, arousal=True,
        classify_hypopneas=True,
        sensors="As v2.0; PAP device flow during titration",
        summary="Adult apnea/hypopnea thresholds unchanged from v2.0: ≥90 % / ≥30 % drops for ≥10 s, "
                "hypopnea confirmed by ≥3 % desaturation or arousal (1A) with the 4 % rule kept as "
                "acceptable (1B).",
        notes="Later versions refined sensors, PAP titration and pediatric rules rather than the adult "
              "thresholds. Listed separately so the 'current manual' can be selected explicitly.",
        source="[7] AASM Manual v2.6 (2020); rule text identical to [4]/[5]",
        url="https://aasm.org/clinical-resources/scoring-manual/",
    ),
]

RULES: dict[str, RuleVersion] = {r.id: r for r in _RULES}
DEFAULT_RULE = "aasm2012_rec"


def get_rule(rule_id: str) -> RuleVersion:
    if rule_id not in RULES:
        raise KeyError(f"Unknown rule version '{rule_id}'. Known: {', '.join(RULES)}")
    return RULES[rule_id]


# ----------------------------------------------------------------------------- flowchart

@dataclass
class Step:
    id: str
    text: str
    kind: str = "decision"           # 'decision' | 'terminal' | 'start'
    yes: str | None = None           # next step id
    no: str | None = None
    outcome: str = ""                # for terminals: 'AO','AC','AM','HO','HC' or 'reject'
    branches: dict[str, str] = field(default_factory=dict)  # multi-way (effort): label -> step id


def flowchart(p: ScoringParams) -> list[Step]:
    """Decision steps of the scorer under parameters `p`, in display order."""
    confirm = f"SpO2 falls ≥{p.hypopnea_desat:g}%" + (" or EEG arousal" if p.hypopnea_arousal else "")
    if p.hypopnea_drop_alone:
        confirm += f"\n(or airflow drop ≥{p.hypopnea_drop_alone * 100:.0f}% alone)"
    frac = (f"\n(≥{p.amplitude_fraction * 100:.0f}% of the event below threshold)" if p.amplitude_fraction else "")
    steps = [
        Step("start", "Breath amplitude vs.\n2-min pre-event baseline", kind="start", yes="sleep"),
        Step("sleep", "Event begins or ends\nin a sleep epoch?", yes="signal", no="reject_wake"),
        Step("signal", "Airflow signal usable\n(< 30% artifact)?", yes="apnea", no="reject_signal"),
        Step("apnea", f"Airflow drop ≥{p.apnea_drop * 100:.0f}%\nfor ≥{p.min_duration:.0f} s?", yes="effort", no="hypopnea"),
        Step("hypopnea", f"Airflow drop ≥{p.hypopnea_drop * 100:.0f}%\nfor ≥{p.min_duration:.0f} s?{frac}", yes="confirm", no="reject_drop"),
        Step("confirm", confirm + "?", yes="hyp_type" if p.classify_hypopneas else "H", no="reject_confirm"),
        Step("effort", "Effort on chest /\nabdomen belts?",
             branches={"continues": "AO", "absent": "AC", "absent then resumes": "AM"}),
        Step("AO", "Obstructive apnea", kind="terminal", outcome="AO"),
        Step("AC", "Central apnea", kind="terminal", outcome="AC"),
        Step("AM", "Mixed apnea", kind="terminal", outcome="AM"),
        Step("reject_wake", "Not scored:\nentirely in wake", kind="terminal", outcome="reject"),
        Step("reject_signal", "Not scored:\nbad signal", kind="terminal", outcome="reject"),
        Step("reject_drop", "Normal breathing", kind="terminal", outcome="reject"),
        Step("reject_confirm", "Not scored:\nno desaturation / arousal", kind="terminal", outcome="reject"),
    ]
    if p.classify_hypopneas:
        steps += [
            Step("hyp_type", "Snoring, flow flattening,\nparadox or preserved effort?", yes="HO", no="HC"),
            Step("HO", "Obstructive hypopnea", kind="terminal", outcome="HO"),
            Step("HC", "Central hypopnea", kind="terminal", outcome="HC"),
        ]
    else:
        steps.append(Step("H", "Hypopnea", kind="terminal", outcome="HO"))
    return steps
