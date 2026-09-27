"""Automatic sleep annotations and visualisation for a recording.

    python annotate.py data/sleepedf/SC4001E0-PSG.edf
    python annotate.py data/ucddb/ucddb003.rec --out reports/annotations

Writes to <out>/<recording>/:
    <name>_auto-annotations.edf   EDF+ annotation file (opens next to the recording in any EDF viewer)
    <name>_annotations.csv        the same annotations as a table
    <name>_annotations.html       summary, whole-night overview, example epoch per stage, agreement
    <name>_overview.png / _examples.png / _agreement.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from psg.annotate import annotate_recording, write_csv, write_edfplus
from psg.io import load_recording
from psg.visualize import write_page


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="reports/annotations")
    args = ap.parse_args(argv)
    for f in args.files:
        t0 = time.time()
        rec = load_recording(f)
        res = annotate_recording(rec)
        out = Path(args.out) / rec.name
        out.mkdir(parents=True, exist_ok=True)
        edf = out / f"{rec.name}_auto-annotations.edf"
        csv = out / f"{rec.name}_annotations.csv"
        write_edfplus(res, edf)
        write_csv(res, csv)
        page = write_page(res, out, {edf.name: "EDF+ annotations (open with the recording in EDFbrowser / MNE)",
                                     csv.name: "CSV table of all annotations"})
        s = res.summary
        print(f"=== {rec.name}: {len(res.annotations)} annotations in {time.time() - t0:.0f} s")
        print(f"  sleep {s['tst_min']:.0f} min, efficiency {s['sleep_efficiency']:.0f} %, "
              f"N1/N2/N3/R {s['pct_n1']:.0f}/{s['pct_n2']:.0f}/{s['pct_n3']:.0f}/{s['pct_rem']:.0f} %")
        print(f"  spindles {s['n_spindles']}, slow waves {s['n_slow_waves']}, REMs {s['n_rems']}, arousals {s['n_arousals']}")
        print(f"  page: {page.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
