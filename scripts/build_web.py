"""Assemble the static browser version into ./public (used locally and mirrored in vercel.json).

    python scripts/build_web.py
    python -m http.server 8765 --directory public     # then open http://localhost:8765
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public"
SAMPLE = ROOT / "test_files" / "sample_known_ucddb003.edf"

if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(ROOT / "web", OUT)
(OUT / "psg").mkdir()
for py in (ROOT / "psg").glob("*.py"):
    shutil.copy2(py, OUT / "psg" / py.name)
if SAMPLE.exists():
    (OUT / "samples").mkdir()
    shutil.copy2(SAMPLE, OUT / "samples" / "sample_psg.edf")
print(f"Built {OUT} ({sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) / 1e6:.1f} MB)")
