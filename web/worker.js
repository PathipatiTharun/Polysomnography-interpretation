// Web Worker: loads Pyodide + the psg package once, then analyses uploaded EDF files.
// Everything runs locally in the browser; the recording is never sent to a server.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js");

const PSG_MODULES = ["__init__", "edf", "io", "preprocess", "respiratory", "spo2", "staging",
                     "arousal", "pipeline", "report", "evaluate", "study"];

self.psgProgress = (pct, msg) => self.postMessage({ type: "progress", pct, msg: String(msg) });

const ready = (async () => {
  self.postMessage({ type: "boot", msg: "Downloading the Python runtime…" });
  const py = await loadPyodide();
  self.postMessage({ type: "boot", msg: "Loading NumPy, SciPy and Matplotlib…" });
  await py.loadPackage(["numpy", "scipy", "matplotlib"]);
  self.postMessage({ type: "boot", msg: "Loading the PSG analysis code…" });
  py.FS.mkdirTree("/app/psg");
  py.FS.mkdirTree("/work");
  const bust = "?v=" + Date.now();
  for (const m of PSG_MODULES) {
    const r = await fetch(`psg/${m}.py${bust}`);
    if (!r.ok) throw new Error(`Could not load psg/${m}.py (${r.status})`);
    py.FS.writeFile(`/app/psg/${m}.py`, await r.text());
  }
  const runner = await fetch(`runner.py${bust}`);
  py.FS.writeFile("/app/runner.py", await runner.text());
  py.runPython("import sys; sys.path.insert(0, '/app')");
  py.runPython("import matplotlib; matplotlib.use('Agg')");
  py.runPython("from runner import run_study");
  self.postMessage({ type: "ready" });
  return py;
})().catch((err) => {
  self.postMessage({ type: "fatal", msg: String(err && err.message || err) });
  throw err;
});

self.onmessage = async (e) => {
  const { file, rule } = e.data;
  let py;
  try {
    py = await ready;
  } catch (_) {
    return;
  }
  try {
    const safe = file.name.replace(/[^A-Za-z0-9_.-]+/g, "_");
    const path = `/work/${safe}`;
    self.postMessage({ type: "progress", pct: 0, msg: "Reading file into memory" });
    py.FS.writeFile(path, new Uint8Array(await file.arrayBuffer()));
    const run = py.globals.get("run_study");
    const out = run(path, rule);
    run.destroy();
    self.postMessage({ type: "result", payload: JSON.parse(out) });
  } catch (err) {
    self.postMessage({ type: "result", payload: { ok: false, error: "The analysis failed in the browser.",
                                                   detail: String(err && err.message || err) } });
  }
};
