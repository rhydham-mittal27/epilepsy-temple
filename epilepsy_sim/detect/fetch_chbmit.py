"""Download a CHB-MIT subset (PhysioNet, open access) and parse seizure annotations.

CHB-MIT Scalp EEG Database: Shoeb AH (2009), MIT PhD thesis; Goldberger et al. (2000) PhysioNet.
https://physionet.org/content/chbmit/1.0.0/

Downloads every seizure-containing file plus N evenly spaced seizure-free files per patient,
so that sensitivity AND false alarms per 24 h can both be estimated.

Usage:
    uv run detect/fetch_chbmit.py                       # chb01, chb03, chb05, 6 seizure-free files each
    uv run detect/fetch_chbmit.py --patients chb01 chb02 --n-free 10
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

# PhysioNet open-data mirror on AWS (same files, much faster than physionet.org)
BASE = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"
DATA = Path(__file__).resolve().parent.parent / "data" / "chbmit"


def parse_summary(text):
    """Return {file: [(start_s, end_s), ...]} for every file in a chbXX-summary.txt."""
    files = {}
    cur = None
    starts = []
    for line in text.splitlines():
        if line.startswith("File Name:"):
            cur = line.split(":", 1)[1].strip()
            files[cur] = []
            starts = []
        elif cur and re.search(r"Seizure\s*\d*\s*Start Time", line):
            starts.append(int(re.findall(r"(\d+)\s*seconds", line)[0]))
        elif cur and re.search(r"Seizure\s*\d*\s*End Time", line):
            files[cur].append((starts.pop(0), int(re.findall(r"(\d+)\s*seconds", line)[0])))
    return files


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return "downloaded"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=["chb01", "chb03", "chb05"])
    ap.add_argument("--n-free", type=int, default=6, help="seizure-free files per patient")
    args = ap.parse_args()

    manifest = {}
    for p in args.patients:
        pdir = DATA / p
        pdir.mkdir(parents=True, exist_ok=True)
        fetch(f"{BASE}/{p}/{p}-summary.txt", pdir / f"{p}-summary.txt")
        ann = parse_summary((pdir / f"{p}-summary.txt").read_text())
        sz = [f for f, s in ann.items() if s]
        free = [f for f, s in ann.items() if not s]
        step = max(len(free) // args.n_free, 1)
        chosen_free = free[::step][: args.n_free]
        for f in sz + chosen_free:
            status = fetch(f"{BASE}/{p}/{f}", pdir / f)
            print(f"{p}/{f}: {status}", flush=True)
        manifest[p] = {f: ann[f] for f in sz + chosen_free}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_sz = sum(len(v) for m in manifest.values() for v in m.values())
    print(f"done: {sum(len(m) for m in manifest.values())} files, {n_sz} seizures -> {DATA}")


if __name__ == "__main__":
    main()
