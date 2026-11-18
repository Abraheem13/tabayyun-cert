"""Build the real multi-source tier from the Flight data set of Li et al. (PVLDB 2012, "Truth Finding on the
Deep Web: Is the Problem Solved?"), distributed at https://www.lunadong.com/fusionDataSets.htm.

38 web sources report scheduled and actual departure and arrival times of about 1,200 flights per day
(December 2011); airline websites provide the gold standard for 100 flights per day.  A fact is
(day, flight, attribute); its value is the reported time, normalized to minutes of the day and
quantized to 10-minute bins (the time zone suffix is ignored; sources report local time).

Output: data/fusion/flight_facts.json = {day: [{"fact": id, "gold": bin, "reports": {source: bin}}]}
Usage: python scripts/prep_fusion.py [--dir DIR_WITH_ZIPS]
"""
import argparse
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.lunadong.com/datasets/"
ATTR = {2: "sched_dep", 3: "actual_dep", 5: "sched_arr", 6: "actual_arr"}
TIME = re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]?\.?")
BIN = 10


def minutes(s):
    m = TIME.search(s or "")
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    if h > 12 or mi > 59:
        return None
    h = h % 12 + (12 if ap == "p" else 0)
    return h * 60 + mi


def to_bin(t):
    return ((t + BIN // 2) // BIN) % (1440 // BIN)


def load_zip(name, d):
    if d and (Path(d) / name).exists():
        return zipfile.ZipFile(Path(d) / name)
    return zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL + name, timeout=120).read()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    a = ap.parse_args()
    data, truth = load_zip("clean_flight.zip", a.dir), load_zip("flight_truth.zip", a.dir)
    days = {}
    for name in truth.namelist():
        if not name.endswith("-truth.txt"):
            continue
        day = Path(name).name.replace("-truth.txt", "")
        gold = {}
        for line in truth.read(name).decode("latin-1").splitlines():
            p = line.split("\t")
            if len(p) < 7:
                continue
            for col, att in ATTR.items():
                t = minutes(p[col - 1])                  # truth files have no source column
                if t is not None:
                    gold[(p[0].strip(), att)] = to_bin(t)
        dname = f"clean_flight/{day}-data.txt"
        if dname not in data.namelist():
            continue
        reps = {}
        for line in data.read(dname).decode("latin-1").splitlines():
            p = line.split("\t")
            if len(p) < 7:
                continue
            src, fl = p[0].strip(), p[1].strip()
            for col, att in ATTR.items():
                if (fl, att) not in gold:
                    continue
                t = minutes(p[col]) if col < len(p) else None
                if t is not None:
                    reps.setdefault((fl, att), {})[src] = to_bin(t)
        days[day] = [{"fact": f"{fl}|{att}", "gold": g, "reports": reps.get((fl, att), {})}
                     for (fl, att), g in sorted(gold.items()) if reps.get((fl, att))]
    out = ROOT / "data" / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    (out / "flight_facts.json").write_text(json.dumps(days))
    facts = [x for v in days.values() for x in v]
    cov = Counter(len(x["reports"]) for x in facts)
    vote = sum(Counter(x["reports"].values()).most_common(1)[0][0] == x["gold"] for x in facts)
    srcs = Counter(s for x in facts for s in x["reports"])
    print(f"days={len(days)} facts={len(facts)} sources={len(srcs)} vote_accuracy={vote / len(facts):.3f}")
    print("sources per fact (count: facts):", dict(sorted(cov.items())))


if __name__ == "__main__":
    main()
