#!/usr/bin/env python3
"""
Dataset Balance Comparison Script
Compares two or more folder JSONs from analyze_dataset.py output.
You decide what to compare against what.

Usage:
  # Compare one folder against another
  python compare_datasets.py raw_metadata/natural.json raw_metadata/suno.json

  # Compare one folder against multiple
  python compare_datasets.py raw_metadata/natural.json raw_metadata/suno.json raw_metadata/boomy.json

  # Compare all JSONs in a directory — first one is the reference
  python compare_datasets.py raw_metadata/natural.json raw_metadata/

The first argument is always the REFERENCE (e.g. natural/human).
All subsequent arguments are compared against it.

Dependencies:
  pip install numpy
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


# Model 3-channel bands — must match architecture config
BANDS = {
    "low":  (20,   500),
    "mid":  (300,  4000),
    "high": (3000, 11025),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    path = Path(path)
    if not path.exists():
        print("ERROR: not found: " + str(path))
        sys.exit(1)
    data = json.loads(path.read_text())
    ok   = [r for r in data if r.get("quality_flag") == "ok"]
    print("  Loaded " + path.stem + ": " +
          str(len(data)) + " total, " + str(len(ok)) + " ok")
    return path.stem, ok


def get_vals(results, key):
    return [r["librosa"][key] for r in results
            if r.get("librosa") and r["librosa"].get(key) is not None]


def ms(vals):
    if not vals:
        return None, None
    clean = [v for v in vals if v is not None]
    if not clean:
        return None, None
    return float(np.mean(clean)), float(np.std(clean))


# ── Per-folder summary ────────────────────────────────────────────────────────

def print_summary(name, results):
    sep = "-" * 65
    print("\n" + sep)
    print("  [" + name + "]  " + str(len(results)) + " ok files")
    print(sep)

    durs = [r["ffprobe"]["duration"] for r in results if r.get("ffprobe")]
    if durs:
        print("  Duration: " + str(round(sum(durs)/3600, 1)) + "h" +
              "  mean " + str(round(np.mean(durs)/60, 1)) + "min")

    brs = [r["ffprobe"]["bitrate_kbps"] for r in results
           if r.get("ffprobe") and r["ffprobe"].get("bitrate_kbps")]
    if brs:
        print("  Bitrate: mean " + str(round(np.mean(brs))) +
              "kbps  min " + str(min(brs)) + "  max " + str(max(brs)))

    src = defaultdict(int)
    for r in results:
        sr = r.get("ffprobe", {}).get("sample_rate")
        if sr:
            src[sr] += 1
    print("  Sample rates: " +
          "  ".join(str(k) + "Hz:" + str(v) for k, v in sorted(src.items())))

    print("  Band energies:")
    for band, (lo, hi) in BANDS.items():
        vals = get_vals(results, band + "_energy_ratio")
        if vals:
            m, s = ms(vals)
            print("    " + (band + "(" + str(lo) + "-" + str(hi) + "Hz)").ljust(22) +
                  "  mean " + str(round(m, 3)) + "  std " + str(round(s, 3)))

    crest = get_vals(results, "crest_factor_db")
    if crest:
        m, s = ms(crest)
        heavy = sum(1 for v in crest if v < 6)
        print("  Crest factor: mean " + str(round(m, 1)) + "dB" +
              "  std " + str(round(s, 1)) + "dB" +
              "  <6dB: " + str(heavy) + " (" + str(round(heavy/len(crest)*100, 1)) + "%)")

    tempos = get_vals(results, "tempo_bpm")
    if tempos:
        m, s = ms(tempos)
        print("  Tempo: mean " + str(round(m, 1)) + " BPM  std " + str(round(s, 1)))

    vocals = get_vals(results, "has_vocals_rough")
    if vocals:
        print("  Vocals: " + str(round(np.mean(vocals)*100, 1)) + "%")


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(ref_name, ref_results, cmp_name, cmp_results):
    sep = "=" * 65
    print("\n" + sep)
    print("  COMPARE: " + ref_name + "  vs  " + cmp_name)
    print("  ref=" + str(len(ref_results)) + " files" +
          "  cmp=" + str(len(cmp_results)) + " files")
    print(sep)
    print("  Feature                        " + ref_name[:12].rjust(12) +
          "  " + cmp_name[:12].rjust(12) + "      Diff  Flag")
    print("  " + "-" * 63)

    checks = [
        ("low_energy_ratio",  "low(20-500Hz)",   0.05),
        ("mid_energy_ratio",  "mid(300-4kHz)",   0.05),
        ("high_energy_ratio", "high(3k-11kHz)",  0.05),
        ("crest_factor_db",   "crest_factor_db", 3.0),
        ("tempo_bpm",         "tempo_bpm",        15.0),
        ("zcr_mean",          "zcr_mean",         0.02),
        ("flatness",          "flatness",         0.01),
        ("rolloff_95_hz",     "rolloff_95_hz",    500.0),
    ]

    confounds = []

    for key, label, thresh in checks:
        rv = get_vals(ref_results, key)
        cv = get_vals(cmp_results, key)
        if not rv or not cv:
            continue
        rm, _ = ms(rv)
        cm, _ = ms(cv)
        diff  = abs(rm - cm)
        flag  = "CONFOUND" if diff > thresh else "ok"
        if flag == "CONFOUND":
            confounds.append(label)
        print("  " + label.ljust(28) +
              str(round(rm, 3)).rjust(14) + "  " +
              str(round(cm, 3)).rjust(14) + "  " +
              str(round(diff, 3)).rjust(8) + "  " + flag)

    # vocals
    rv = get_vals(ref_results, "has_vocals_rough")
    cv = get_vals(cmp_results, "has_vocals_rough")
    if rv and cv:
        rm   = np.mean(rv) * 100
        cm   = np.mean(cv) * 100
        diff = abs(rm - cm)
        flag = "CONFOUND" if diff > 15 else "ok"
        if flag == "CONFOUND":
            confounds.append("has_vocals_%")
        print("  " + "has_vocals_%".ljust(28) +
              str(round(rm, 1)).rjust(14) + "  " +
              str(round(cm, 1)).rjust(14) + "  " +
              str(round(diff, 1)).rjust(8) + "  " + flag)

    if confounds:
        print("\n  " + str(len(confounds)) + " confound(s): " + ", ".join(confounds))
        print("  Fix: balance dataset or add targeted augmentation")
    else:
        print("\n  No significant confounds detected.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python compare_datasets.py reference.json compare.json [compare2.json ...]")
        print("  python compare_datasets.py reference.json folder_with_jsons/")
        print("")
        print("First argument is always the reference (e.g. natural.json).")
        sys.exit(1)

    # load reference
    ref_name, ref_results = load_json(sys.argv[1])

    # load comparison targets
    compare_targets = []
    for arg in sys.argv[2:]:
        p = Path(arg)
        if p.is_dir():
            # load all JSONs in directory except reference
            for jf in sorted(p.glob("*.json")):
                if jf.stem != ref_name:
                    compare_targets.append(load_json(jf))
        elif p.suffix == ".json":
            compare_targets.append(load_json(p))

    if not compare_targets:
        print("No comparison targets found.")
        sys.exit(1)

    print("\n  Reference: " + ref_name + " (" + str(len(ref_results)) + " ok files)")
    print("  Comparing against: " + ", ".join(n for n, _ in compare_targets))

    # print summary for reference
    print_summary(ref_name, ref_results)

    # print summary and comparison for each target
    for cmp_name, cmp_results in compare_targets:
        print_summary(cmp_name, cmp_results)
        compare(ref_name, ref_results, cmp_name, cmp_results)

    print("\n  Done.")


if __name__ == "__main__":
    main()