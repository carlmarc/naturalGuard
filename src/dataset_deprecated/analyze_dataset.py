#!/usr/bin/env python3
"""
Dataset Audio Analysis Script
Features aligned with model 3-channel input (low/mid/high bands).
Saves one JSON per folder. Resumes automatically if interrupted.

Usage:
  python analyze_dataset.py /path/to/dataset/

Dependencies:
  pip install librosa numpy
  ffprobe (comes with ffmpeg)
"""

import sys
import json
import subprocess
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

try:
    import librosa
except ImportError:
    print("ERROR: pip install librosa")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

AUDIO_EXT = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".aiff", ".opus"}

# Model 3-channel bands — must match architecture config
# Nyquist at 22050Hz = 11025Hz
BANDS = {
    "low":  (20,   500),
    "mid":  (300,  4000),
    "high": (3000, 11025),
}

ANALYSIS_SR      = 22050
SEG_SEC          = 7.0
SEG_OVERLAP      = 0.5
MIN_BITRATE      = 128     # kbps
MIN_DURATION     = 10.0    # seconds
SILENCE_DB       = -40.0
LOAD_DURATION    = 120     # seconds per file (None = full track)
SAVE_EVERY       = 50      # save after every N new files


# ── FFprobe ───────────────────────────────────────────────────────────────────

def ffprobe(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        d   = json.loads(out.stdout)
        s   = next((x for x in d.get("streams", [])
                    if x.get("codec_type") == "audio"), None)
        if not s:
            return None
        fmt  = d.get("format", {})
        tags = fmt.get("tags", {})
        bps  = int(s.get("bit_rate", 0)) or int(fmt.get("bit_rate", 0))
        enc  = tags.get("encoder", tags.get("ENCODER", "")).strip()
        return {
            "ext":          Path(path).suffix.lower(),
            "codec":        s.get("codec_name", "unknown"),
            "bitrate_kbps": round(bps / 1000) if bps else None,
            "sample_rate":  int(s.get("sample_rate", 0)) or None,
            "channels":     s.get("channels"),
            "duration":     float(fmt.get("duration", 0) or 0),
            "encoder":      enc,
            "is_lame":      "LAME" in enc.upper(),
        }
    except Exception:
        return None


# ── Features ──────────────────────────────────────────────────────────────────

def band_energies(audio, sr):
    S     = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = S.sum() + 1e-10
    out   = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs <= hi)
        out[name + "_energy_ratio"] = float(S[m].sum() / total)
    return out


def track_features(audio, sr):
    f   = {}
    rms = librosa.feature.rms(y=audio)[0]
    rm  = rms.mean()

    f["rms_db"]          = float(librosa.amplitude_to_db(np.array([rm]))[0])
    f["rms_db_std"]      = float(np.std(librosa.amplitude_to_db(rms + 1e-10)))
    peak                 = np.abs(audio).max()
    f["crest_factor_db"] = float(20 * np.log10(peak / (rm + 1e-10) + 1e-10))

    f["rolloff_85_hz"]  = float(librosa.feature.spectral_rolloff(
        y=audio, sr=sr, roll_percent=0.85).mean())
    f["rolloff_95_hz"]  = float(librosa.feature.spectral_rolloff(
        y=audio, sr=sr, roll_percent=0.95).mean())
    f["flatness"]       = float(librosa.feature.spectral_flatness(y=audio).mean())
    f["bandwidth_hz"]   = float(librosa.feature.spectral_bandwidth(
        y=audio, sr=sr).mean())

    zcr         = librosa.feature.zero_crossing_rate(audio)
    f["zcr_mean"] = float(zcr.mean())
    f["zcr_std"]  = float(zcr.std())

    try:
        tempo, _    = librosa.beat.beat_track(y=audio, sr=sr)
        f["tempo_bpm"] = float(tempo)
    except Exception:
        f["tempo_bpm"] = None

    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i, v in enumerate(mfcc.mean(axis=1)):
            f["mfcc_" + str(i+1)] = float(v)
    except Exception:
        pass

    be = band_energies(audio, sr)
    f.update(be)
    f["has_vocals_rough"]    = bool(be["mid_energy_ratio"] > 0.45)
    f["high_freq_low_flag"]  = bool(be["high_energy_ratio"] < 0.03)
    return f


def segment_features(audio, sr):
    hop = int(SEG_SEC * (1 - SEG_OVERLAP) * sr)
    seg = int(SEG_SEC * sr)
    segs, start = [], 0
    while start + seg <= len(audio):
        s      = audio[start:start + seg]
        rms    = librosa.feature.rms(y=s)[0].mean()
        rdb    = float(librosa.amplitude_to_db(np.array([rms]))[0])
        be     = band_energies(s, sr)
        segs.append({"rdb": rdb, "silent": rdb < SILENCE_DB, **be})
        start += hop
    if not segs:
        return {}
    ns     = [s for s in segs if not s["silent"]]
    sc     = len(segs) - len(ns)
    out    = {
        "total":         len(segs),
        "silent":        sc,
        "usable":        len(ns),
        "silence_ratio": sc / len(segs),
    }
    if ns:
        for b in ["low", "mid", "high"]:
            k            = b + "_energy_ratio"
            vals         = [s[k] for s in ns]
            out["seg_" + b + "_mean"] = float(np.mean(vals))
            out["seg_" + b + "_std"]  = float(np.std(vals))
    return out


# ── Quality ───────────────────────────────────────────────────────────────────

def quality(fp, lf):
    reasons = []
    br = fp.get("bitrate_kbps")
    if br and br < MIN_BITRATE:
        reasons.append("bitrate_" + str(br) + "kbps_below_" + str(MIN_BITRATE))
    dur = fp.get("duration", 0)
    if dur < MIN_DURATION:
        reasons.append("too_short_" + str(round(dur, 1)) + "s")
    if lf and lf.get("rms_db", 0) < SILENCE_DB:
        reasons.append("near_silence")
    sr = fp.get("sample_rate")
    if sr and sr <= 16000:
        reasons.append("low_sample_rate_" + str(sr) + "Hz")
    return ("ok" if not reasons else "excluded"), reasons


# ── Per-file ──────────────────────────────────────────────────────────────────

def analyze_file(path):
    r = {"filepath": str(path), "filename": path.name}

    fp = ffprobe(path)
    if fp is None:
        r["quality_flag"]      = "error"
        r["exclusion_reasons"] = ["ffprobe_failed"]
        return r
    r["ffprobe"] = fp

    lf = sf = None
    try:
        audio, sr = librosa.load(str(path), sr=ANALYSIS_SR,
                                 mono=True, duration=LOAD_DURATION)
        lf = track_features(audio, sr)
        sf = segment_features(audio, sr)
    except Exception as e:
        lf = {"error": str(e)}

    r["librosa"]  = lf
    r["segments"] = sf
    flag, reasons = quality(fp, lf)
    r["quality_flag"]      = flag
    r["exclusion_reasons"] = reasons
    return r


# ── Folder scan ───────────────────────────────────────────────────────────────

def analyze_folder(folder_path, out_dir):
    folder_path = Path(folder_path)
    out_dir     = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    out_path    = out_dir / (folder_path.name + ".json")

    # resume
    existing = {}
    if out_path.exists():
        try:
            saved    = json.loads(out_path.read_text())
            existing = {r["filepath"]: r for r in saved}
            print("  Resuming — " + str(len(existing)) + " files already done.")
        except Exception:
            existing = {}

    all_files = sorted([f for f in folder_path.rglob("*")
                        if f.suffix.lower() in AUDIO_EXT and f.is_file()])
    results   = list(existing.values())
    done      = set(existing.keys())
    todo      = [f for f in all_files if str(f) not in done]
    new_n     = 0

    print("\n  " + str(len(todo)) + " new files in '" + folder_path.name +
          "' (" + str(len(done)) + " done, " + str(len(all_files)) + " total)")

    for i, fp in enumerate(todo):
        if i % 10 == 0 and i > 0:
            pct = (len(done) + i) / len(all_files) * 100
            print("    " + str(len(done)+i) + "/" + str(len(all_files)) +
                  " (" + str(round(pct, 1)) + "%)...", end="\r")
        try:
            results.append(analyze_file(fp))
        except Exception as e:
            results.append({"filepath": str(fp),
                            "quality_flag": "error",
                            "exclusion_reasons": [str(e)]})
        new_n += 1
        if new_n % SAVE_EVERY == 0:
            out_path.write_text(json.dumps(results, indent=2, default=str))
            print("    [" + str(len(results)) + " saved]              ", end="\r")

    out_path.write_text(json.dumps(results, indent=2, default=str))
    size = out_path.stat().st_size / (1024 * 1024)
    print("    " + str(len(all_files)) + "/" + str(len(all_files)) +
          " done — " + str(len(results)) + " records (" +
          str(round(size, 1)) + "MB)     ")
    return results


# ── Reports ───────────────────────────────────────────────────────────────────

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


def folder_report(name, results):
    ok  = [r for r in results if r.get("quality_flag") == "ok"]
    exc = [r for r in results if r.get("quality_flag") == "excluded"]
    err = [r for r in results if r.get("quality_flag") == "error"]
    sep = "-" * 65

    print("\n" + sep)
    print("  [" + name + "]")
    print(sep)
    print("  Total: " + str(len(results)) +
          "  OK: " + str(len(ok)) +
          "  Excluded: " + str(len(exc)) +
          "  Errors: " + str(len(err)))

    if not ok:
        print("  No usable files.")
        return

    durs = [r["ffprobe"]["duration"] for r in ok if r.get("ffprobe")]
    h    = sum(durs) / 3600
    print("  Duration: " + str(round(h, 1)) + "h total" +
          " | mean " + str(round(np.mean(durs)/60, 1)) + "min" +
          " | min " + str(round(min(durs))) + "s" +
          " | max " + str(round(max(durs))) + "s")

    brs = [r["ffprobe"]["bitrate_kbps"] for r in ok
           if r.get("ffprobe") and r["ffprobe"].get("bitrate_kbps")]
    if brs:
        print("  Bitrate: mean " + str(round(np.mean(brs))) +
              " | min " + str(min(brs)) + " | max " + str(max(brs)) + " kbps")

    src = defaultdict(int)
    for r in ok:
        sr = r.get("ffprobe", {}).get("sample_rate")
        if sr:
            src[sr] += 1
    print("  Sample rates: " +
          "  ".join(str(k) + "Hz:" + str(v) for k, v in sorted(src.items())))

    enc_c = defaultdict(int)
    for r in ok:
        enc_c[(r.get("ffprobe") or {}).get("encoder") or "none"] += 1
    print("  Encoders:")
    for enc, c in sorted(enc_c.items(), key=lambda x: x[1], reverse=True)[:5]:
        print("    " + enc[:42].ljust(42) + str(c).rjust(6) +
              " (" + str(round(c/len(ok)*100, 1)) + "%)")

    print("\n  Band energies (model 3-channel bands):")
    print("    Band                         Mean      Std    Note")
    for band, (lo, hi) in BANDS.items():
        vals = get_vals(ok, band + "_energy_ratio")
        if vals:
            m, s = ms(vals)
            note = ""
            if band == "high" and m < 0.03:
                note = "low — phone/codec?"
            label = band + "(" + str(lo) + "-" + str(hi) + "Hz)"
            print("    " + label.ljust(28) +
                  str(round(m, 3)).rjust(7) + "   " +
                  str(round(s, 3)).rjust(7) + "  " + note)

    crest = get_vals(ok, "crest_factor_db")
    if crest:
        m, s = ms(crest)
        heavy = sum(1 for v in crest if v < 6)
        print("\n  Crest factor: mean " + str(round(m, 1)) +
              "dB  std " + str(round(s, 1)) +
              "dB  | <6dB (heavy compress): " +
              str(heavy) + " (" + str(round(heavy/len(crest)*100, 1)) + "%)")

    tempos = get_vals(ok, "tempo_bpm")
    if tempos:
        m, s = ms(tempos)
        print("  Tempo: mean " + str(round(m, 1)) +
              " BPM  std " + str(round(s, 1)))

    vocals = get_vals(ok, "has_vocals_rough")
    if vocals:
        print("  Vocals: " + str(round(np.mean(vocals)*100, 1)) +
              "% with vocals (rough detection)")

    seg_std = [r["segments"].get("seg_high_std")
               for r in ok if r.get("segments") and
               r["segments"].get("seg_high_std")]
    if seg_std:
        m, _ = ms(seg_std)
        note = "  suspicious consistency" if m < 0.02 else ""
        print("  Seg high-band std: " + str(round(m, 4)) + note)

    if exc:
        rc = defaultdict(int)
        for r in exc:
            for reason in r.get("exclusion_reasons", []):
                rc[reason] += 1
        print("\n  Exclusions:")
        for reason, c in sorted(rc.items(), key=lambda x: x[1], reverse=True):
            print("    " + reason.ljust(45) + str(c))


def balance_report(all_results):
    sep = "=" * 65
    print("\n" + sep)
    print("  BALANCE — Natural vs AI  (flag = potential confound)")
    print(sep + "\n")

    nat, ai = [], []
    for name, results in all_results.items():
        ok = [r for r in results if r.get("quality_flag") == "ok"]
        if "natural" in name.lower():
            nat.extend(ok)
        else:
            ai.extend(ok)

    if not nat:
        print("  No folder named 'natural' found.")
        return
    if not ai:
        print("  No AI folders found.")
        return

    print("  Natural: " + str(len(nat)) + "  AI: " + str(len(ai)) + "\n")
    print("  Feature                          Natural      AI      Diff  Flag")
    print("  " + "-" * 63)

    checks = [
        ("low_energy_ratio",  "low(20-500Hz)",    0.05),
        ("mid_energy_ratio",  "mid(300-4kHz)",    0.05),
        ("high_energy_ratio", "high(3k-11kHz)",   0.05),
        ("crest_factor_db",   "crest_factor_db",  3.0),
        ("tempo_bpm",         "tempo_bpm",         15.0),
        ("zcr_mean",          "zcr_mean",          0.02),
        ("flatness",          "flatness",          0.01),
    ]

    for key, label, thresh in checks:
        nv = get_vals(nat, key)
        av = get_vals(ai,  key)
        if nv and av:
            nm, _ = ms(nv)
            am, _ = ms(av)
            diff  = abs(nm - am)
            flag  = "CONFOUND" if diff > thresh else "ok"
            fmt   = ".1f" if abs(nm) > 5 else ".3f"
            print("  " + label.ljust(30) +
                  str(round(nm, 3)).rjust(10) +
                  str(round(am, 3)).rjust(10) +
                  str(round(diff, 3)).rjust(9) + "  " + flag)

    nv = get_vals(nat, "has_vocals_rough")
    av = get_vals(ai,  "has_vocals_rough")
    if nv and av:
        nm   = np.mean(nv) * 100
        am   = np.mean(av) * 100
        diff = abs(nm - am)
        flag = "CONFOUND" if diff > 15 else "ok"
        print("  " + "has_vocals_%".ljust(30) +
              str(round(nm, 1)).rjust(10) +
              str(round(am, 1)).rjust(10) +
              str(round(diff, 1)).rjust(9) + "  " + flag)

    print("\n  CONFOUND = systematic difference likely exploited by model")
    print("  Fix: balance dataset by style/genre or add targeted augmentation")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_dataset.py /path/to/dataset/")
        sys.exit(1)

    root    = Path(sys.argv[1])
    out_dir = Path("raw_metadata")

    # discover subfolders with audio
    subs = sorted([f for f in root.iterdir() if f.is_dir()])
    folders = {}
    for sf in subs:
        if any(f.suffix.lower() in AUDIO_EXT
               for f in sf.rglob("*") if f.is_file()):
            folders[sf.name] = sf
    if not folders:
        folders = {root.name: root}

    sep = "=" * 65
    print("\n" + sep)
    print("  AUDIO DATASET ANALYSIS")
    print("  Bands: low(20-500Hz) | mid(300-4kHz) | high(3k-11025Hz)")
    print("  SR: " + str(ANALYSIS_SR) + "Hz  Min bitrate: " +
          str(MIN_BITRATE) + "kbps  Saves every: " + str(SAVE_EVERY) + " files")
    print("  Folders: " + str(len(folders)))
    print(sep)

    all_results = {}
    all_flat    = []

    for name, path in folders.items():
        res = analyze_folder(path, out_dir)
        all_results[name] = res
        folder_report(name, res)
        all_flat.extend(res)

    # final summary
    total_ok  = sum(1 for r in all_flat if r.get("quality_flag") == "ok")
    total_exc = sum(1 for r in all_flat if r.get("quality_flag") == "excluded")
    total_err = sum(1 for r in all_flat if r.get("quality_flag") == "error")

    print("\n" + sep)
    print("  DONE")
    print(sep)
    for name, results in all_results.items():
        ok  = sum(1 for r in results if r.get("quality_flag") == "ok")
        exc = sum(1 for r in results if r.get("quality_flag") == "excluded")
        err = sum(1 for r in results if r.get("quality_flag") == "error")
        out_path = out_dir / (name + ".json")
        size     = out_path.stat().st_size / (1024*1024) if out_path.exists() else 0
        print("  " + name.ljust(25) + str(len(results)).rjust(7) +
              " records  OK:" + str(ok) +
              "  Excl:" + str(exc) +
              "  Err:" + str(err) +
              "  (" + str(round(size, 1)) + "MB)")
    print("-" * 65)
    print("  TOTAL" + " "*20 + str(len(all_flat)).rjust(7) +
          " records  OK:" + str(total_ok) +
          "  Excl:" + str(total_exc) +
          "  Err:" + str(total_err))
    print("  Output dir: " + str(out_dir.resolve()))
    print(sep + "\n")


if __name__ == "__main__":
    main()