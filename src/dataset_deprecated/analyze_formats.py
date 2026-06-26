#!/usr/bin/env python3
"""
Dataset Audio Format Analyzer
Scans dataset folders and reports format, bitrate, sample rate distribution.
Usage: python analyze_formats.py /path/to/dataset
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict


AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.aiff', '.opus'}


def get_audio_info(filepath):
    """Extract audio metadata using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams', '-show_format',
        str(filepath)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)

        # find audio stream
        audio_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break

        if not audio_stream:
            return None

        fmt = data.get('format', {})

        # bitrate — prefer stream bitrate, fall back to format bitrate
        bitrate_bps = (
            int(audio_stream.get('bit_rate', 0)) or
            int(fmt.get('bit_rate', 0))
        )
        bitrate_kbps = round(bitrate_bps / 1000) if bitrate_bps else None

        # sample rate
        sample_rate = int(audio_stream.get('sample_rate', 0)) or None

        # channels
        channels = audio_stream.get('channels', None)

        # codec
        codec = audio_stream.get('codec_name', 'unknown')

        # duration
        duration = float(fmt.get('duration', 0) or audio_stream.get('duration', 0))

        # encoder tag (important for LAME detection)
        tags = fmt.get('tags', {})
        encoder = tags.get('encoder', tags.get('ENCODER', '')).strip()

        # file format from extension
        ext = Path(filepath).suffix.lower()

        return {
            'ext': ext,
            'codec': codec,
            'bitrate_kbps': bitrate_kbps,
            'sample_rate': sample_rate,
            'channels': channels,
            'duration': duration,
            'encoder': encoder,
        }

    except Exception as e:
        return None


def bucket_bitrate(kbps):
    """Group bitrate into readable buckets."""
    if kbps is None:
        return 'unknown'
    if kbps < 96:
        return '<96kbps'
    elif kbps < 128:
        return '96-127kbps'
    elif kbps <= 128:
        return '128kbps'
    elif kbps <= 192:
        return '129-192kbps'
    elif kbps <= 256:
        return '193-256kbps'
    elif kbps <= 320:
        return '257-320kbps'
    else:
        return '>320kbps'


def bucket_samplerate(hz):
    """Group sample rate into readable buckets."""
    if hz is None:
        return 'unknown'
    if hz <= 22050:
        return f'{hz}Hz'
    elif hz <= 44100:
        return f'{hz}Hz'
    elif hz <= 48000:
        return f'{hz}Hz'
    else:
        return f'{hz}Hz'


def analyze_folder(folder_path):
    """Analyze all audio files in a folder recursively."""
    folder_path = Path(folder_path)
    
    stats = {
        'total_files': 0,
        'total_duration_hours': 0.0,
        'errors': 0,
        'by_format': defaultdict(int),           # ext → count
        'by_codec': defaultdict(int),            # codec → count
        'by_bitrate': defaultdict(int),          # bitrate bucket → count
        'by_samplerate': defaultdict(int),       # sample rate → count
        'by_channels': defaultdict(int),         # channels → count
        'by_encoder': defaultdict(int),          # encoder tag → count
        'below_128kbps': 0,                      # files below minimum quality
        'lame_encoded': 0,                       # LAME MP3s
        'non_lame_mp3': 0,                       # MP3s with other encoder
        'combinations': defaultdict(int),        # format+bitrate+samplerate → count
    }

    audio_files = [
        f for f in folder_path.rglob('*')
        if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file()
    ]

    if not audio_files:
        return stats

    print(f"    Scanning {len(audio_files)} files...", end='', flush=True)

    for i, filepath in enumerate(audio_files):
        if i % 50 == 0 and i > 0:
            print(f"{i}...", end='', flush=True)

        info = get_audio_info(filepath)

        if info is None:
            stats['errors'] += 1
            continue

        stats['total_files'] += 1
        stats['total_duration_hours'] += info['duration'] / 3600

        # format
        stats['by_format'][info['ext']] += 1
        stats['by_codec'][info['codec']] += 1

        # bitrate
        br_bucket = bucket_bitrate(info['bitrate_kbps'])
        stats['by_bitrate'][br_bucket] += 1
        if info['bitrate_kbps'] and info['bitrate_kbps'] < 128:
            stats['below_128kbps'] += 1

        # sample rate
        sr_bucket = bucket_samplerate(info['sample_rate'])
        stats['by_samplerate'][sr_bucket] += 1

        # channels
        ch = info['channels']
        stats['by_channels'][str(ch) if ch else 'unknown'] += 1

        # encoder
        enc = info['encoder'] or 'none'
        if info['ext'] == '.mp3':
            if 'LAME' in enc.upper():
                stats['lame_encoded'] += 1
            else:
                stats['non_lame_mp3'] += 1

        short_enc = enc[:30] if enc != 'none' else 'none'
        stats['by_encoder'][short_enc] += 1

        # combination summary
        combo = f"{info['ext']}/{br_bucket}/{info['sample_rate']}Hz"
        stats['combinations'][combo] += 1

    print(" done.")
    return stats


def print_dict_sorted(d, total, label, top_n=10):
    """Print a counter dict sorted by count."""
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for key, count in sorted_items:
        pct = (count / total * 100) if total > 0 else 0
        print(f"      {key:<30} {count:>6} files  ({pct:.1f}%)")


def print_folder_report(folder_name, stats):
    """Print formatted report for one folder."""
    total = stats['total_files']
    hours = stats['total_duration_hours']

    print(f"\n  {'─'*60}")
    print(f"  📁 {folder_name}")
    print(f"  {'─'*60}")
    print(f"    Total files:       {total}")
    print(f"    Total duration:    {hours:.1f} hours  ({hours*60:.0f} minutes)")
    print(f"    Errors/skipped:    {stats['errors']}")

    if total == 0:
        print("    (no audio files found)")
        return

    print(f"\n    Format (extension):")
    print_dict_sorted(stats['by_format'], total, 'format')

    print(f"\n    Bitrate distribution:")
    print_dict_sorted(stats['by_bitrate'], total, 'bitrate')

    if stats['below_128kbps'] > 0:
        print(f"    ⚠️  Below 128kbps (minimum quality): {stats['below_128kbps']} files")

    print(f"\n    Sample rate:")
    print_dict_sorted(stats['by_samplerate'], total, 'samplerate')

    print(f"\n    Channels:")
    print_dict_sorted(stats['by_channels'], total, 'channels')

    mp3_count = stats['by_format'].get('.mp3', 0)
    if mp3_count > 0:
        print(f"\n    MP3 encoder breakdown ({mp3_count} MP3 files):")
        print(f"      LAME encoded:      {stats['lame_encoded']:>6} ({stats['lame_encoded']/mp3_count*100:.1f}%)")
        print(f"      Non-LAME encoder:  {stats['non_lame_mp3']:>6} ({stats['non_lame_mp3']/mp3_count*100:.1f}%)")

    print(f"\n    Top combinations (format/bitrate/samplerate):")
    print_dict_sorted(stats['combinations'], total, 'combo', top_n=8)


def print_recommendation(all_stats):
    """Print unified preprocessing recommendation based on findings."""
    print(f"\n{'═'*62}")
    print("  PREPROCESSING RECOMMENDATION")
    print(f"{'═'*62}")

    total_files = sum(s['total_files'] for s in all_stats.values())
    total_hours = sum(s['total_duration_hours'] for s in all_stats.values())
    total_below_128 = sum(s['below_128kbps'] for s in all_stats.values())
    total_lame = sum(s['lame_encoded'] for s in all_stats.values())
    total_non_lame_mp3 = sum(s['non_lame_mp3'] for s in all_stats.values())

    all_formats = defaultdict(int)
    all_bitrates = defaultdict(int)
    all_samplerates = defaultdict(int)
    for s in all_stats.values():
        for k, v in s['by_format'].items():
            all_formats[k] += v
        for k, v in s['by_bitrate'].items():
            all_bitrates[k] += v
        for k, v in s['by_samplerate'].items():
            all_samplerates[k] += v

    print(f"\n  Dataset overview:")
    print(f"    Total files:  {total_files}")
    print(f"    Total hours:  {total_hours:.1f}h")

    print(f"\n  Format mix:")
    for fmt, count in sorted(all_formats.items(), key=lambda x: x[1], reverse=True):
        print(f"    {fmt:<10} {count:>6} files ({count/total_files*100:.1f}%)")

    print(f"\n  Bitrate mix:")
    for br, count in sorted(all_bitrates.items(), key=lambda x: x[1], reverse=True):
        print(f"    {br:<20} {count:>6} files ({count/total_files*100:.1f}%)")

    print(f"\n  Action per file type:")
    wav_count = all_formats.get('.wav', 0) + all_formats.get('.aiff', 0)
    flac_count = all_formats.get('.flac', 0)
    mp3_count = all_formats.get('.mp3', 0)

    if wav_count > 0:
        print(f"    WAV/AIFF ({wav_count} files):")
        print(f"      → encode to 192kbps LAME MP3 → decode to WAV 22050Hz")

    if flac_count > 0:
        print(f"    FLAC ({flac_count} files):")
        print(f"      → encode to 192kbps LAME MP3 → decode to WAV 22050Hz")

    if mp3_count > 0:
        print(f"    MP3 ({mp3_count} files):")
        if total_lame > 0:
            print(f"      LAME encoded ({total_lame} files):")
            print(f"        ≥192kbps → decode directly to WAV 22050Hz (no re-encode)")
            print(f"        <192kbps → keep as-is if ≥128kbps, decode to WAV 22050Hz")
        if total_non_lame_mp3 > 0:
            print(f"      Non-LAME ({total_non_lame_mp3} files):")
            print(f"        → re-encode to 192kbps LAME → decode to WAV 22050Hz")

    if total_below_128 > 0:
        print(f"\n  ⚠️  {total_below_128} files below 128kbps minimum — exclude from training")

    sr_dominant = max(all_samplerates.items(), key=lambda x: x[1])[0] if all_samplerates else 'unknown'
    non_22050 = sum(v for k, v in all_samplerates.items() if '22050' not in k)
    if non_22050 > 0:
        print(f"\n  Sample rate: {non_22050} files not at 22050Hz → all resampled to 22050Hz in preprocessing")

    estimated_wav_gb = (total_hours * 3600 * 22050 * 2) / (1024**3)
    print(f"\n  Estimated storage after preprocessing (WAV 22050Hz 16-bit):")
    print(f"    ~{estimated_wav_gb:.1f} GB")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_formats.py /path/to/dataset")
        print("       python analyze_formats.py /path/to/dataset/folder1 /path/to/dataset/folder2")
        sys.exit(1)

    dataset_paths = [Path(p) for p in sys.argv[1:]]

    # if single path and it has subfolders, analyze each subfolder
    if len(dataset_paths) == 1 and dataset_paths[0].is_dir():
        root = dataset_paths[0]
        subfolders = [f for f in sorted(root.iterdir()) if f.is_dir()]

        if subfolders:
            # check if subfolders contain audio directly or have further structure
            folders_to_analyze = {}
            for sf in subfolders:
                has_audio = any(
                    f.suffix.lower() in AUDIO_EXTENSIONS
                    for f in sf.rglob('*') if f.is_file()
                )
                if has_audio:
                    folders_to_analyze[sf.name] = sf
            
            if not folders_to_analyze:
                # no audio in subfolders, analyze root directly
                folders_to_analyze = {root.name: root}
        else:
            folders_to_analyze = {root.name: root}
    else:
        folders_to_analyze = {p.name: p for p in dataset_paths if p.is_dir()}

    print(f"\n{'═'*62}")
    print("  AUDIO DATASET FORMAT ANALYZER")
    print(f"{'═'*62}")
    print(f"  Analyzing {len(folders_to_analyze)} folder(s)...\n")

    all_stats = {}
    for folder_name, folder_path in folders_to_analyze.items():
        print(f"  Scanning: {folder_path}")
        stats = analyze_folder(folder_path)
        all_stats[folder_name] = stats
        print_folder_report(folder_name, stats)

    if len(all_stats) > 1:
        print_recommendation(all_stats)
    elif len(all_stats) == 1:
        print_recommendation(all_stats)

    # save raw stats to JSON
    output_path = Path('dataset_format_analysis.json')
    serializable = {
        k: {
            kk: dict(vv) if isinstance(vv, defaultdict) else vv
            for kk, vv in v.items()
        }
        for k, v in all_stats.items()
    }
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Raw stats saved to: {output_path}")


if __name__ == '__main__':
    main()
