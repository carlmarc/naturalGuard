"""
tests/test_pipeline.py — Integration smoke tests for the data pipeline.

PURPOSE
-------
These are integration-style checks, not unit tests. Each test validates a
complete pipeline artifact (manifest file, dataset record structure) rather
than a single function in isolation. The goal is to catch regressions when
pipeline scripts are re-run (e.g. after adding mixed examples).

Running:
    pytest tests/test_pipeline.py -v

The tests require the manifests to have been built (run build_splits.py first)
and the dataset root to be accessible at the default path. Override with:
    DATASET_ROOT=/path/to/data pytest tests/test_pipeline.py -v
"""

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent

SPLITS_DIR = REPO_ROOT / "data" / "splits"

DATASET_ROOT = Path(
    os.environ.get(
        "DATASET_ROOT",
        "/home/cma/Samsung990_2T/datasets/music_detection",
    )
)

# Canonical class list — must match SOURCEID_CLASSES in build_splits.py and
# SOURCEID_CLASSES in src/detector/dataset.py (once that file exists).
SOURCEID_CLASSES = [
    "human",
    "suno",
    "boomy",
    "mubert",
    "mureka",
    "musicgen",
    "stable_audio",
    "elevenlabsmusic",
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _load_manifest(name: str) -> dict:
    path = SPLITS_DIR / name
    if not path.exists():
        pytest.skip(f"Manifest not found: {path}. Run build_splits.py first.")
    with open(path) as f:
        return json.load(f)


def _check_entry_structure(entry: dict, manifest_name: str) -> None:
    """Assert that one manifest entry has the expected fields and types."""
    required = [
        "song_id",
        "source_class",
        "processed_path",
        "duration_sec",
        "labels",
        "vocal_class",
        "duration_bucket",
        "high_energy_bucket",
    ]
    for field in required:
        assert field in entry, f"{manifest_name}: entry missing field '{field}' — {entry.get('song_id')}"

    assert isinstance(entry["song_id"], str) and entry["song_id"], \
        f"{manifest_name}: song_id is empty"

    assert isinstance(entry["labels"], dict), \
        f"{manifest_name}: labels must be a dict"
    assert set(entry["labels"].keys()) == set(SOURCEID_CLASSES), \
        f"{manifest_name}: labels keys mismatch. Got {set(entry['labels'].keys())}"
    for cls, val in entry["labels"].items():
        assert val in (0, 1), \
            f"{manifest_name}: label value must be 0 or 1, got {val} for class '{cls}'"

    assert entry["vocal_class"] in ("voiced", "none"), \
        f"{manifest_name}: unexpected vocal_class '{entry['vocal_class']}'"
    assert entry["high_energy_bucket"] in ("low", "mid", "high"), \
        f"{manifest_name}: unexpected high_energy_bucket '{entry['high_energy_bucket']}'"

    assert isinstance(entry["processed_path"], str) and entry["processed_path"], \
        f"{manifest_name}: processed_path is empty"


# ---------------------------------------------------------------------------
# TESTS: naturalguard_pretrain.json
# ---------------------------------------------------------------------------


class TestNaturalguardPretrain:
    """Checks for naturalguard_pretrain.json (Phase 1 — natural-only contrastive)."""

    def setup_method(self):
        self.manifest = _load_manifest("naturalguard_pretrain.json")

    def test_top_level_structure(self):
        """Manifest must have an 'entries' list."""
        assert "entries" in self.manifest, "Missing 'entries' key"
        assert isinstance(self.manifest["entries"], list)

    def test_entry_count(self):
        """Must have at least 9000 natural entries (9627 expected currently)."""
        n = len(self.manifest["entries"])
        assert n >= 9000, f"Expected >= 9000 entries, got {n}"

    def test_entry_structure(self):
        """Spot-check 100 evenly-spaced entries for correct structure."""
        entries = self.manifest["entries"]
        step = max(1, len(entries) // 100)
        for entry in entries[::step]:
            _check_entry_structure(entry, "naturalguard_pretrain.json")

    def test_all_entries_are_natural(self):
        """Every entry must have source_class == 'natural' and human=1."""
        entries = self.manifest["entries"]
        bad_source = [e["song_id"] for e in entries if e["source_class"] != "natural"]
        bad_label  = [e["song_id"] for e in entries if e["labels"].get("human") != 1]
        assert not bad_source, f"Non-natural entries found: {bad_source[:5]}"
        assert not bad_label,  f"Entries without human=1: {bad_label[:5]}"

    def test_all_ai_labels_zero(self):
        """All AI class labels must be 0 for natural entries."""
        ai_classes = [c for c in SOURCEID_CLASSES if c != "human"]
        bad = []
        for e in self.manifest["entries"]:
            for cls in ai_classes:
                if e["labels"].get(cls) != 0:
                    bad.append((e["song_id"], cls))
        assert not bad, f"Natural entries with non-zero AI labels: {bad[:5]}"

    def test_no_duplicate_song_ids(self):
        """Song IDs must be unique within the manifest."""
        ids = [e["song_id"] for e in self.manifest["entries"]]
        assert len(ids) == len(set(ids)), "Duplicate song_ids in naturalguard_pretrain"

    def test_processed_paths_point_to_natural_dir(self):
        """Processed paths should all be under processed/train/natural/."""
        bad = [
            e["processed_path"]
            for e in self.manifest["entries"]
            if "natural" not in e["processed_path"]
        ]
        assert not bad, f"Entries with unexpected processed_path: {bad[:5]}"


# ---------------------------------------------------------------------------
# TESTS: sourceid_train.json
# ---------------------------------------------------------------------------


class TestSourceidTrain:
    """Checks for sourceid_train.json (Phase 2 — SourceID supervised training)."""

    def setup_method(self):
        self.manifest = _load_manifest("sourceid_train.json")

    def test_top_level_structure(self):
        """Must have class_names, pos_weight, and entries."""
        for key in ("class_names", "pos_weight", "entries"):
            assert key in self.manifest, f"Missing top-level key '{key}'"

    def test_class_names(self):
        """class_names must exactly match SOURCEID_CLASSES in the canonical order."""
        assert self.manifest["class_names"] == SOURCEID_CLASSES, \
            f"class_names mismatch: {self.manifest['class_names']}"

    def test_pos_weight_keys(self):
        """pos_weight must have one key per SourceID class."""
        assert set(self.manifest["pos_weight"].keys()) == set(SOURCEID_CLASSES), \
            f"pos_weight key mismatch: {set(self.manifest['pos_weight'].keys())}"

    def test_pos_weight_values_positive(self):
        """Every pos_weight value must be strictly positive."""
        for cls, w in self.manifest["pos_weight"].items():
            assert w > 0, f"pos_weight[{cls}] = {w} is not positive"

    def test_human_pos_weight_less_than_ai(self):
        """Human pos_weight must be < AI pos_weights because natural is the majority class."""
        hw = self.manifest["pos_weight"]["human"]
        for cls in SOURCEID_CLASSES:
            if cls == "human":
                continue
            assert hw < self.manifest["pos_weight"][cls], \
                f"Expected human pos_weight < {cls} pos_weight, got {hw} vs {self.manifest['pos_weight'][cls]}"

    def test_entry_count(self):
        """Must have >= 20000 train entries."""
        n = len(self.manifest["entries"])
        assert n >= 20000, f"Expected >= 20000 entries, got {n}"

    def test_entry_structure(self):
        """Spot-check 100 evenly-spaced entries for correct structure."""
        entries = self.manifest["entries"]
        step = max(1, len(entries) // 100)
        for entry in entries[::step]:
            _check_entry_structure(entry, "sourceid_train.json")

    def test_no_duplicate_song_ids(self):
        """Song IDs must be unique in the train manifest."""
        ids = [e["song_id"] for e in self.manifest["entries"]]
        assert len(ids) == len(set(ids)), "Duplicate song_ids in sourceid_train"

    def test_all_sources_present(self):
        """Every expected source class must appear at least once."""
        expected = {"natural"} | set(
            s for s in ["suno", "boomy", "mubert", "mureka", "musicgen", "stable_audio", "elevenlabsmusic"]
        )
        seen = {e["source_class"] for e in self.manifest["entries"]}
        missing = expected - seen
        assert not missing, f"Sources absent from train: {missing}"

    def test_each_entry_has_exactly_one_positive_label(self):
        """Pure-source entries must have exactly one positive label."""
        bad = []
        for e in self.manifest["entries"]:
            n_pos = sum(v for v in e["labels"].values())
            if n_pos != 1:
                bad.append((e["song_id"], n_pos))
        # Allow a small number to be mixed examples once build_mixed_examples runs.
        # For now (pure sources only), all must have exactly 1 positive.
        assert not bad, f"Entries with != 1 positive label: {bad[:5]}"

    def test_natural_entries_have_human_label(self):
        """natural source_class must have human=1."""
        bad = [
            e["song_id"]
            for e in self.manifest["entries"]
            if e["source_class"] == "natural" and e["labels"].get("human") != 1
        ]
        assert not bad, f"Natural entries without human=1: {bad[:5]}"

    def test_ai_entries_have_correct_label(self):
        """Each AI source must have its own class=1 and human=0."""
        bad = []
        for e in self.manifest["entries"]:
            src = e["source_class"]
            if src == "natural":
                continue
            if src not in e["labels"]:
                bad.append((e["song_id"], src, "class not in labels"))
            elif e["labels"][src] != 1:
                bad.append((e["song_id"], src, f"label={e['labels'][src]}"))
            elif e["labels"]["human"] != 0:
                bad.append((e["song_id"], src, f"human label={e['labels']['human']} (should be 0)"))
        assert not bad, f"AI entries with wrong labels: {bad[:5]}"

    def test_no_excluded_sources(self):
        """melodia, ldm2, melodia_test must not appear in train."""
        excluded = {"melodia", "ldm2", "melodia_test"}
        bad = [e["song_id"] for e in self.manifest["entries"] if e["source_class"] in excluded]
        assert not bad, f"Excluded sources found in train: {bad[:5]}"


# ---------------------------------------------------------------------------
# TESTS: sourceid_val.json
# ---------------------------------------------------------------------------


class TestSourceidVal:
    """Checks for sourceid_val.json (Phase 2 — SourceID validation set)."""

    def setup_method(self):
        self.manifest = _load_manifest("sourceid_val.json")

    def test_top_level_structure(self):
        for key in ("class_names", "pos_weight", "entries"):
            assert key in self.manifest, f"Missing top-level key '{key}'"

    def test_class_names(self):
        assert self.manifest["class_names"] == SOURCEID_CLASSES

    def test_entry_count(self):
        """Must have >= 2000 val entries (10% of ~26k total)."""
        n = len(self.manifest["entries"])
        assert n >= 2000, f"Expected >= 2000 val entries, got {n}"

    def test_entry_structure(self):
        entries = self.manifest["entries"]
        step = max(1, len(entries) // 100)
        for entry in entries[::step]:
            _check_entry_structure(entry, "sourceid_val.json")

    def test_no_duplicate_song_ids(self):
        ids = [e["song_id"] for e in self.manifest["entries"]]
        assert len(ids) == len(set(ids)), "Duplicate song_ids in sourceid_val"

    def test_val_pos_weight_matches_train(self):
        """Val manifest must carry the same pos_weight as train (computed on train)."""
        train = _load_manifest("sourceid_train.json")
        assert self.manifest["pos_weight"] == train["pos_weight"], \
            "Val pos_weight differs from train pos_weight — should be identical"

    def test_no_overlap_with_train(self):
        """No song_id should appear in both train and val."""
        train = _load_manifest("sourceid_train.json")
        train_ids = {e["song_id"] for e in train["entries"]}
        val_ids   = {e["song_id"] for e in self.manifest["entries"]}
        overlap = train_ids & val_ids
        assert not overlap, f"Train/val overlap: {len(overlap)} shared song_ids, e.g. {list(overlap)[:3]}"

    def test_val_is_roughly_ten_percent(self):
        """Val should be ~10% of total SourceID tracks."""
        train = _load_manifest("sourceid_train.json")
        total = len(train["entries"]) + len(self.manifest["entries"])
        pct = 100.0 * len(self.manifest["entries"]) / total
        assert 8.0 <= pct <= 12.0, \
            f"Expected val fraction near 10%, got {pct:.1f}%"

    def test_all_sources_present(self):
        expected = {"natural"} | set(["suno", "boomy", "mubert", "mureka", "musicgen", "stable_audio", "elevenlabsmusic"])
        seen = {e["source_class"] for e in self.manifest["entries"]}
        missing = expected - seen
        assert not missing, f"Sources absent from val: {missing}"


# ---------------------------------------------------------------------------
# TESTS: processing_state/train/mixed.json
# ---------------------------------------------------------------------------


MIXED_MANIFEST_PATH = DATASET_ROOT / "processing_state" / "train" / "mixed.json"
MIXED_EXPECTED_COMBOS = [
    f"{vd}_over_{src}"
    for vd in ["librispeech_devclean", "nus_48e_sing"]
    for src in ["suno", "boomy", "mubert", "mureka", "musicgen", "stable_audio", "elevenlabsmusic"]
]
MIXED_REQUIRED_MIX_PARAMS = [
    "vocal_gain_db", "vocal_shelf_db", "deess_db",
    "reverb_preset", "reverb_wet_db",
    "compression_target_db", "compression_threshold_db",
]


def _load_mixed_manifest() -> list[dict]:
    if not MIXED_MANIFEST_PATH.exists():
        pytest.skip(f"Mixed manifest not found: {MIXED_MANIFEST_PATH}. Run build_mixed_examples.py first.")
    with open(MIXED_MANIFEST_PATH) as f:
        return json.load(f)


class TestMixedManifest:
    """Checks for processing_state/train/mixed.json."""

    def setup_method(self):
        self.records = _load_mixed_manifest()
        self.ok = [r for r in self.records if r.get("status") == "ok"]

    def test_manifest_loads(self):
        """Manifest must be a non-empty JSON list."""
        assert isinstance(self.records, list), "Manifest must be a list"
        assert len(self.records) > 0, "Manifest must be non-empty"

    def test_ok_count(self):
        """Must have at least 1400 ok records (1500 target, allowing some failures)."""
        assert len(self.ok) >= 1400, f"Expected >= 1400 ok records, got {len(self.ok)}"

    def test_all_ok(self):
        """In a clean run there should be zero failures."""
        failed = [r for r in self.records if r.get("status") != "ok"]
        assert not failed, f"{len(failed)} failed records: {[r.get('song_id') for r in failed[:3]]}"

    def test_record_has_required_fields(self):
        """Spot-check 50 records for mandatory fields."""
        required = [
            "song_id", "status", "idx", "combo",
            "vocal_dataset", "vocal_speaker",
            "instrumental_source", "instrumental_song_id",
            "processed_path", "labels", "mix_params",
        ]
        step = max(1, len(self.ok) // 50)
        for rec in self.ok[::step]:
            for field in required:
                assert field in rec, f"Record {rec.get('song_id')} missing field '{field}'"

    def test_mix_params_keys(self):
        """mix_params must contain all expected keys."""
        step = max(1, len(self.ok) // 50)
        for rec in self.ok[::step]:
            params = rec.get("mix_params", {})
            for key in MIXED_REQUIRED_MIX_PARAMS:
                assert key in params, f"mix_params missing '{key}' in {rec['song_id']}"

    def test_labels_are_two_hot(self):
        """Each mixed record must have exactly human=1 and one AI class=1."""
        bad = []
        for rec in self.ok:
            labels = rec.get("labels", {})
            if labels.get("human") != 1:
                bad.append((rec["song_id"], "human != 1"))
                continue
            ai_positives = [cls for cls in SOURCEID_CLASSES if cls != "human" and labels.get(cls) == 1]
            if len(ai_positives) != 1:
                bad.append((rec["song_id"], f"ai_positives={ai_positives}"))
        assert not bad, f"Labels wrong in {len(bad)} records: {bad[:3]}"

    def test_ai_label_matches_instrumental_source(self):
        """The AI class that is 1 must match instrumental_source."""
        bad = []
        for rec in self.ok:
            src = rec.get("instrumental_source")
            labels = rec.get("labels", {})
            if labels.get(src) != 1:
                bad.append((rec["song_id"], src, labels.get(src)))
        assert not bad, f"instrumental_source/label mismatch: {bad[:3]}"

    def test_all_combos_present(self):
        """All 14 (vocal_dataset, ai_source) combos must appear at least once."""
        seen_combos = {r["combo"] for r in self.ok}
        missing = set(MIXED_EXPECTED_COMBOS) - seen_combos
        assert not missing, f"Missing combos: {missing}"

    def test_output_files_exist(self):
        """Spot-check 20 processed_path files actually exist on disk."""
        step = max(1, len(self.ok) // 20)
        missing = []
        for rec in self.ok[::step]:
            p = DATASET_ROOT / rec["processed_path"]
            if not p.exists():
                missing.append(rec["processed_path"])
        assert not missing, f"Missing output files: {missing[:5]}"

    def test_output_duration_is_seven_seconds(self):
        """Spot-check 5 files: ffprobe duration must be between 6.9 and 7.2 seconds.

        The slight overshoot (7.053 s typical) comes from MP3's frame-boundary
        encoding — the decoder emits full frames, so the last frame may extend
        slightly past the 7.0 s content.
        """
        import subprocess
        step = max(1, len(self.ok) // 5)
        bad = []
        for rec in self.ok[::step]:
            p = DATASET_ROOT / rec["processed_path"]
            if not p.exists():
                continue
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", str(p)],
                capture_output=True,
            )
            if result.returncode != 0:
                bad.append((rec["processed_path"], "ffprobe failed"))
                continue
            info = json.loads(result.stdout)
            duration = float(info["streams"][0].get("duration", 0))
            if not (6.9 <= duration <= 7.2):
                bad.append((rec["processed_path"], f"duration={duration:.3f}s"))
        assert not bad, f"Unexpected durations: {bad}"

    def test_no_duplicate_song_ids(self):
        """song_ids must be unique in the manifest."""
        ids = [r["song_id"] for r in self.ok]
        assert len(ids) == len(set(ids)), "Duplicate song_ids in mixed manifest"

    def test_reverb_preset_values_valid(self):
        """reverb_preset must be one of tight / loose / none."""
        valid = {"tight", "loose", "none"}
        bad = [
            r["song_id"]
            for r in self.ok
            if r["mix_params"].get("reverb_preset") not in valid
        ]
        assert not bad, f"Invalid reverb_preset in: {bad[:5]}"

    def test_vocal_gain_in_range(self):
        """vocal_gain_db must be within ±6 dB."""
        bad = [
            (r["song_id"], r["mix_params"]["vocal_gain_db"])
            for r in self.ok
            if not (-6.0 <= r["mix_params"]["vocal_gain_db"] <= 6.0)
        ]
        assert not bad, f"vocal_gain_db out of range: {bad[:5]}"
