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
