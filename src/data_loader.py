"""
Data loader for candidate profiles.

Handles loading from JSONL (plain or gzipped) and JSON formats.
Validates candidate structure and reports loading statistics.
"""

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import (
    CANDIDATES_JSONL,
    CANDIDATES_JSONL_GZ,
    SAMPLE_CANDIDATES,
)

logger = logging.getLogger(__name__)

CandidateRecord = dict[str, Any]


def load_candidates(
    path: Path | None = None,
    max_candidates: int | None = None,
) -> list[CandidateRecord]:
    """
    Load candidate records from a JSONL, JSONL.GZ, or JSON file.

    Automatically detects the file format based on extension.
    Returns a list of parsed candidate dicts.

    Args:
        path: Path to the candidates file. If None, auto-detects from config.
        max_candidates: If set, stop loading after this many candidates.

    Returns:
        List of candidate dicts, each with structure matching candidate_schema.json.

    Raises:
        FileNotFoundError: If no candidates file can be found.
        ValueError: If a record is missing candidate_id.
    """
    if path is None:
        path = _find_candidates_file()

    logger.info("Loading candidates from %s", path)
    start_time = time.time()

    if path.suffix == ".json":
        candidates = _load_json(path, max_candidates)
    elif path.name.endswith(".jsonl.gz"):
        candidates = _load_jsonl_gz(path, max_candidates)
    elif path.suffix == ".jsonl":
        candidates = _load_jsonl(path, max_candidates)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    elapsed = time.time() - start_time
    logger.info(
        "Loaded %d candidates in %.2fs (%.1f candidates/sec)",
        len(candidates),
        elapsed,
        len(candidates) / elapsed if elapsed > 0 else 0,
    )

    _validate_candidates(candidates)
    return candidates


def load_sample_candidates() -> list[CandidateRecord]:
    """Load the sample_candidates.json file (50 candidates for testing)."""
    return load_candidates(SAMPLE_CANDIDATES)


def _find_candidates_file() -> Path:
    """Auto-detect the candidates file path from config."""
    if CANDIDATES_JSONL.exists():
        return CANDIDATES_JSONL
    if CANDIDATES_JSONL_GZ.exists():
        return CANDIDATES_JSONL_GZ
    if SAMPLE_CANDIDATES.exists():
        logger.warning(
            "Full candidates file not found. Falling back to sample_candidates.json"
        )
        return SAMPLE_CANDIDATES
    raise FileNotFoundError(
        f"No candidates file found. Searched:\n"
        f"  {CANDIDATES_JSONL}\n"
        f"  {CANDIDATES_JSONL_GZ}\n"
        f"  {SAMPLE_CANDIDATES}"
    )


def _load_json(path: Path, max_candidates: int | None) -> list[CandidateRecord]:
    """Load from a JSON array file (e.g., sample_candidates.json)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    if max_candidates is not None:
        data = data[:max_candidates]
    return data


def _load_jsonl(path: Path, max_candidates: int | None) -> list[CandidateRecord]:
    """Load from a plain JSONL file (one JSON object per line)."""
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_candidates is not None and i >= max_candidates:
                break
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    return candidates


def _load_jsonl_gz(path: Path, max_candidates: int | None) -> list[CandidateRecord]:
    """Load from a gzipped JSONL file."""
    candidates = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_candidates is not None and i >= max_candidates:
                break
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    return candidates


def _validate_candidates(candidates: list[CandidateRecord]) -> None:
    """
    Validate loaded candidates for structural integrity.

    Checks:
    - Each record has a candidate_id
    - No duplicate candidate_ids
    - Each record has a profile section
    """
    if not candidates:
        raise ValueError("No candidates loaded")

    ids_seen: set[str] = set()
    missing_id_count = 0
    missing_profile_count = 0
    duplicate_count = 0

    for c in candidates:
        cid = c.get("candidate_id")
        if cid is None:
            missing_id_count += 1
            continue

        if cid in ids_seen:
            duplicate_count += 1
        ids_seen.add(cid)

        if "profile" not in c:
            missing_profile_count += 1

    if missing_id_count > 0:
        logger.error("%d candidates missing candidate_id", missing_id_count)

    if duplicate_count > 0:
        logger.warning("%d duplicate candidate_ids found", duplicate_count)

    if missing_profile_count > 0:
        logger.warning("%d candidates missing profile section", missing_profile_count)

    logger.info(
        "Validation: %d total, %d unique IDs, %d duplicates, %d missing profile",
        len(candidates),
        len(ids_seen),
        duplicate_count,
        missing_profile_count,
    )
