"""
Submission CSV generator and validator.

Produces the final submission CSV and validates it against all spec rules.
"""

import csv
import logging
from pathlib import Path
from typing import Any

from src.config import TOP_K, OUTPUT_DIR

logger = logging.getLogger(__name__)

FeatureDict = dict[str, Any]


def generate_submission(
    ranked_features: list[FeatureDict],
    reasonings: list[str],
    output_path: Path | None = None,
) -> Path:
    """
    Generate the submission CSV.

    Args:
        ranked_features: Feature dicts sorted by final_score descending.
                         Must contain exactly TOP_K entries.
        reasonings: Corresponding reasoning strings.
        output_path: Path for the CSV file. Default: OUTPUT_DIR/submission.csv.

    Returns:
        Path to the generated CSV.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "submission.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Trim to TOP_K
    actual_count = min(len(ranked_features), TOP_K)
    ranked_features = ranked_features[:actual_count]
    reasonings = reasonings[:actual_count]

    if actual_count < TOP_K:
        logger.warning(
            "Only %d candidates available (need %d for full submission).",
            actual_count, TOP_K,
        )

    # Ensure scores are monotonically non-increasing
    scores = [f["final_score"] for f in ranked_features]
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            # Force monotonicity
            scores[i] = scores[i - 1]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for i, (features, reasoning) in enumerate(zip(ranked_features, reasonings)):
            rank = i + 1
            candidate_id = features["candidate_id"]
            score = scores[i]

            # Escape reasoning for CSV
            reasoning_clean = reasoning.replace('"', "'").replace("\n", " ").strip()

            writer.writerow([candidate_id, rank, f"{score:.6f}", reasoning_clean])

    logger.info("Submission CSV written: %s (%d rows)", output_path, actual_count)
    return output_path


def validate_submission(
    csv_path: Path,
    valid_candidate_ids: set[str] | None = None,
) -> list[str]:
    """
    Validate a submission CSV against all spec rules.

    Returns:
        List of error messages. Empty list = valid submission.
    """
    errors: list[str] = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Check header
            expected_cols = {"candidate_id", "rank", "score", "reasoning"}
            if reader.fieldnames is None:
                errors.append("No header row found")
                return errors

            actual_cols = set(reader.fieldnames)
            missing = expected_cols - actual_cols
            if missing:
                errors.append(f"Missing columns: {missing}")

            rows = list(reader)

    except Exception as e:
        errors.append(f"Cannot read CSV: {e}")
        return errors

    # Rule: exactly 100 rows
    if len(rows) != TOP_K:
        errors.append(f"Expected {TOP_K} rows, got {len(rows)}")

    # Collect values for validation
    candidate_ids: list[str] = []
    ranks: list[int] = []
    scores_list: list[float] = []

    for i, row in enumerate(rows):
        cid = row.get("candidate_id", "")
        candidate_ids.append(cid)

        try:
            rank = int(row.get("rank", 0))
            ranks.append(rank)
        except ValueError:
            errors.append(f"Row {i+1}: invalid rank '{row.get('rank')}'")
            continue

        try:
            score = float(row.get("score", 0))
            scores_list.append(score)
        except ValueError:
            errors.append(f"Row {i+1}: invalid score '{row.get('score')}'")
            continue

    # Rule: each rank 1-100 exactly once
    expected_ranks = set(range(1, TOP_K + 1))
    actual_ranks = set(ranks)
    if actual_ranks != expected_ranks:
        missing_ranks = expected_ranks - actual_ranks
        extra_ranks = actual_ranks - expected_ranks
        if missing_ranks:
            errors.append(f"Missing ranks: {sorted(missing_ranks)[:10]}")
        if extra_ranks:
            errors.append(f"Extra ranks: {sorted(extra_ranks)[:10]}")

    # Rule: each candidate_id exactly once
    if len(set(candidate_ids)) != len(candidate_ids):
        seen: dict[str, int] = {}
        for cid in candidate_ids:
            seen[cid] = seen.get(cid, 0) + 1
        dupes = {k: v for k, v in seen.items() if v > 1}
        errors.append(f"Duplicate candidate_ids: {dupes}")

    # Rule: candidate_ids exist in dataset
    if valid_candidate_ids:
        invalid = [cid for cid in candidate_ids if cid not in valid_candidate_ids]
        if invalid:
            errors.append(f"Invalid candidate_ids: {invalid[:10]}")

    # Rule: scores non-increasing with rank
    if len(scores_list) >= 2:
        for i in range(1, len(scores_list)):
            if scores_list[i] > scores_list[i - 1]:
                errors.append(
                    f"Score increases at rank {ranks[i]}: "
                    f"{scores_list[i]} > {scores_list[i-1]}"
                )
                break

    # Rule: ranks start at 1
    if ranks and min(ranks) == 0:
        errors.append("Ranks start at 0 instead of 1")

    # Rule: not all same score
    if len(set(scores_list)) == 1:
        errors.append("All scores are identical")

    # Log results
    if errors:
        for err in errors:
            logger.error("Validation error: %s", err)
    else:
        logger.info("Submission validation PASSED: %s", csv_path)

    return errors
